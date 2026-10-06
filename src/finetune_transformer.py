import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset


class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies inverse-frequency class weights to the
    loss. HF's base Trainer has no built-in support for this, so we override
    compute_loss. Used when --class-weights is passed — important given
    real-world label imbalance (e.g. a 33x gap between the largest and
    smallest class is common in scraped news data)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

from data_utils import load_clean_splits

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def slugify(model_name: str) -> str:
    return model_name.replace("/", "__")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    slug = slugify(args.model)
    model_out_dir = os.path.join(MODELS_DIR, f"transformer_{slug}")
    os.makedirs(model_out_dir, exist_ok=True)

    train, val, test = load_clean_splits()
    if val is None:
        raise RuntimeError(
            "No validation split found. Re-run data/download_data.py."
        )
    eval_df = test if test is not None else val

    labels_sorted = sorted(train["label"].unique())
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    id2label = {i: l for l, i in label2id.items()}

    print(f"Loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    def to_hf_dataset(df):
        ds = HFDataset.from_pandas(
            df[["text", "label"]].rename(columns={"label": "label_str"})
        )
        ds = ds.map(lambda ex: {"labels": label2id[ex["label_str"]]})
        ds = ds.map(
            lambda ex: tokenizer(
                ex["text"], truncation=True, max_length=args.max_length,
                padding="max_length",
            ),
            batched=True,
        )
        return ds

    print("Tokenizing splits (this can take a few minutes for 40k+ rows)...")
    train_ds = to_hf_dataset(train)
    val_ds = to_hf_dataset(val)
    eval_ds = to_hf_dataset(eval_df)

    training_args = TrainingArguments(
        output_dir=os.path.join(model_out_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=args.fp16,
        report_to="none",
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if args.class_weights:
        counts = train["label"].value_counts()
        freq = np.array([counts[id2label[i]] for i in range(len(label2id))])
        weights = freq.sum() / (len(freq) * freq)
        weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Using class weights (inverse frequency): "
              f"{dict(zip(labels_sorted, weights.numpy().round(2)))}")
        trainer = WeightedLossTrainer(class_weights=weights, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    print("Starting fine-tuning...")
    trainer.train()

    print("Evaluating on held-out eval split...")
    eval_results = trainer.evaluate(eval_ds)
    print(eval_results)

    preds_output = trainer.predict(eval_ds)
    pred_ids = np.argmax(preds_output.predictions, axis=1)
    pred_labels = [id2label[i] for i in pred_ids]

    trainer.save_model(model_out_dir)
    tokenizer.save_pretrained(model_out_dir)

    run_name = f"transformer_{slug}" + ("_weighted" if args.class_weights else "")
    metrics = {
        "run_name": run_name,
        "approach": f"Fine-tuned {args.model} (max_len={args.max_length}, "
                    f"epochs={args.epochs}, lr={args.lr})",
        "eval_split": "test" if test is not None else "validation",
        "accuracy": eval_results["eval_accuracy"],
        "macro_precision": eval_results["eval_macro_precision"],
        "macro_recall": eval_results["eval_macro_recall"],
        "macro_f1": eval_results["eval_macro_f1"],
        "config": vars(args),
    }
    with open(os.path.join(RESULTS_DIR, f"{run_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    preds_df = eval_df.copy()
    preds_df["predicted_label"] = pred_labels
    preds_df.to_csv(os.path.join(RESULTS_DIR, f"{run_name}_preds.csv"), index=False)

    print(f"\nSaved metrics -> experiments/results/{run_name}_metrics.json")
    print(f"Saved model -> {model_out_dir}")
    print(
        f"\n>>> Paste into experiments/experiments_log.md:\n"
        f"| {run_name} | {metrics['approach']} | "
        f"acc={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f} | "
        f"<your note on what you learned> |"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="rasyosef/bert-medium-amharic",
        help="HF model id, e.g. rasyosef/bert-medium-amharic or xlm-roberta-base",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true",
                         help="Enable mixed precision (requires a CUDA GPU)")
    parser.add_argument("--class-weights", action="store_true",
                         help="Use inverse-frequency class weighting in the loss "
                              "(recommended given label imbalance — try both "
                              "with and without as an experiment)")
    args = parser.parse_args()
    main(args)