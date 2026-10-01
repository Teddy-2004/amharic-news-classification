import argparse
import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from data_utils import load_clean_splits

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")


def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train, val, test = load_clean_splits()
    eval_df = test if test is not None else val
    eval_name = "test" if test is not None else "validation"
    print(f"Evaluating on: {eval_name} split ({len(eval_df)} rows)")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train["text"])
    X_eval = vectorizer.transform(eval_df["text"])
    y_train, y_eval = train["label"], eval_df["label"]

    if args.model == "lr":
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=args.C, n_jobs=-1
        )
    elif args.model == "svm":
        clf = LinearSVC(class_weight="balanced", C=args.C)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Fitting {args.model} on {X_train.shape[0]} examples, "
          f"{X_train.shape[1]} TF-IDF features...")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_eval)

    acc = accuracy_score(y_eval, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_eval, preds, average="macro", zero_division=0
    )
    report = classification_report(y_eval, preds, zero_division=0)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}\n")
    print(report)

    run_name = f"baseline_{args.model}"
    metrics = {
        "run_name": run_name,
        "approach": f"TF-IDF (ngram 1-{args.ngram_max}, max_features={args.max_features}) + "
                    f"{'Logistic Regression' if args.model == 'lr' else 'Linear SVM'}",
        "eval_split": eval_name,
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "config": vars(args),
    }
    with open(os.path.join(RESULTS_DIR, f"{run_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    preds_df = eval_df.copy()
    preds_df["predicted_label"] = preds
    preds_df.to_csv(os.path.join(RESULTS_DIR, f"{run_name}_preds.csv"), index=False)

    print(f"\nSaved metrics -> experiments/results/{run_name}_metrics.json")
    print(f"Saved predictions -> experiments/results/{run_name}_preds.csv")
    print(
        f"\n>>> Paste into experiments/experiments_log.md:\n"
        f"| {run_name} | {metrics['approach']} | acc={acc:.4f}, macro-F1={f1:.4f} | "
        f"<your note on what you learned> |"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lr", "svm"], default="lr")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-max", type=int, default=1)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
