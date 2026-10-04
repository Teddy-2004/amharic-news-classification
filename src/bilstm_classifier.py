import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_utils import load_clean_splits

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

PAD, UNK = "<pad>", "<unk>"


def simple_tokenize(text: str):
    return text.split()


def build_vocab(texts, min_freq=2, max_size=30000):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    vocab = [PAD, UNK] + [
        w for w, c in counter.most_common(max_size) if c >= min_freq
    ]
    return {w: i for i, w in enumerate(vocab)}


def encode(text, vocab, max_len):
    ids = [vocab.get(tok, vocab[UNK]) for tok in simple_tokenize(text)][:max_len]
    ids = ids + [vocab[PAD]] * (max_len - len(ids))
    return ids


class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2id, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        label = self.label2id[self.labels[idx]]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes,
                 n_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        emb = self.embedding(x)                     # (B, T, E)
        out, (h_n, _) = self.lstm(emb)               # h_n: (2*layers, B, H)
        # concat final forward + backward hidden states of last layer
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)          # (B, 2H)
        h = self.dropout(h)
        return self.fc(h)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)


def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train, val, test = load_clean_splits()
    if val is None:
        raise RuntimeError(
            "No validation split found. Re-run data/download_data.py "
            "(it auto-creates one from train if missing)."
        )
    eval_df = test if test is not None else val

    labels_sorted = sorted(train["label"].unique())
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    id2label = {i: l for l, i in label2id.items()}

    print("Building vocabulary...")
    vocab = build_vocab(train["text"].tolist(), min_freq=args.min_freq,
                         max_size=args.vocab_size)
    print(f"Vocab size: {len(vocab)}")

    train_ds = NewsDataset(train["text"].tolist(), train["label"].tolist(),
                            vocab, label2id, args.max_len)
    val_ds = NewsDataset(val["text"].tolist(), val["label"].tolist(),
                          vocab, label2id, args.max_len)
    eval_ds = NewsDataset(eval_df["text"].tolist(), eval_df["label"].tolist(),
                           vocab, label2id, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    model = BiLSTMClassifier(
        vocab_size=len(vocab), embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim, n_classes=len(label2id),
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.class_weights:
        # Inverse-frequency class weights — important given real-world label
        # imbalance (your Amharic news data has ~33x imbalance between the
        # largest and smallest class). Without this, the model can get a
        # deceptively high accuracy by mostly ignoring rare classes.
        counts = train["label"].value_counts()
        freq = np.array([counts[id2label[i]] for i in range(len(label2id))])
        weights = (freq.sum() / (len(freq) * freq))
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Using class weights (inverse frequency): "
              f"{dict(zip(labels_sorted, weights.cpu().numpy().round(2)))}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_ds)

        val_preds, val_labels = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_labels, val_preds)
        _, _, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="macro", zero_division=0
        )
        print(f"Epoch {epoch}/{args.epochs} — train_loss={train_loss:.4f} "
              f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "bilstm_best.pt"))

    eval_preds, eval_labels = evaluate(model, eval_loader, device)
    acc = accuracy_score(eval_labels, eval_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        eval_labels, eval_preds, average="macro", zero_division=0
    )
    print(f"\nFinal eval — acc={acc:.4f} macro_P={precision:.4f} "
          f"macro_R={recall:.4f} macro_F1={f1:.4f}")

    run_name = "bilstm_weighted" if args.class_weights else "bilstm"
    metrics = {
        "run_name": run_name,
        "approach": f"BiLSTM (embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}, "
                    f"layers={args.n_layers}), trained embeddings, vocab={len(vocab)}"
                    + (", class-weighted loss" if args.class_weights else ""),
        "eval_split": "test" if test is not None else "validation",
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "best_val_macro_f1": best_val_f1,
        "config": vars(args),
    }
    with open(os.path.join(RESULTS_DIR, f"{run_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    preds_df = eval_df.copy()
    preds_df["predicted_label"] = [id2label[p] for p in eval_preds]
    preds_df.to_csv(os.path.join(RESULTS_DIR, f"{run_name}_preds.csv"), index=False)

    print(f"\nSaved metrics -> experiments/results/{run_name}_metrics.json")
    print(f"Saved model -> models/bilstm_best.pt")
    print(
        f"\n>>> Paste into experiments/experiments_log.md:\n"
        f"| {run_name} | {metrics['approach']} | acc={acc:.4f}, macro-F1={f1:.4f} | "
        f"<your note on what you learned> |"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--class-weights", action="store_true",
                         help="Use inverse-frequency class weighting in the loss "
                              "(recommended given label imbalance — try both "
                              "with and without as an experiment)")
    args = parser.parse_args()
    main(args)