"""
Error analysis for a trained model's predictions.

"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "plots")
FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts",
                          "NotoSansEthiopic-Regular.ttf")


def _setup_amharic_font():
    """See src/eda.py for the same helper — kept duplicated rather than
    shared to keep each script runnable standalone."""
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams["font.family"] = prop.get_name()
    else:
        print(f"[WARNING] Amharic font not found at {FONT_PATH} — "
              f"confusion matrix labels may render as boxes.")


def main(args):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    _setup_amharic_font()
    df = pd.read_csv(args.predictions)
    run_name = os.path.basename(args.predictions).replace("_preds.csv", "")

    if "label" not in df.columns or "predicted_label" not in df.columns:
        raise ValueError(
            f"Expected 'label' and 'predicted_label' columns in {args.predictions}, "
            f"found: {list(df.columns)}"
        )

    y_true = df["label"]
    y_pred = df["predicted_label"]
    labels_sorted = sorted(set(y_true) | set(y_pred))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels_sorted, yticklabels=labels_sorted,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {run_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, f"{run_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # --- Per-class report ---
    report = classification_report(y_true, y_pred, labels=labels_sorted, zero_division=0)
    print("\nPer-class report:\n")
    print(report)

    report_path = os.path.join(RESULTS_DIR, f"{run_name}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # --- Most-confused label pairs (off-diagonal cells, sorted descending) ---
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    confused_pairs = []
    for true_label in labels_sorted:
        for pred_label in labels_sorted:
            if true_label != pred_label and cm_df.loc[true_label, pred_label] > 0:
                confused_pairs.append(
                    (true_label, pred_label, cm_df.loc[true_label, pred_label])
                )
    confused_pairs.sort(key=lambda x: -x[2])

    print("\nTop confused label pairs (true -> predicted : count):")
    for true_l, pred_l, count in confused_pairs[:10]:
        print(f"  {true_l} -> {pred_l} : {count}")

    # --- Misclassified examples ---
    mistakes = df[y_true != y_pred].copy()
    mistakes["text_len"] = mistakes["text"].str.len()
    mistakes_path = os.path.join(RESULTS_DIR, f"{run_name}_misclassified.csv")
    mistakes.sort_values("text_len").to_csv(mistakes_path, index=False)
    print(f"\n{len(mistakes)} misclassified examples ({100*len(mistakes)/len(df):.1f}% "
          f"of eval set) saved to {mistakes_path}")

    # --- Markdown snippet with a handful of concrete examples ---
    md_lines = [f"### Error Analysis Examples — {run_name}\n"]
    md_lines.append(f"- Overall accuracy on this split: computed from metrics.json")
    md_lines.append(f"- {len(mistakes)}/{len(df)} misclassified ({100*len(mistakes)/len(df):.1f}%)\n")
    md_lines.append("**Top confused pairs:**\n")
    for true_l, pred_l, count in confused_pairs[:5]:
        md_lines.append(f"- `{true_l}` mistaken for `{pred_l}`: {count} times")

    md_lines.append("\n**Sample misclassified examples (fill in your own analysis "
                     "of *why* for each):**\n")
    sample_n = min(args.n_examples, len(mistakes))
    for _, row in mistakes.sample(sample_n, random_state=42).iterrows():
        snippet = row["text"][:200].replace("\n", " ")
        md_lines.append(
            f"> **True:** {row['label']} | **Predicted:** {row['predicted_label']}\n"
            f"> \"{snippet}...\"\n> \n> _Why might the model have made this mistake?_ "
            f"<your analysis here>\n"
        )

    md_lines.append("\n**Sample CORRECT examples (useful for the 'what works well' "
                     "half of your error analysis):**\n")
    correct = df[y_true == y_pred]
    for _, row in correct.sample(min(3, len(correct)), random_state=42).iterrows():
        snippet = row["text"][:200].replace("\n", " ")
        md_lines.append(
            f"> **Label:** {row['label']} (correctly predicted)\n"
            f"> \"{snippet}...\"\n"
        )

    md_path = os.path.join(RESULTS_DIR, f"{run_name}_error_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nMarkdown snippet for your report saved to {md_path}")
    print("(Paste/adapt this into Section 7 of report/report_outline.md)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True,
                         help="Path to a <run_name>_preds.csv file")
    parser.add_argument("--n-examples", type=int, default=8,
                         help="Number of misclassified examples to sample into the markdown report")
    args = parser.parse_args()
    main(args)