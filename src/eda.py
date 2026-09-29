"""
Exploratory Data Analysis for the Amharic news classification dataset.

"""

import os
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from data_utils import load_clean_splits

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts",
                          "NotoSansEthiopic-Regular.ttf")


def _setup_amharic_font():
    """Registers a font with Ethiopic script support so plot labels render
    correctly instead of as missing-glyph boxes. Falls back to default font
    (with a warning) if the bundled font file is missing for any reason."""
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams["font.family"] = prop.get_name()
        return prop
    else:
        print(f"[WARNING] Amharic font not found at {FONT_PATH} — "
              f"plot labels with Amharic text may render as boxes. "
              f"Re-download assets/fonts/NotoSansEthiopic-Regular.ttf.")
        return None


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    _setup_amharic_font()
    train, val, test = load_clean_splits()

    lines = ["# EDA Report — Amharic News Classification\n"]

    for name, df in [("train", train), ("validation", val), ("test", test)]:
        if df is None:
            continue
        lines.append(f"## {name.capitalize()} split\n")
        lines.append(f"- Rows: **{len(df)}**")
        lines.append(f"- Unique labels: **{df['label'].nunique()}**")
        lines.append(f"- Empty/duplicate text rows dropped during cleaning "
                      f"(see src/data_utils.py::basic_clean)\n")

        counts = df["label"].value_counts()
        lines.append("| Label | Count | % of split |")
        lines.append("|---|---|---|")
        for label, count in counts.items():
            pct = 100 * count / len(df)
            lines.append(f"| {label} | {count} | {pct:.1f}% |")
        lines.append("")

        # Imbalance flag
        imbalance_ratio = counts.max() / counts.min()
        lines.append(
            f"**Class imbalance ratio (max/min): {imbalance_ratio:.2f}x** "
            f"{'⚠️ consider class weighting or macro-F1 as primary metric' if imbalance_ratio > 3 else '(reasonably balanced)'}\n"
        )

        # Text length stats (in whitespace-separated tokens, a rough proxy)
        df = df.copy()
        df["n_tokens"] = df["text"].str.split().apply(len)
        lines.append("**Text length (whitespace tokens):**\n")
        lines.append(f"- mean: {df['n_tokens'].mean():.1f}")
        lines.append(f"- median: {df['n_tokens'].median():.1f}")
        lines.append(f"- min: {df['n_tokens'].min()}")
        lines.append(f"- max: {df['n_tokens'].max()}")
        lines.append(f"- 95th percentile: {df['n_tokens'].quantile(0.95):.0f}")
        lines.append(
            "\n_Use the 95th percentile as a sanity check for your "
            "tokenizer's max_length setting when fine-tuning Transformers._\n"
        )

        # Plots (train split only, to keep it simple)
        if name == "train":
            plt.figure(figsize=(8, 5))
            sns.countplot(y=df["label"], order=counts.index)
            plt.title("Class Balance — Train Split")
            plt.xlabel("Count")
            plt.ylabel("Category")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "class_balance.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(8, 5))
            sns.histplot(df["n_tokens"], bins=50)
            plt.title("Article Length Distribution (tokens) — Train Split")
            plt.xlabel("Number of whitespace-separated tokens")
            plt.axvline(df["n_tokens"].quantile(0.95), color="red",
                        linestyle="--", label="95th percentile")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "text_length_hist.png"), dpi=150)
            plt.close()

            lines.append(f"Plots saved to `experiments/plots/`.\n")

    lines.append("\n## Things to check manually before modeling\n")
    lines.append("- [ ] Skim 20-30 random articles per class — do labels look correct?")
    lines.append("- [ ] Check for near-duplicate articles (common in scraped news corpora)")
    lines.append("- [ ] Check for any English-language leakage in the Amharic text")
    lines.append("- [ ] Confirm train/test label sets match exactly (no unseen labels in test)")

    report_path = os.path.join(OUT_DIR, "eda_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"EDA report written to {report_path}")
    print(f"Plots written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()