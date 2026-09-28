import argparse
import os
import sys

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

# Same mapping as src/data_utils.py — kept in sync manually since this script
# runs standalone before the rest of the pipeline.
COLUMN_MAP_CANDIDATES = {
    "headline": "headline",
    "title": "headline",
    "article": "text",
    "text": "text",
    "content": "text",
    "category": "label",
    "label": "label",
}


def standardize_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_MAP_CANDIDATES:
            rename[col] = COLUMN_MAP_CANDIDATES[key]
    df = df.rename(columns=rename)

    missing = {"text", "label"} - set(df.columns)
    if missing:
        print(f"\n[ERROR] {source_name}: missing required columns {missing} "
              f"after standardization.")
        print(f"  Columns found: {list(df.columns)}")
        print(f"  Update COLUMN_MAP_CANDIDATES in this script AND in "
              f"src/data_utils.py to match.")
        sys.exit(1)
    return df


def main(args):
    train_path = os.path.join(RAW_DIR, "train.csv")
    test_path = os.path.join(RAW_DIR, "test.csv")

    for p, name in [(train_path, "train.csv"), (test_path, "test.csv")]:
        if not os.path.exists(p):
            print(f"[ERROR] {p} not found.")
            print(f"  Place your manually downloaded {name} at that exact "
                  f"path before running this script.")
            sys.exit(1)

    print(f"Loading {train_path} ...")
    train = pd.read_csv(train_path)
    print(f"  raw shape: {train.shape}, columns: {list(train.columns)}")

    print(f"\nLoading {test_path} ...")
    test = pd.read_csv(test_path)
    print(f"  raw shape: {test.shape}, columns: {list(test.columns)}")

    train = standardize_columns(train, "train.csv")
    test = standardize_columns(test, "test.csv")

    # Drop fully-empty text rows up front (rare, but cheap to check)
    before = len(train)
    train = train[train["text"].notna() & (train["text"].str.strip() != "")]
    if len(train) < before:
        print(f"\nDropped {before - len(train)} rows with empty article text from train.")

    print("\n--- Label distribution (train, before split) ---")
    print(train["label"].value_counts())
    print(f"\nTotal distinct labels: {train['label'].nunique()}")

    # Check train/test label sets match
    train_labels = set(train["label"].unique())
    test_labels = set(test["label"].unique())
    if train_labels != test_labels:
        print(f"\n[WARNING] Label mismatch between train and test!")
        print(f"  In train but not test: {train_labels - test_labels}")
        print(f"  In test but not train: {test_labels - train_labels}")
        print(f"  This can break stratified splitting and evaluation — "
              f"investigate before proceeding (likely a rare class with "
              f"very few examples, or a label typo).")

    # Stratified split for validation set
    print(f"\nCarving {args.val_size:.0%} validation split from train "
          f"(stratified by label)...")
    try:
        from sklearn.model_selection import train_test_split
        new_train, val = train_test_split(
            train, test_size=args.val_size, random_state=42,
            stratify=train["label"],
        )
    except ValueError as e:
        print(f"[WARNING] Stratified split failed ({e}) — likely a class "
              f"with too few examples for the requested split size. "
              f"Falling back to a non-stratified split.")
        new_train, val = train_test_split(
            train, test_size=args.val_size, random_state=42,
        )

    print(f"  new train: {len(new_train)} rows")
    print(f"  validation: {len(val)} rows")
    print(f"  test (unchanged): {len(test)} rows")

    # Back up the original full train.csv before overwriting, just in case
    backup_path = os.path.join(RAW_DIR, "train_full_backup.csv")
    if not os.path.exists(backup_path):
        train.to_csv(backup_path, index=False)
        print(f"\nBacked up original full train set -> {backup_path}")

    new_train.to_csv(train_path, index=False)
    val.to_csv(os.path.join(RAW_DIR, "validation.csv"), index=False)
    test.to_csv(test_path, index=False)  # re-save standardized version

    print(f"\nWrote standardized + split files:")
    print(f"  {train_path} ({len(new_train)} rows)")
    print(f"  {os.path.join(RAW_DIR, 'validation.csv')} ({len(val)} rows)")
    print(f"  {test_path} ({len(test)} rows)")
    print(f"\nDone. Run `python src/eda.py` next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-size", type=float, default=0.10,
                         help="Fraction of train.csv to carve out as validation")
    args = parser.parse_args()
    main(args)