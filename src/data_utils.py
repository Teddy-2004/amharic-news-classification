"""
Shared utilities for loading and lightly cleaning the Amharic news dataset.

"""

import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Amharic uses the Ge'ez script; punctuation includes both Latin-style marks
# and native marks like the section mark (::) and word-space (::). We keep
# these configurable rather than silently stripping them.
GEEZ_PUNCT_PATTERN = re.compile(r"[፡።፣፤፥፦፧፨]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"http\S+|www\.\S+")


def load_split(split: str) -> pd.DataFrame:
    """Load one split ('train', 'validation', or 'test') as a DataFrame.

    Expects columns roughly like: ['headline', 'category', 'article']
    (rename map applied below) — check the printed columns on first run and
    adjust COLUMN_MAP if the mirror you're using differs.
    """
    path = os.path.join(DATA_DIR, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python data/download_data.py` first."
        )
    df = pd.read_csv(path)
    df = _standardize_columns(df)
    return df


COLUMN_MAP_CANDIDATES = {
    # left: possible source column names -> right: standardized name
    "headline": "headline",
    "title": "headline",
    "article": "text",
    "text": "text",
    "content": "text",
    "category": "label",
    "label": "label",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_MAP_CANDIDATES:
            rename[col] = COLUMN_MAP_CANDIDATES[key]
    df = df.rename(columns=rename)

    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns 'text' and 'label' after standardization, "
            f"missing {missing}. Actual columns: {list(df.columns)}. "
            f"Update COLUMN_MAP_CANDIDATES in src/data_utils.py to match "
            f"the real column names printed by data/download_data.py."
        )
    return df


def basic_clean(text: str, remove_urls: bool = True) -> str:
    """Minimal, explainable cleaning — document every step in your report.

    Deliberately does NOT strip Amharic punctuation by default, since sentence
    boundaries (marked with ።) can matter for some models. Toggle as needed
    and justify the choice in the Dataset & Preprocessing section.
    """
    if not isinstance(text, str):
        return ""
    if remove_urls:
        text = URL_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text


def load_clean_splits():
    """Convenience loader: returns (train_df, val_df, test_df), all cleaned."""
    splits = {}
    for name in ["train", "validation", "test"]:
        try:
            df = load_split(name)
        except FileNotFoundError:
            continue
        df["text"] = df["text"].apply(basic_clean)
        df = df[df["text"].str.len() > 0].reset_index(drop=True)
        splits[name] = df
    return splits.get("train"), splits.get("validation"), splits.get("test")


if __name__ == "__main__":
    train, val, test = load_clean_splits()
    print("train:", None if train is None else train.shape)
    print("validation:", None if val is None else val.shape)
    print("test:", None if test is None else test.shape)
    if train is not None:
        print("\nLabel distribution (train):")
        print(train["label"].value_counts())
