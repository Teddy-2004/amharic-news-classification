import os
from datasets import load_dataset

DATASET_NAME = "rasyosef/amharic-news-category-classification"
OUT_DIR = os.path.join(os.path.dirname(__file__), "raw")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading '{DATASET_NAME}' from the Hugging Face Hub...")
    ds = load_dataset(DATASET_NAME)
    print(ds)

    # Save each split to CSV for easy inspection / reuse across scripts
    for split in ds.keys():
        out_path = os.path.join(OUT_DIR, f"{split}.csv")
        ds[split].to_csv(out_path, index=False)
        print(f"  wrote {len(ds[split])} rows -> {out_path}")

    # If the dataset only ships a single split, carve out a val split so we
    # have train/val/test rather than just train/test.
    if "validation" not in ds and "test" in ds and "train" in ds:
        print(
            "\nNo validation split found — creating one by carving 10% "
            "off the training set (stratified by label)."
        )
        split = ds["train"].train_test_split(
            test_size=0.1, seed=42, stratify_by_column="category"
        )
        split["train"].to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
        split["test"].to_csv(os.path.join(OUT_DIR, "validation.csv"), index=False)
        print(
            f"  train: {len(split['train'])} rows, "
            f"validation: {len(split['test'])} rows"
        )

    print("\nDone. Run `python src/eda.py` next.")


if __name__ == "__main__":
    main()
