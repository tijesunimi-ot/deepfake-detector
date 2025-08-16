# src/split_dataset.py  (fixed, robust)
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import argparse
import numpy as np

def split_by_video(meta_csv, out_root, val_size=0.1, test_size=0.1, random_state=42):
    # Read CSV robustly (avoid mixed-type inference)
    df = pd.read_csv(meta_csv, low_memory=False, dtype=str)

    # Ensure relevant columns exist
    if 'video' not in df.columns:
        raise ValueError("metadata CSV must contain a 'video' column")

    # Normalize video column to uniform string type (strip whitespace)
    df['video'] = df['video'].astype(str).str.strip()

    # Drop rows with missing/empty video IDs
    before = len(df)
    df = df[df['video'].notnull() & (df['video'] != '')].copy()
    after = len(df)
    if after < before:
        print(f"Dropped {before-after} rows with missing/empty 'video' values")

    # Convert label to int if present (optional)
    if 'label' in df.columns:
        # some label rows may be non-numeric strings; try safe cast
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    groups = df['video'].values  # all strings now

    total_holdout = val_size + test_size
    if total_holdout <= 0:
        # all train
        train = df
        val = df.iloc[0:0]
        test = df.iloc[0:0]
    else:
        # First split: train vs temp (val+test)
        splitter = GroupShuffleSplit(n_splits=1, test_size=total_holdout, random_state=random_state)
        train_idx, temp_idx = next(splitter.split(df, y=None, groups=groups))
        temp = df.iloc[temp_idx].copy()

        # Second split: temp -> val and test (split by groups in temp)
        rel_test_size = test_size / total_holdout if total_holdout > 0 else 0.0
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=rel_test_size, random_state=random_state)
        temp_groups = temp['video'].values
        if len(temp) == 0:
            val = temp.iloc[0:0]
            test = temp.iloc[0:0]
        else:
            val_idx_rel, test_idx_rel = next(splitter2.split(temp, y=None, groups=temp_groups))
            val = temp.iloc[val_idx_rel]
            test = temp.iloc[test_idx_rel]

        train = df.iloc[train_idx]

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_root / "train_meta.csv", index=False)
    val.to_csv(out_root / "val_meta.csv", index=False)
    test.to_csv(out_root / "test_meta.csv", index=False)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_csv", default="data/processed/metadata.csv")
    parser.add_argument("--out_root", default="data/processed")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()
    split_by_video(args.meta_csv, args.out_root, args.val_size, args.test_size)
    print("Dataset split completed successfully.")