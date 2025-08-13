# src/split_dataset.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import argparse

def split_by_video(meta_csv, out_root, val_size=0.1, test_size=0.1, random_state=42):
    df = pd.read_csv(meta_csv)
    # group by video to avoid leakage
    videos = df['video'].unique()
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size+test_size, random_state=random_state)
    groups = df['video']
    train_idx, temp_idx = next(splitter.split(df, groups=groups, groups=groups))
    temp = df.iloc[temp_idx].copy()
    # split temp further into val/test by video
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=test_size/(val_size+test_size), random_state=random_state)
    temp_groups = temp['video']
    val_idx_rel, test_idx_rel = next(splitter2.split(temp, groups=temp_groups, groups=temp_groups))
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
