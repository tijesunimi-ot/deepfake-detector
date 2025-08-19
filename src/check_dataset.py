# scripts/check_dataset.py
from dataset import DeepfakeDataset

meta = "data/processed/train_meta.csv"
ds = DeepfakeDataset(meta)
print("meta:", meta)
print("num samples:", len(ds))
batch_size = 8
print(f"batch_size {batch_size} -> steps per epoch: {len(ds) // batch_size}")


# command to run this script: python .\src\check_dataset.py