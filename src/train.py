from torch.utils.data import DataLoader
from src.dataset import DeepfakeDataset

train_ds = DeepfakeDataset("data/processed/train_meta.csv")
val_ds = DeepfakeDataset("data/processed/val_meta.csv")
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
