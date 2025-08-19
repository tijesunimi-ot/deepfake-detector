# src/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class DeepfakeDataset(Dataset):
    def __init__(self, meta_csv, transforms=None):
        self.df = pd.read_csv(meta_csv, dtype={'video': str}, low_memory=False)
        self.transforms = transforms
        if self.transforms is None:
            self.transforms = A.Compose([
                A.Resize(224, 224),  # <-- change from 160 to 224
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['crop_path']
        label = int(row['label'])
        img = np.array(Image.open(img_path).convert('RGB'))
        augmented = self.transforms(image=img)
        img_t = augmented['image']
        return img_t, torch.tensor(label, dtype=torch.long)
