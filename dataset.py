"""
dataset.py  -  turns rows of the split csv into (image tensor, label, row index).
Used for training (augmentation on) and for prediction (augmentation off).
"""
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

import config as C

def subtype_of(filename: str) -> str:
    """BreaKHis tumour subtype from the filename: SOB_B_TA-14-4659-40-001.png -> 'TA'
    (benign: A, F, PT, TA | malignant: DC, LC, MC, PC)."""
    m = re.match(r"SOB_[BM]_([A-Z]+)-", str(filename), re.IGNORECASE)
    return m.group(1).upper() if m else "?"


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet stats (pretrained backbone)


def build_transform(img_size: int, train: bool):
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # mild stain/colour variation: histology slides differ between labs/patients
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class ImageRows(Dataset):
    """Reads the pre-resized copy from cache/<img_size>/ if cache_images.py was run
    (much faster); otherwise falls back to the original full-size image."""

    def __init__(self, df, img_size: int, train: bool):
        df = df.reset_index(drop=True)
        self.paths = df["image_path"].tolist()
        self.names = df["filename"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.cache_dir = C.CACHE_DIR / str(img_size)
        self.tf = build_transform(img_size, train)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        cached = self.cache_dir / self.names[i]
        img = Image.open(cached if cached.exists() else self.paths[i]).convert("RGB")
        return self.tf(img), torch.tensor(self.labels[i]), i
