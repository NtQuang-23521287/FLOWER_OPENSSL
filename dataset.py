"""
dataset.py — Load và chia dataset COVID-19 Radiography cho từng client

Chiến lược non-IID:
  Client 1 → COVID     + Normal  (mô phỏng bệnh viện chuyên COVID)
  Client 2 → Viral Pn. + Normal  (mô phỏng bệnh viện đa khoa)
"""
import os
import glob
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from config import (
    DATASET_DIR, CLIENT_DATA, MAX_IMAGES_PER_CLASS,
    IMAGE_SIZE, BATCH_SIZE, CLASSES
)


# ── Transform ───────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # ImageNet mean/std — dùng được cho ảnh grayscale chuyển RGB
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset class ────────────────────────────────────────────────────────────
class CovidDataset(Dataset):
    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── Loader factory ───────────────────────────────────────────────────────────
def load_client_data(client_id: int) -> Tuple[DataLoader, DataLoader]:
    """
    Trả về (train_loader, val_loader) cho client_id (1 hoặc 2).
    Tỷ lệ train/val = 80/20.
    """
    assigned_classes = CLIENT_DATA[client_id]
    class_to_idx     = {c: CLASSES.index(c) for c in CLASSES}

    all_paths, all_labels = [], []

    for class_name in assigned_classes:
        # Thư mục ảnh: COVID-19_Radiography_Dataset/<class>/images/
        img_dir = os.path.join(DATASET_DIR, class_name, "images")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục: {img_dir}\n"
                f"Hãy kiểm tra lại DATASET_DIR trong config.py"
            )

        paths = (
            glob.glob(os.path.join(img_dir, "*.png")) +
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.jpeg"))
        )

        if MAX_IMAGES_PER_CLASS:
            paths = paths[:MAX_IMAGES_PER_CLASS]

        label = class_to_idx[class_name]
        all_paths  += paths
        all_labels += [label] * len(paths)

        print(f"  [Client {client_id}] {class_name}: {len(paths)} ảnh (label={label})")

    dataset = CovidDataset(all_paths, all_labels)

    # Chia train/val
    n_val   = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Gắn transform riêng
    train_ds.dataset = CovidDataset(
        [all_paths[i]  for i in train_ds.indices],
        [all_labels[i] for i in train_ds.indices],
        transform=TRAIN_TRANSFORM,
    )
    val_ds.dataset = CovidDataset(
        [all_paths[i]  for i in val_ds.indices],
        [all_labels[i] for i in val_ds.indices],
        transform=VAL_TRANSFORM,
    )

    # Tạo DataLoader đơn giản (không dùng indices wrapper)
    train_loader = DataLoader(train_ds.dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds.dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"  [Client {client_id}] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader