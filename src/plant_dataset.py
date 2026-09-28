#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""plant_dataset.py

Dataset loader module for plant disease classification.
Supports:
1. PlantVillage (12 selected classes across Apple, Grape, Peach, and Strawberry)
2. Cassava (5 classes with numeric labels 0-4 from merged.csv)
"""

import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# 1. PLANTVILLAGE_CLASSES: 12 specified classes
PLANTVILLAGE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Strawberry___healthy',
    'Strawberry___Leaf_scorch'
]

# 2. PLANTVILLAGE_CLASS_TO_IDX: Deterministic mapping from class name to integer (0-11)
PLANTVILLAGE_CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(PLANTVILLAGE_CLASSES)}


# 3. load_plantvillage_samples()
def load_plantvillage_samples(data_dir=r"E:\problem-statement-8\datasets\PlantVillage-Dataset\raw\color"):
    """Collects (image_path, class_idx) for the 12 required PlantVillage classes.

    Args:
        data_dir (str): Root folder containing class subdirectories.

    Returns:
        list of tuple: [(image_path, label_int), ...]
        dict: Mapping of class name to label index.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"PlantVillage directory not found: {data_dir}")

    samples = []
    valid_exts = {'.jpg', '.jpeg', '.png'}

    for cls_name in PLANTVILLAGE_CLASSES:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Required PlantVillage class folder missing: {cls_dir}")

        label = PLANTVILLAGE_CLASS_TO_IDX[cls_name]
        for fname in os.listdir(cls_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                full_path = os.path.join(cls_dir, fname)
                samples.append((full_path, label))

    return samples, PLANTVILLAGE_CLASS_TO_IDX


# 4. load_cassava_samples()
def load_cassava_samples(
    images_dir=r"E:\problem-statement-8\datasets\train",
    csv_path=r"E:\problem-statement-8\datasets\merged.csv"
):
    """Collects (image_path, numeric_label) for Cassava using merged.csv.

    Uses numeric labels 0-4 exactly as stored in merged.csv without any invented mappings.

    Args:
        images_dir (str): Folder containing Cassava train images.
        csv_path (str): Path to merged.csv containing image_id and label columns.

    Returns:
        list of tuple: [(image_path, label_int), ...]
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Cassava images directory not found: {images_dir}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Cassava merged.csv file not found: {csv_path}")

    samples = []
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['image_id']
            label = int(row['label'])  # Numeric label 0-4 directly from CSV
            full_path = os.path.join(images_dir, image_id)
            if os.path.isfile(full_path):
                samples.append((full_path, label))

    return samples


# 5. PlantDataset
class PlantDataset(Dataset):
    """PyTorch Dataset wrapping a list of (image_path, integer_label) pairs."""

    def __init__(self, samples, transform=None):
        """
        Args:
            samples (list of tuple): List of (image_path, label) pairs.
            transform (callable, optional): Transform to be applied on a PIL Image.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# 6. get_transforms()
def get_transforms(condition="raw", img_size=224):
    """Returns image transformation pipeline for a given preprocessing condition.

    Currently implements only the 'raw' condition.

    Args:
        condition (str): Preprocessing condition (currently only 'raw').
        img_size (int): Target square image resolution (default: 224).

    Returns:
        torchvision.transforms.Compose: Transform pipeline returning [3, img_size, img_size] tensor.
    """
    if condition == "raw":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise NotImplementedError(f"Condition '{condition}' is not implemented yet. Only 'raw' is supported.")


# 7. split_samples()
def split_samples(samples, val_ratio=0.2, seed=42, stratify=True):
    """Splits sample tuples into reproducible train and validation sets using file paths and labels.

    Does not copy, move, or modify any files on disk.

    Args:
        samples (list of tuple): List of (image_path, label) pairs.
        val_ratio (float): Fraction of data to use for validation (default 0.2).
        seed (int): Random seed for reproducibility.
        stratify (bool): Whether to stratify split by class label.

    Returns:
        tuple: (train_samples, val_samples)
    """
    if not samples:
        raise ValueError("Samples list is empty.")

    labels = [s[1] for s in samples] if stratify else None
    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels
    )
    return train_samples, val_samples


# 8. get_plantvillage_loaders()
def get_plantvillage_loaders(
    data_dir=r"E:\problem-statement-8\datasets\PlantVillage-Dataset\raw\color",
    condition="raw",
    img_size=224,
    val_ratio=0.2,
    seed=42,
    batch_size=32,
    num_workers=0
):
    """Creates reproducible train and validation DataLoaders for the 12 PlantVillage classes.

    Returns:
        tuple: (train_loader, val_loader, class_to_idx)
    """
    samples, class_to_idx = load_plantvillage_samples(data_dir=data_dir)
    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio, seed=seed)

    transform = get_transforms(condition=condition, img_size=img_size)

    train_dataset = PlantDataset(train_samples, transform=transform)
    val_dataset = PlantDataset(val_samples, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, class_to_idx


# 9. get_cassava_loaders()
def get_cassava_loaders(
    images_dir=r"E:\problem-statement-8\datasets\train",
    csv_path=r"E:\problem-statement-8\datasets\merged.csv",
    condition="raw",
    img_size=224,
    val_ratio=0.2,
    seed=42,
    batch_size=32,
    num_workers=0
):
    """Creates reproducible train and validation DataLoaders for Cassava (numeric labels 0-4).

    Returns:
        tuple: (train_loader, val_loader)
    """
    samples = load_cassava_samples(images_dir=images_dir, csv_path=csv_path)
    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio, seed=seed)

    transform = get_transforms(condition=condition, img_size=img_size)

    train_dataset = PlantDataset(train_samples, transform=transform)
    val_dataset = PlantDataset(val_samples, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
    