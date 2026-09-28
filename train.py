#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train.py (PlantVillage Branch)

B-CNN Training Script for PlantVillage Dataset (12 classes).
Supports:
- Safe pause and resume training (RESUME = True / False)
- Checkpointing after every epoch (latest and best)
- Metrics history persistence
- Graceful Ctrl+C handling
- Immediate unbuffered terminal progress flushing
"""

import sys
import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg16, VGG16_Weights
from sklearn.metrics import f1_score

# Ensure src is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from plant_dataset import get_plant_loaders, PLANTVILLAGE_12_CLASSES


# ==============================================================================
# RESUME & EXPERIMENT CONFIGURATION (PLANTVILLAGE)
# ==============================================================================
RESUME = True             # Set to True to resume from latest checkpoint; False to start from Epoch 1
TOTAL_EPOCHS = 500        # Total epochs to train
BATCH_SIZE = 32           # Batch size
IMAGE_SIZE = 224          # Input image resolution (224x224)
NUM_CLASSES = 12          # 12 PlantVillage classes
LEARNING_RATE = 0.01      # SGD base learning rate
MOMENTUM = 0.9            # SGD momentum
WEIGHT_DECAY = 5e-4       # SGD weight decay
SEED = 42                 # Random seed for reproducibility
DATASET_CONDITION = "raw" # Preprocessing condition
VAL_RATIO = 0.2           # 80/20 train/val split
NUM_WORKERS = 2           # DataLoader workers (safe on Windows)

# Checkpoint paths
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "bcnn_plantvillage_raw_latest.pth")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "bcnn_plantvillage_raw_best.pth")
METRICS_HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "bcnn_plantvillage_raw_metrics.json")
# ==============================================================================


# ------------------------------------------------------------------------------
# B-CNN ARCHITECTURE (Confirmed HaoMood single-branch VGG-16 + self-bilinear pooling)
# ------------------------------------------------------------------------------
class BCNN(nn.Module):
    """B-CNN for plant disease classification.

    Architecture:
    VGG-16 features without pool5 -> 512*14*14
    -> self-bilinear pooling (X @ X.T) / (14^2) -> 512*512
    -> square-root normalization -> L2 normalization -> Linear(512^2, num_classes).
    """
    def __init__(self, num_classes=12, pretrained=True):
        super(BCNN, self).__init__()
        self.num_classes = num_classes
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg16(weights=weights)
        self.features = nn.Sequential(*list(vgg.features.children())[:-1])  # Remove pool5
        self.fc = nn.Linear(512**2, num_classes)  # Default PyTorch initialization

    def forward(self, X):
        N = X.size(0)
        assert X.size() == (N, 3, 224, 224), f"Expected shape ({N}, 3, 224, 224), got {X.size()}"
        X = self.features(X)
        assert X.size() == (N, 512, 14, 14), f"Expected shape ({N}, 512, 14, 14), got {X.size()}"
        X = X.view(N, 512, 14**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (14**2)  # Self-bilinear pooling
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)  # Square-root normalization
        X = nn.functional.normalize(X)  # L2 normalization
        X = self.fc(X)
        assert X.size() == (N, self.num_classes)
        return X


def set_seed(seed=42):
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += images.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set and return loss, accuracy, and Macro-F1."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += images.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    macro_f1 = float(f1_score(all_targets, all_preds, average='macro')) if all_targets else 0.0
    return avg_loss, accuracy, macro_f1


def save_checkpoint(path, epoch, model, optimizer, best_macro_f1, train_loss, train_acc, val_loss, val_acc, val_macro_f1, history, total_elapsed_time):
    """Save full training state to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_macro_f1': best_macro_f1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_macro_f1': val_macro_f1,
        'history': history,
        'total_elapsed_time': total_elapsed_time,
        'num_classes': NUM_CLASSES,
        'classes': PLANTVILLAGE_12_CLASSES,
        'seed': SEED
    }
    torch.save(checkpoint, path)


def run_training(resume=RESUME, max_epochs=TOTAL_EPOCHS):
    """Main training loop supporting pause/resume for PlantVillage."""
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    print("=" * 65, flush=True)
    print("B-CNN TRAINING PIPELINE (PLANTVILLAGE DATASET)", flush=True)
    print("=" * 65, flush=True)
    print(f"Device: {device} ({gpu_name})", flush=True)
    print(f"Target Total Epochs: {max_epochs}", flush=True)
    print(f"Batch Size: {BATCH_SIZE}", flush=True)
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}", flush=True)
    print(f"Number of Classes: {NUM_CLASSES}", flush=True)
    print(f"Dataset Condition: {DATASET_CONDITION}", flush=True)

    # 1. Prepare PlantVillage data loaders
    print("Loading PlantVillage dataset...", flush=True)
    train_loader, val_loader = get_plant_loaders(
        condition=DATASET_CONDITION,
        img_size=IMAGE_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"Loaded: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.", flush=True)

    # 2. Instantiate model and optimizer
    model = BCNN(num_classes=NUM_CLASSES, pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # 3. Resume handling
    start_epoch = 1
    best_val_macro_f1 = 0.0
    history = []
    total_elapsed_time = 0.0
    last_completed_epoch = 0

    if resume:
        if os.path.isfile(LATEST_CHECKPOINT_PATH):
            checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_completed_epoch = checkpoint['epoch']
            start_epoch = last_completed_epoch + 1
            best_val_macro_f1 = checkpoint.get('best_val_macro_f1', 0.0)
            history = checkpoint.get('history', [])
            total_elapsed_time = checkpoint.get('total_elapsed_time', 0.0)

            print("\nResuming from PlantVillage checkpoint...", flush=True)
            print(f"Last completed epoch: {last_completed_epoch}", flush=True)
            print(f"Starting Epoch {start_epoch}/{max_epochs}", flush=True)
        else:
            print(f"\n[INFO] RESUME=True but no PlantVillage checkpoint found at '{LATEST_CHECKPOINT_PATH}'.", flush=True)
            print("Starting fresh PlantVillage training from Epoch 1.", flush=True)
    else:
        print("\nStarting fresh PlantVillage training from Epoch 1 (RESUME=False).", flush=True)

    if start_epoch > max_epochs:
        print(f"\nTraining already completed! (Completed: {last_completed_epoch}/{max_epochs})", flush=True)
        return

    # 4. Training loop
    try:
        for epoch in range(start_epoch, max_epochs + 1):
            print(flush=True)
            print("=" * 50, flush=True)
            print(f"Epoch {epoch}/{max_epochs}", flush=True)
            print("=" * 50, flush=True)

            epoch_start_time = time.time()

            # Train and validate
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device)

            epoch_time = time.time() - epoch_start_time
            total_elapsed_time += epoch_time
            last_completed_epoch = epoch

            # Check if this is the best Macro-F1 so far
            is_best = val_macro_f1 > best_val_macro_f1
            if is_best:
                best_val_macro_f1 = val_macro_f1

            # Record epoch metrics in history
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_macro_f1': val_macro_f1,
                'epoch_time_seconds': epoch_time,
                'total_elapsed_time_seconds': total_elapsed_time
            }
            history.append(epoch_metrics)

            # Save latest checkpoint after EVERY epoch
            save_checkpoint(
                LATEST_CHECKPOINT_PATH,
                epoch,
                model,
                optimizer,
                best_val_macro_f1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_macro_f1,
                history,
                total_elapsed_time
            )

            # Save separate best checkpoint if Macro-F1 improved
            if is_best:
                save_checkpoint(
                    BEST_CHECKPOINT_PATH,
                    epoch,
                    model,
                    optimizer,
                    best_val_macro_f1,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    val_macro_f1,
                    history,
                    total_elapsed_time
                )

            # Persist metrics history to JSON
            with open(METRICS_HISTORY_PATH, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

            # Print progress according to requirements
            print(f"Training loss:         {train_loss:.4f}", flush=True)
            print(f"Training accuracy:     {train_acc:.2f}%", flush=True)
            print(f"Validation loss:       {val_loss:.4f}", flush=True)
            print(f"Validation accuracy:   {val_acc:.2f}%", flush=True)
            print(f"Validation Macro-F1:   {val_macro_f1:.4f} {'(* Best)' if is_best else ''}", flush=True)
            print(f"Epoch time:            {epoch_time:.2f}s ({epoch_time / 60:.2f}m)", flush=True)
            print(f"Total elapsed time:    {total_elapsed_time:.2f}s ({total_elapsed_time / 60:.2f}m)", flush=True)

        print("\n" + "=" * 65, flush=True)
        print(f"PLANTVILLAGE TRAINING COMPLETE ({max_epochs}/{max_epochs} epochs)", flush=True)
        print(f"Best Validation Macro-F1: {best_val_macro_f1:.4f}", flush=True)
        print("=" * 65, flush=True)

    except KeyboardInterrupt:
        print("\n" + "=" * 65, flush=True)
        print("[PAUSED] PlantVillage training interrupted by user (Ctrl+C).", flush=True)
        if last_completed_epoch > 0:
            print(f"Latest completed epoch {last_completed_epoch} is safely saved in:", flush=True)
            print(f"  {LATEST_CHECKPOINT_PATH}", flush=True)
            print("You can resume anytime by keeping RESUME = True in train.py.", flush=True)
            print(f"When resumed, training will automatically continue from Epoch {last_completed_epoch + 1}/{max_epochs}.", flush=True)
        else:
            print("Interrupted before completing the first epoch. No completed checkpoint was saved.", flush=True)
        print("=" * 65, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train B-CNN on PlantVillage with pause/resume support.")
    parser.add_argument('--resume', action='store_true', default=RESUME, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=TOTAL_EPOCHS, help='Total epochs to train')
    args = parser.parse_args()

    run_training(resume=args.resume, max_epochs=args.epochs)
