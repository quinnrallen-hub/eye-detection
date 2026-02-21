#!/usr/bin/env python3
"""
Train a CNN to classify eyes as open or closed.
Improved version with augmentation and better architecture.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# Config
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EyeDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if self.augment:
            img = self._augment(img)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        label = self.labels[idx]
        return img, label

    def _augment(self, img):
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # Random rotation (-15 to 15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        # Random noise
        if random.random() > 0.7:
            noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img


class EyeCNN(nn.Module):
    """Improved CNN with BatchNorm."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_dataset(coco_json, images_dir):
    """Load images and labels from COCO format."""
    with open(coco_json) as f:
        data = json.load(f)

    images_dir = Path(images_dir)

    id_to_file = {img['id']: img['file_name'] for img in data['images']}

    image_paths = []
    labels = []

    for ann in data['annotations']:
        img_file = id_to_file.get(ann['image_id'])
        if img_file:
            img_path = images_dir / img_file
            if img_path.exists():
                image_paths.append(img_path)
                # category_id 1 = open (label 0), category_id 2 = closed (label 1)
                labels.append(ann['category_id'] - 1)

    return image_paths, labels


def train():
    print(f"Using device: {DEVICE}")

    # Load data - use cropped eye images
    _base = Path(__file__).parent
    image_paths, labels = load_dataset(
        str(_base / "cropped_eyes_coco.json"),
        str(_base / "cropped_eyes")
    )

    print(f"Loaded {len(image_paths)} images")
    print(f"  Eyes open (0): {labels.count(0)}")
    print(f"  Eyes closed (1): {labels.count(1)}")

    # Split train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Augmentation only on training set
    train_dataset = EyeDataset(train_paths, train_labels, augment=True)
    val_dataset = EyeDataset(val_paths, val_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    model = EyeCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0
    patience = 0
    max_patience = 10

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels_batch in train_loader:
            images = images.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels_batch.size(0)
            train_correct += predicted.eq(labels_batch).sum().item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            save_path = str(_base / "eye_classifier.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (Val Acc: {val_acc:.1f}%)")
        else:
            patience += 1
            if patience >= max_patience:
                print("Early stopping!")
                break

    print(f"\nTraining complete! Best validation accuracy: {best_acc:.1f}%")
    print(f"Model saved to: {_base / 'eye_classifier.pth'}")


if __name__ == "__main__":
    train()
