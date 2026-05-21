"""
Training script for the PyTorch gesture classifier.

Trains CNN/ResNet18 on hand crop images with augmentation.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.config import CNN, PREPROCESSING
from src.deep_learning.gesture_net import get_model


def get_transforms(phase: str):
    """Get data transforms for train/val/test."""
    size = PREPROCESSING["classifier_input_size"]
    mean = PREPROCESSING["normalize_mean"]
    std = PREPROCESSING["normalize_std"]

    if phase == "train":
        return transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomCrop((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def train_classifier(data_dir: Path, output_path: Path, epochs: int = None):
    """
    Train gesture classifier.

    Args:
        data_dir: Directory with train/val/test splits.
        output_path: Path to save best checkpoint.
        epochs: Number of training epochs.
    """
    epochs = epochs or CNN["epochs"]
    batch_size = CNN["batch_size"]
    lr = CNN["lr"]
    patience = CNN["patience"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Datasets
    train_dataset = datasets.ImageFolder(
        root=str(data_dir / "train"),
        transform=get_transforms("train"),
    )
    val_dataset = datasets.ImageFolder(
        root=str(data_dir / "val"),
        transform=get_transforms("val"),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model
    num_classes = len(train_dataset.classes)
    model = get_model(num_classes=num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience // 2
    )

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}")

        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"  Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train gesture CNN classifier")
    parser.add_argument("--data", default="data/processed/classifier",
                        help="Path to classifier dataset directory")
    parser.add_argument("--output", default="models/gesture_cnn.pt",
                        help="Output model path")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_classifier(Path(args.data), Path(args.output), args.epochs)


if __name__ == "__main__":
    main()
