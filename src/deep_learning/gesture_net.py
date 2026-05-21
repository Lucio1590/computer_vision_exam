"""
PyTorch gesture classifier.

Options:
    - ResNet18 (pretrained) with custom FC head
    - Custom lightweight CNN
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config import CNN


class GestureNet(nn.Module):
    """Gesture classification network."""

    def __init__(self, num_classes: int = None, architecture: str = None):
        """
        Initialize gesture classifier.

        Args:
            num_classes: Number of gesture classes.
            architecture: 'resnet18' or 'custom'.
        """
        super().__init__()
        num_classes = num_classes or CNN["num_classes"]
        architecture = architecture or CNN["architecture"]

        if architecture == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Replace final FC layer
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif architecture == "custom":
            self.model = self._build_custom_cnn(num_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _build_custom_cnn(self, num_classes: int) -> nn.Sequential:
        """Build a lightweight custom CNN."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Logits tensor (B, num_classes).
        """
        return self.model(x)


def get_model(num_classes: int = None, architecture: str = None):
    """
    Factory function to create model.

    Args:
        num_classes: Number of gesture classes.
        architecture: Model architecture name.

    Returns:
        torch.nn.Module instance.
    """
    return GestureNet(num_classes=num_classes, architecture=architecture)
