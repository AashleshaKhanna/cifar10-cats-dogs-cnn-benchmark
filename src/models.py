"""Model definitions for CIFAR-10 cats vs dogs experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LargeNet(nn.Module):
    """Larger convolutional network used in the lab."""

    name = "large"

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.fc1 = nn.Linear(10 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)


class SmallNet(nn.Module):
    """Smaller convolutional network used in the lab."""

    name = "small"

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(5 * 7 * 7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.pool(F.relu(self.conv(x))))
        x = x.view(-1, 5 * 7 * 7)
        x = self.fc(x)
        return x.squeeze(1)


class TwoLayerANN(nn.Module):
    """Two-layer fully connected ANN baseline using flattened RGB inputs."""

    name = "ann"

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(32 * 32 * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)


def build_model(model_name: str, hidden_dim: int = 256) -> nn.Module:
    """Factory for supported model architectures."""
    model_name = model_name.lower()

    if model_name == "large":
        return LargeNet()
    if model_name == "small":
        return SmallNet()
    if model_name == "ann":
        return TwoLayerANN(hidden_dim=hidden_dim)

    raise ValueError(f"Unknown model '{model_name}'. Choose from: large, small, ann.")
