"""Dataset utilities for CIFAR-10 cats vs dogs classification."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms


CIFAR10_CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_relevant_indices(
    dataset: torchvision.datasets.CIFAR10,
    classes: Iterable[str],
    target_classes: Iterable[str],
) -> list[int]:
    """Return indices for samples whose labels are in target_classes."""
    classes = list(classes)
    target_classes = set(target_classes)
    indices: list[int] = []

    for i in range(len(dataset)):
        _, label_idx = dataset[i]
        label_class = classes[label_idx]
        if label_class in target_classes:
            indices.append(i)

    return indices


def get_data_loader(
    target_classes: list[str] | None = None,
    batch_size: int = 64,
    data_dir: str = "./data",
    seed: int = 1000,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Load CIFAR-10 and return train, validation, and test loaders.

    The train split is created from CIFAR-10 train data after filtering to cats/dogs.
    80% is used for training and 20% for validation. The test split uses all
    CIFAR-10 test images belonging to the selected classes.
    """
    if target_classes is None:
        target_classes = ["cat", "dog"]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    relevant_indices = get_relevant_indices(trainset, CIFAR10_CLASSES, target_classes)

    np.random.seed(seed)
    np.random.shuffle(relevant_indices)
    split = int(len(relevant_indices) * 0.8)

    relevant_train_indices = relevant_indices[:split]
    relevant_val_indices = relevant_indices[split:]

    train_sampler = SubsetRandomSampler(relevant_train_indices)
    val_sampler = SubsetRandomSampler(relevant_val_indices)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    relevant_test_indices = get_relevant_indices(testset, CIFAR10_CLASSES, target_classes)
    test_sampler = SubsetRandomSampler(relevant_test_indices)
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, target_classes


def normalize_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    """Normalize CIFAR labels for selected binary classes into 0/1 labels."""
    labels = labels.float()
    max_val = torch.max(labels)
    min_val = torch.min(labels)
    return (labels - min_val) / (max_val - min_val)
