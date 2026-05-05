"""Shared utilities for training, checkpointing, and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data import normalize_binary_labels


def get_model_name(model_name: str, batch_size: int, learning_rate: float, epoch: int) -> str:
    """Create a checkpoint/metrics stem from hyperparameters."""
    return f"model_{model_name}_bs{batch_size}_lr{learning_rate}_epoch{epoch}"


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)


@torch.no_grad()
def evaluate(
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """Return classification error and average loss on a dataloader."""
    net.eval()
    total_err = 0.0
    total_loss = 0.0
    total_epoch = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = normalize_binary_labels(labels).to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        predictions = (outputs > 0.0).long()
        total_err += int((predictions.cpu() != labels.cpu().long()).sum())
        total_loss += loss.item()
        total_epoch += len(labels)

    err = float(total_err) / total_epoch
    avg_loss = float(total_loss) / len(loader)
    return err, avg_loss


def save_metric_arrays(
    stem: str,
    train_err: np.ndarray,
    train_loss: np.ndarray,
    val_err: np.ndarray,
    val_loss: np.ndarray,
) -> None:
    """Save training curves as CSV files."""
    np.savetxt(f"results/{stem}_train_err.csv", train_err)
    np.savetxt(f"results/{stem}_train_loss.csv", train_loss)
    np.savetxt(f"results/{stem}_val_err.csv", val_err)
    np.savetxt(f"results/{stem}_val_loss.csv", val_loss)
