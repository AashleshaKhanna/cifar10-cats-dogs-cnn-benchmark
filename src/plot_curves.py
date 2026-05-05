"""Plot train/validation error and loss curves from saved CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_model_name


def plot_training_curve(stem: str) -> Path:
    train_err = np.loadtxt(f"results/{stem}_train_err.csv")
    val_err = np.loadtxt(f"results/{stem}_val_err.csv")
    train_loss = np.loadtxt(f"results/{stem}_train_loss.csv")
    val_loss = np.loadtxt(f"results/{stem}_val_loss.csv")

    epochs = np.arange(1, len(train_err) + 1)

    plt.figure(figsize=(8, 5))
    plt.title("Train vs Validation Error")
    plt.plot(epochs, train_err, label="Train")
    plt.plot(epochs, val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    error_path = Path("results") / f"{stem}_error_curve.png"
    plt.savefig(error_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.title("Train vs Validation Loss")
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    loss_path = Path("results") / f"{stem}_loss_curve.png"
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    print(f"Saved {error_path}")
    print(f"Saved {loss_path}")
    return error_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot saved training curves.")
    parser.add_argument("--model", choices=["small", "large", "ann"], default="large")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=29)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stem = get_model_name(args.model, args.batch_size, args.learning_rate, args.epoch)
    plot_training_curve(stem)
