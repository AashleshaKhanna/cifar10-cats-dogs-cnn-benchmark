"""Evaluate a saved model checkpoint on the cats vs dogs test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data import get_data_loader
from src.models import build_model
from src.utils import evaluate, get_model_name


def evaluate_checkpoint(
    model_name: str,
    batch_size: int,
    learning_rate: float,
    epoch: int,
    hidden_dim: int = 256,
    data_dir: str = "./data",
    device: str | None = None,
) -> tuple[float, float]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    _, _, test_loader, _ = get_data_loader(
        target_classes=["cat", "dog"],
        batch_size=batch_size,
        data_dir=data_dir,
    )

    net = build_model(model_name, hidden_dim=hidden_dim).to(device_obj)
    checkpoint_stem = get_model_name(model_name, batch_size, learning_rate, epoch)
    checkpoint_path = Path("checkpoints") / f"{checkpoint_stem}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint: {checkpoint_path}. Train the model first."
        )

    state = torch.load(checkpoint_path, map_location=device_obj)
    net.load_state_dict(state)

    criterion = nn.BCEWithLogitsLoss()
    test_err, test_loss = evaluate(net, test_loader, criterion, device_obj)

    print(f"Test classification error: {test_err:.4f}")
    print(f"Test classification accuracy: {1.0 - test_err:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    return test_err, test_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved cats vs dogs classifier.")
    parser.add_argument("--model", choices=["small", "large", "ann"], default="large")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=29)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        model_name=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epoch=args.epoch,
        hidden_dim=args.hidden_dim,
        data_dir=args.data_dir,
        device=args.device,
    )
