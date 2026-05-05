"""Train CNN/ANN models on CIFAR-10 cats vs dogs."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_data_loader, normalize_binary_labels
from src.models import build_model
from src.utils import ensure_dirs, evaluate, get_model_name, save_metric_arrays


def train_net(
    model_name: str = "large",
    batch_size: int = 64,
    learning_rate: float = 0.01,
    num_epochs: int = 30,
    hidden_dim: int = 256,
    data_dir: str = "./data",
    device: str | None = None,
) -> str:
    """Train a model and save checkpoints/metrics.

    Returns the stem name for the final epoch outputs.
    """
    ensure_dirs()

    torch.manual_seed(1000)
    np.random.seed(1000)

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, _, _ = get_data_loader(
        target_classes=["cat", "dog"],
        batch_size=batch_size,
        data_dir=data_dir,
    )

    net = build_model(model_name, hidden_dim=hidden_dim).to(device_obj)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    start_time = time.time()

    for epoch in range(num_epochs):
        net.train()
        total_train_err = 0.0
        total_train_loss = 0.0
        total_epoch = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device_obj)
            labels = normalize_binary_labels(labels).to(device_obj)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predictions = (outputs > 0.0).long()
            total_train_err += int((predictions.cpu() != labels.cpu().long()).sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (len(train_loader))
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion, device_obj)

        print(
            f"Epoch {epoch + 1}: "
            f"Train err: {train_err[epoch]:.4f}, "
            f"Train loss: {train_loss[epoch]:.4f} | "
            f"Validation err: {val_err[epoch]:.4f}, "
            f"Validation loss: {val_loss[epoch]:.4f}"
        )

        checkpoint_stem = get_model_name(model_name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), f"checkpoints/{checkpoint_stem}.pt")

    elapsed_time = time.time() - start_time
    print(f"Finished training. Total time elapsed: {elapsed_time:.2f} seconds")

    final_stem = get_model_name(model_name, batch_size, learning_rate, num_epochs - 1)
    save_metric_arrays(final_stem, train_err, train_loss, val_err, val_loss)

    return final_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 cats vs dogs classifier.")
    parser.add_argument("--model", choices=["small", "large", "ann"], default="large")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_net(
        model_name=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        data_dir=args.data_dir,
        device=args.device,
    )
