"""Run selected lab-style experiments.

This is intentionally small so it can be launched locally or on Colab.
Comment out experiments you do not want to run.
"""

from src.train import train_net


EXPERIMENTS = [
    # CNN architecture comparison
    {"model_name": "small", "batch_size": 64, "learning_rate": 0.01, "num_epochs": 30},
    {"model_name": "large", "batch_size": 64, "learning_rate": 0.01, "num_epochs": 30},

    # Learning-rate experiments for LargeNet
    {"model_name": "large", "batch_size": 64, "learning_rate": 0.001, "num_epochs": 30},
    {"model_name": "large", "batch_size": 64, "learning_rate": 0.1, "num_epochs": 30},

    # Batch-size experiments for LargeNet
    {"model_name": "large", "batch_size": 512, "learning_rate": 0.01, "num_epochs": 30},
    {"model_name": "large", "batch_size": 16, "learning_rate": 0.01, "num_epochs": 30},

    # ANN baseline
    {
        "model_name": "ann",
        "hidden_dim": 256,
        "batch_size": 128,
        "learning_rate": 0.005,
        "num_epochs": 30,
    },
]


if __name__ == "__main__":
    for config in EXPERIMENTS:
        print("=" * 80)
        print(f"Running experiment: {config}")
        print("=" * 80)
        train_net(**config)
