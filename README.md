# CIFAR-10 Cats vs Dogs CNN Benchmark

A PyTorch image-classification project that compares convolutional neural networks (CNNs) and a fully connected neural network baseline on a binary **cats vs dogs** classification task using CIFAR-10.

This repository turns a university lab into a hiring-manager-friendly engineering project: clean modules, reproducible training scripts, model checkpoints, training curves, hyperparameter sweeps, and test-set evaluation.

## Why this project is useful

This project demonstrates practical ML engineering skills:

- Data filtering and reproducible train/validation/test splitting from CIFAR-10
- PyTorch model design with small and larger CNN architectures
- A non-convolutional ANN baseline for architecture comparison
- Training loop implementation with checkpointing and metrics logging
- Hyperparameter experiments for learning rate and batch size
- Evaluation on a held-out test set
- Visualization of training vs validation error/loss curves

## Dataset

The project uses CIFAR-10 and filters it to only the `cat` and `dog` classes.

Expected split:

| Split | Examples |
|---|---:|
| Train | 8,000 |
| Validation | 2,000 |
| Test | 2,000 |

CIFAR-10 is downloaded automatically by `torchvision`.

## Repository structure

```text
cifar10-cats-dogs-cnn-benchmark/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── plot_curves.py
│   └── utils.py
├── experiments/
│   └── run_experiments.py
├── checkpoints/
└── results/
```

## Quick start

```bash
git clone <your-repo-url>
cd cifar10-cats-dogs-cnn-benchmark

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Train a model

Train the larger CNN using the best lab hyperparameters:

```bash
python -m src.train \
  --model large \
  --batch-size 512 \
  --learning-rate 0.01 \
  --num-epochs 30
```

Train the smaller CNN:

```bash
python -m src.train --model small --batch-size 64 --learning-rate 0.01 --num-epochs 30
```

Train the ANN baseline:

```bash
python -m src.train --model ann --hidden-dim 256 --batch-size 128 --learning-rate 0.005 --num-epochs 30
```

## Evaluate the best checkpoint

```bash
python -m src.evaluate \
  --model large \
  --batch-size 512 \
  --learning-rate 0.01 \
  --epoch 29
```

## Plot training curves

```bash
python -m src.plot_curves \
  --model large \
  --batch-size 512 \
  --learning-rate 0.01 \
  --epoch 29
```

The script saves a PNG into `results/`.

## Run the lab-style experiment suite

```bash
python experiments/run_experiments.py
```

This runs selected CNN and ANN experiments from the lab, including different learning rates and batch sizes.

## Lab result summary

Representative results from the lab:

| Model | Batch size | Learning rate | Best/Final epoch | Test error | Test accuracy |
|---|---:|---:|---:|---:|---:|
| LargeNet CNN | 512 | 0.01 | 29 | 0.355 | 0.645 |
| TwoLayerANN | 128 | 0.005 | 16 | 0.3515 | 0.6485 |

The CNN is architecturally stronger because convolutions preserve spatial structure and share weights across image regions. The ANN baseline can perform competitively on this small binary CIFAR-10 task, but it flattens the image and discards spatial locality.

## Notes for hiring managers

This project highlights fundamentals expected in entry-level and early-career ML/AI engineering roles:

- Strong PyTorch training-loop fundamentals
- Debuggable, modular code instead of notebook-only implementation
- Experiment tracking through CSV metrics and checkpoint naming
- Understanding of overfitting, underfitting, validation curves, and held-out testing
- Clear comparison between CNN and MLP/ANN architectures
