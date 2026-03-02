# R2D2-FL: Reliability-Weighted Robust Distillation for Federated Learning

This repository contains the implementation of **R2D2-FL**, a reliability-aware federated learning framework designed to improve robustness under heterogeneous and noisy client environments.

The method integrates reliability estimation and proxy-based knowledge distillation to mitigate the impact of corrupted or unreliable clients in non-IID federated settings.

---

## Project Structure
main.py # Main training script
config.py # Hyperparameter configuration
core/ # Core FL implementation
├── client.py
├── server.py
├── reliability.py
├── distillation.py
├── partition.py
└── models.py


---

## Requirements

- Python 3.9+
- PyTorch

Install dependencies:

```bash
pip install torch torchvision numpy tqdm matplotlib
Supported Datasets

Currently implemented:

CIFAR-10

EMNIST

APTOS 2019 (Medical Dataset)

CIFAR-10 and EMNIST are automatically downloaded via torchvision.

APTOS requires manual dataset preparation.

Federated Data Simulation

The framework supports realistic non-IID and noisy federated scenarios:

Dirichlet-based data partitioning (α configurable)

Symmetric label noise

Asymmetric label noise

Heterogeneous client-level corruption

Noise injection is performed after client data partitioning.

Implemented Methods
Baselines

FedAvg

FedProx

Proposed Method

R2D2-FL, including:

Client-side:

Confidence-based sample selection

Soft label correction

Local knowledge distillation

Server-side:

Client-level reliability estimation

Class-level reliability weighting

Reliability-weighted ensemble teacher construction

Proxy-based global distillation

Configuration

All hyperparameters are defined in config.py.

Key parameters include:

NUM_CLIENTS

CLIENT_FRACTION

LOCAL_EPOCHS

ROUNDS

LR

DIRICHLET_ALPHA

NOISE_RATE

NOISE_TYPE

PROXY_SIZE

SEED

USE_R2D2

USE_FEDPROX

Modify config.py before running experiments.

Running Experiments

To start training:

python main.py

Experiment type is controlled via:

USE_R2D2 = True/False

USE_FEDPROX = True/False

Evaluation Metrics

Global test accuracy

Worst-client accuracy

Macro-F1 score (APTOS)

Results are averaged across multiple seeds for statistical stability.

Reproducibility

Fixed random seeds

Deterministic data partitioning

Fully configurable hyperparameters

Logging of training rounds

Summary

R2D2-FL introduces reliability-aware aggregation through proxy-based distillation.

The implementation provides a structured benchmark for evaluating robustness under:

Non-IID data distributions

Symmetric and asymmetric label corruption

Client-level corruption variability
