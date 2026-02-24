# R2D2-FL: Reliability-Weighted Robust Distillation for Federated Learning

This repository contains the implementation of R2D2-FL, a reliability-aware federated learning framework designed to improve robustness under heterogeneous and noisy client settings.

## Structure

- main.py – main training script
- config.py – configuration file with hyperparameters
- core/ – implementation of server, client, reliability, distillation, and training modules

## Requirements

Python 3.9+

Install dependencies:

pip install torch torchvision numpy tqdm matplotlib

## Dataset

Currently implemented:
- CIFAR-10 (automatically downloaded via torchvision)

Non-IID simulation:
- Dirichlet partitioning (alpha configurable in config.py)
- Symmetric noise
- Asymmetric noise
- Heterogeneous client-level noise

## Configuration

All hyperparameters are defined in config.py:

- NUM_CLIENTS
- CLIENT_FRACTION
- LOCAL_EPOCHS
- ROUNDS
- LR
- DIRICHLET_ALPHA
- NOISE_RATE
- NOISE_TYPE
- PROXY_SIZE
- SEED

Modify config.py before running experiments.

## Running Experiments

To run training:

python main.py

Experiment type (FedAvg or R2D2) is controlled inside main.py or config.py.

## Implemented Components

Client-side:
- Confidence-based sample selection
- Soft label correction
- Local knowledge distillation

Server-side:
- Client-level reliability estimation
- Class-level reliability weighting
- Reliability-weighted ensemble teacher
- Proxy-based distillation

## Reproducibility

- Fixed random seed
- Deterministic partitioning
- Configurable hyperparameters
