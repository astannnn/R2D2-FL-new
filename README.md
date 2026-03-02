# R2D2-FL: Reliability-Weighted Robust Distillation for Federated Learning

## Project Structure

```
.
├── main.py                # Main training script
├── config.py              # Hyperparameter configuration
├── README.md
├── technical_report.md
│
├── core/                  # Core Federated Learning logic
│   ├── __init__.py
│   ├── client.py          # Client-side local training (FedAvg, FedProx, R2D2)
│   ├── server.py          # Server aggregation and distillation logic
│   ├── partition.py       # Dirichlet data partitioning
│   ├── models.py          # Model architectures
│
└── data/
    └── aptos_loader.py    # APTOS medical dataset loader
```

---

## Requirements

- Python 3.9+
- PyTorch

Install dependencies:

```bash
pip install torch torchvision numpy tqdm matplotlib
```

---

## Supported Datasets

### CIFAR-10
Automatically downloaded via torchvision.

### EMNIST (Digits split)
Automatically downloaded via torchvision.

### APTOS 2019 (Medical Retinopathy Dataset)

If the dataset is not found locally, a small synthetic fallback dataset is automatically generated to verify that the training pipeline runs correctly.

To run full APTOS experiments:

Download the dataset from Kaggle:
https://www.kaggle.com/competitions/aptos2019-blindness-detection

Extract training images into:

```
data/aptos/train/
```

Folder structure:

```
data/aptos/train/
 ├── 0/
 ├── 1/
 ├── 2/
 ├── 3/
 └── 4/
```

---

## Configuration

All hyperparameters are defined in `config.py`.

Configuration classes:

- `BaseConfig`
- `CIFARConfig`
- `EMNISTConfig`
- `APTOSConfig`

Key parameters:

- `NUM_CLIENTS`
- `CLIENT_FRACTION`
- `LOCAL_EPOCHS`
- `ROUNDS`
- `LR`
- `DIRICHLET_ALPHA`
- `NOISE_RATE`
- `NOISE_TYPE`
- `PROXY_SIZE`
- `USE_R2D2`
- `USE_FEDPROX`
- `SEED`

To switch datasets, modify the dataset selection inside `main.py`.

---

## Federated Data Simulation

- Dirichlet-based non-IID partitioning (α configurable)
- Symmetric label noise
- Asymmetric label noise
- Heterogeneous client-level corruption

Noise injection is applied after client partitioning.

---

## Implemented Methods

### Baselines

- FedAvg
- FedProx

### Proposed Method

**R2D2-FL**

Client-side:
- Confidence-based sample selection
- Soft label correction
- Local knowledge distillation

Server-side:
- Client-level reliability estimation
- Class-level reliability weighting
- Reliability-weighted ensemble teacher
- Proxy-based global distillation

---

## Running Experiments

```bash
python main.py
```

Control method via:

```python
USE_R2D2 = True  # or False
USE_FEDPROX = True  # or False
```

---

## Evaluation Metrics

- Global test accuracy
- Worst-client accuracy
- Macro-F1 (APTOS)
- Convergence curves

Results are averaged across multiple seeds.

---

## Reproducibility

- Fixed random seeds
- Deterministic data partitioning
- Fully configurable hyperparameters
- Round-wise logging

---

## Summary

R2D2-FL introduces reliability-aware aggregation through proxy-based distillation.

The framework benchmarks robustness under:

- Non-IID distributions
- Symmetric and asymmetric label noise
- Heterogeneous client corruption
