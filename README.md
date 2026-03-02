# R2D2-FL: Reliability-Weighted Robust Distillation for Federated Learning

This repository implements a modular Federated Learning (FL) framework for studying robustness under label noise, including the proposed **Reliability-Weighted Robust Distillation (R2D2-FL)** method.

The project supports multiple datasets and provides reproducible Google Colab notebooks for full pipeline validation.

---

## 📂 Project Structure

```
R2D2-FL/
│
├── main.py                  # Dataset-agnostic training pipeline
├── config.py                # BaseConfig + dataset-specific configurations
├── README.md
├── technical_report.md
│
├── test_cifar.ipynb         # CIFAR-10 sanity notebook (Colab-ready)
├── test_emnist.ipynb        # EMNIST sanity notebook (Colab-ready)
├── test_aptos.ipynb         # APTOS sanity notebook (Colab-ready)
│
├── core/
│   ├── client.py            # Client-side training (FedAvg, FedProx, R2D2)
│   ├── server.py            # Aggregation and distillation logic
│   ├── partition.py         # Dirichlet non-IID partitioning
│   ├── models.py            # Model architectures
│
└── data/
    └── aptos_loader.py      # APTOS dataset loader
```

---

## ▶ Quick Pipeline Validation (Google Colab)

The following notebooks validate the full training pipeline in a clean environment:

- **CIFAR-10 (Sanity Check)**  
  https://colab.research.google.com/github/astannnn/R2D2-FL/blob/main/test_cifar.ipynb

- **EMNIST (Sanity Check)**  
  https://colab.research.google.com/github/astannnn/R2D2-FL/blob/main/test_emnist.ipynb

- **APTOS (Sanity Check – Mini Fallback Dataset)**  
  https://colab.research.google.com/github/astannnn/R2D2-FL/blob/main/test_aptos.ipynb

Each notebook:

- Automatically clones the repository  
- Runs a minimal configuration (1 round, 1 epoch)  
- Verifies data loading, partitioning, training, aggregation, and evaluation  

These notebooks are intended for reproducibility validation only.

---

## ⚙ Configuration

All hyperparameters are defined in `config.py`.

### Configuration Classes

- `BaseConfig` – shared hyperparameters  
- `CIFARConfig`  
- `EMNISTConfig`  
- `APTOSConfig`  

The `main(config)` function is fully dataset-agnostic and dynamically adapts to the selected configuration.

### Key Parameters

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

---

## 📊 Supported Datasets

### CIFAR-10

Automatically downloaded via `torchvision`.

### EMNIST (Digits split)

Automatically downloaded via `torchvision`.

### APTOS 2019 (Diabetic Retinopathy Dataset)

If the dataset is not found locally, a small synthetic fallback dataset is automatically generated to ensure the training pipeline executes correctly.

---

## 🏥 Running Full APTOS Experiments

To execute full medical experiments:

1. Download the official dataset:  
   **APTOS 2019 Blindness Detection**  
   https://www.kaggle.com/competitions/aptos2019-blindness-detection

2. Place the dataset in the following structure:

```
data/aptos/
    train_images/
    train.csv
```

3. Run using:

```python
from config import APTOSConfig
from main import main

config = APTOSConfig()
main(config)
```

---

## 🧠 Federated Data Simulation

- Dirichlet-based non-IID client partitioning (α configurable)
- Symmetric label noise
- Asymmetric label noise
- Heterogeneous client-level corruption

Noise is injected after client partitioning.

---

## 🏗 Implemented Methods

### Baselines

- FedAvg
- FedProx

### Proposed Method: R2D2-FL

#### Client-side

- Confidence-based sample selection
- Soft label correction
- Local knowledge distillation

#### Server-side

- Client-level reliability estimation
- Class-level reliability weighting
- Reliability-weighted ensemble teacher
- Proxy-based global distillation

---

## ▶ Running Experiments (Local / VM)

```bash
git clone https://github.com/astannnn/R2D2-FL.git
cd R2D2-FL
python main.py
```

Modify configuration parameters in `config.py` before execution.

---

## 📈 Evaluation Metrics

- Global test accuracy
- Worst-client accuracy
- Macro-F1 (APTOS)
- Convergence across rounds

Results are averaged across multiple random seeds.

---

## 🔁 Reproducibility

- Fixed random seeds
- Deterministic data partitioning
- Fully configurable hyperparameters
- Round-wise logging

---

## 🎯 Summary

R2D2-FL introduces reliability-aware aggregation via proxy-based distillation and evaluates robustness under:

- Non-IID client distributions
- Symmetric and asymmetric label noise
- Heterogeneous client corruption

The framework is modular, extensible, and designed for research-grade experimentation in noisy federated settings.
