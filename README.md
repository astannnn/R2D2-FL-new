# R2D2-FL: Reliability-Weighted Robust Distillation for Federated Learning

This repository implements a modular Federated Learning (FL) framework for studying robustness under label noise and heterogeneous client data. It includes standard FL baselines and the proposed **Reliability-Weighted Robust Distillation (R2D2-FL)** method.

The project was developed and tested locally on macOS using Cursor/VS Code, with additional Jupyter/Colab notebooks kept only for validation and quick testing. The main implementation is contained in Python source files.

---

## Project Structure

```text
R2D2-FL-new/
│
├── main.py                  # Main dataset-aware training pipeline
├── config.py                # BaseConfig and dataset-specific configurations
├── README.md                # Repository documentation
├── technical_report.md      # Technical report
├── prepare_aptos.py         # APTOS preprocessing script
│
├── test_cifar.ipynb         # CIFAR-10 validation / Colab testing
├── test_emnist.ipynb        # EMNIST validation / Colab testing
├── test_aptos.ipynb         # APTOS validation / Colab testing
│
├── core/
│   ├── client.py            # Client-side training logic
│   ├── server.py            # Server-side aggregation and distillation logic
│   ├── partition.py         # Dirichlet non-IID partitioning
│   ├── models.py            # Model architectures
│
└── data/
    └── aptos_loader.py      # APTOS dataset loading code only
```

The repository intentionally excludes large local files such as datasets, virtual environments, checkpoints, logs, and generated experiment outputs.

---

## Repository and Dataset Policy

The GitHub repository contains only:

- Source code
- Configuration files
- Documentation
- Technical report
- Dataset loading code
- Validation notebooks

The following files and folders are excluded through `.gitignore`:

```text
.venv/
venv/
__pycache__/
*.pyc
data/aptos/
data/EMNIST/
data/cifar-10-batches-py/
data/cifar-10-python.tar.gz
results/
logs/
checkpoints/
*.pt
*.pth
*.pkl
*.zip
*.tar.gz
.DS_Store
```

This prevents large datasets, virtual environments, and generated files from being accidentally uploaded to GitHub.

---

## Important Note About Datasets

The datasets themselves are not stored in this repository.

### CIFAR-10

CIFAR-10 is downloaded automatically through `torchvision`.

### EMNIST

EMNIST is downloaded automatically through `torchvision`.

### APTOS 2019

APTOS 2019 is not included in the repository because of its size. It must be downloaded manually and stored locally.

Expected local structure:

```text
data/aptos/
    train.csv
    train_images/
```

The repository includes only the code required to prepare and load the APTOS dataset:

```text
prepare_aptos.py
data/aptos_loader.py
```

---

## Quick Validation in Google Colab

The repository includes three Colab-ready notebooks for quick pipeline validation:

- **CIFAR-10 validation notebook**  
  https://colab.research.google.com/github/astannnn/R2D2-FL-new/blob/main/test_cifar.ipynb

- **EMNIST validation notebook**  
  https://colab.research.google.com/github/astannnn/R2D2-FL-new/blob/main/test_emnist.ipynb

- **APTOS validation notebook**  
  https://colab.research.google.com/github/astannnn/R2D2-FL-new/blob/main/test_aptos.ipynb

These notebooks are intended only for quick validation and Colab-based testing. They verify that the pipeline can load data, create client partitions, run a minimal training setup, and complete basic evaluation.

They are not the main implementation. The main training logic is implemented in:

```text
main.py
config.py
core/
data/aptos_loader.py
prepare_aptos.py
```

---

## Configuration

All main hyperparameters are defined in `config.py`.

### Configuration Classes

- `BaseConfig`
- `CIFARConfig`
- `EMNISTConfig`
- `APTOSConfig`

Example usage:

```python
from config import CIFARConfig
from main import main

config = CIFARConfig()
main(config)
```

### Key Parameters

```text
DATASET
NUM_CLASSES
IN_CHANNELS
NUM_CLIENTS
CLIENT_FRACTION
LOCAL_EPOCHS
ROUNDS
BATCH_SIZE
LR
DIRICHLET_ALPHA
NOISE_RATE
NOISE_TYPE
PROXY_SIZE
SEED
USE_FEDPROX
USE_FEDDF
USE_R2D2
USE_SELECTIVE_FD
USE_FEDNORO
```

---

## Implemented Methods

### FedAvg

Standard federated averaging baseline with local client training and weighted server aggregation.

### FedProx

FedAvg extension with a proximal regularization term to reduce client drift under heterogeneous data.

### R2D2-FL

The proposed method combines client-side noise-aware training and server-side reliability-weighted proxy distillation.

#### Client-side components

- Confidence-based sample selection
- Soft label correction
- Local knowledge distillation

#### Server-side components

- Client-level reliability estimation
- Class-level reliability weighting
- Reliability-weighted ensemble teacher
- Proxy-based global distillation

### Additional Method Switches

The code also includes method switches for comparison and ablation experiments:

```text
USE_FEDDF
USE_SELECTIVE_FD
USE_FEDNORO
```

---

## Federated Data Simulation

The framework supports:

- Dirichlet-based non-IID partitioning
- Configurable number of clients
- Configurable client sampling fraction
- Symmetric label noise
- Asymmetric label noise
- Heterogeneous client-level corruption

Noise is injected after client partitioning, which allows each client to have its own local corruption pattern.

---

## Experimental Campaign

The experiments were run for each dataset under the following noise settings:

- 0% noise
- 20% symmetric noise
- 40% symmetric noise
- 40% asymmetric noise
- 40% heterogeneous client-level noise

The experiments were repeated using three random seeds, as required for reproducibility.

An ablation study was also conducted, but not across all noise settings. The ablation experiments were focused on the main noisy setting in order to evaluate the contribution of individual R2D2-FL components.

---

## Running Experiments Locally

Clone the repository:

```bash
git clone https://github.com/astannnn/R2D2-FL-new.git
cd R2D2-FL-new
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages. If a `requirements.txt` file is available, use:

```bash
pip install -r requirements.txt
```

Otherwise, install the main dependencies manually:

```bash
pip install torch torchvision numpy scikit-learn pandas pillow matplotlib
```

Run the default configuration:

```bash
python main.py
```

Before running a different dataset or method, update the relevant settings in `config.py`.

---

## Running Full APTOS Experiments

To run full APTOS experiments:

1. Download the official dataset:

   **APTOS 2019 Blindness Detection**  
   https://www.kaggle.com/competitions/aptos2019-blindness-detection

2. Place the files locally:

```text
data/aptos/
    train.csv
    train_images/
```

3. If preprocessing is required, run:

```bash
python prepare_aptos.py
```

4. Run the experiment:

```python
from config import APTOSConfig
from main import main

config = APTOSConfig()
main(config)
```

---

## Evaluation Metrics

The framework reports:

- Global test accuracy
- Worst-client accuracy
- Macro-F1 score
- Round-wise training time
- Communication cost estimate
- Convergence across communication rounds
- Best round performance

For APTOS, Macro-F1 is especially important because the dataset is imbalanced.

---

## Reproducibility

The project supports reproducibility through:

- Fixed random seeds
- Deterministic data partitioning
- Configuration-based experiment control
- Round-wise logging
- Dataset-specific configuration classes
- Clean GitHub repository without local datasets or virtual environment files

---

## Current Repository Status

The current repository state is:

- Source code is pushed to GitHub
- `.venv/` is excluded from Git tracking
- Large datasets are excluded from Git tracking
- APTOS dataset remains local
- APTOS loading code is tracked in GitHub
- CIFAR-10 and EMNIST are handled through automatic download
- Jupyter notebooks are kept only for validation and Colab testing

This keeps the repository clean, lightweight, and suitable for academic review.

---

## Summary

R2D2-FL introduces reliability-aware aggregation through proxy-based distillation and evaluates robustness under:

- Non-IID client distributions
- Symmetric and asymmetric label noise
- Heterogeneous client-level corruption
- Medical image classification with APTOS 2019

The framework is modular, extensible, and designed for experimentation in noisy federated learning settings.
