# Technical Report  
## R2D2-FL: Reliability-Weighted Robust Distillation for Federated Learning

---

## 1. Introduction

This report presents the implementation and experimental validation of **R2D2-FL (Reliability-Weighted Robust Distillation for Federated Learning)** under heterogeneous and noisy federated learning environments.

The objective of this project is to evaluate whether reliability-aware aggregation combined with proxy-based knowledge distillation can improve robustness compared to standard federated optimization methods in realistic non-IID and noisy-label settings.

The project includes:

- Implementation of **FedAvg** baseline
- Implementation of **FedProx** baseline
- Modular implementation of **R2D2-FL**
- Dirichlet-based non-IID simulation with α = 0.3
- Multiple label noise regimes
- Experiments on **CIFAR-10, EMNIST, and APTOS 2019**
- Three-seed experimental evaluation
- Ablation study for R2D2-FL components
- Reproducible configuration control
- GitHub repository containing only source code, configuration files, documentation, and validation notebooks

Large datasets, virtual environments, checkpoints, and generated outputs are intentionally excluded from the repository.

---

## 2. Development and Repository Context

Earlier parts of the project were executed through GitHub and Google Colab due to limited local computational resources. After moving to a Mac environment, the project was further developed and tested locally using Cursor/VS Code.

The final repository is organized so that the professor can clone or pull the project without downloading unnecessary large files.

The repository contains:

```text
core/
config.py
main.py
prepare_aptos.py
data/aptos_loader.py
README.md
technical_report.md
test_cifar.ipynb
test_emnist.ipynb
test_aptos.ipynb
.gitignore
```

The Jupyter notebooks are included only for dataset validation, Google Colab testing, and quick experimental checks. The main implementation is contained in the Python source files.

The following files are excluded from GitHub:

```text
.venv/
venv/
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
```

This avoids GitHub size limitations and keeps the repository clean.

---

## 3. Dataset Management

The project uses three datasets.

### CIFAR-10

CIFAR-10 is downloaded automatically through `torchvision`.

### EMNIST

EMNIST is downloaded automatically through `torchvision`.

### APTOS 2019

APTOS 2019 is used as the medical image classification dataset.

The full APTOS dataset is stored locally and is not uploaded to GitHub.

Expected local structure:

```text
data/aptos/
    train.csv
    train_images/
```

The repository includes only the code required to prepare and load the dataset:

```text
prepare_aptos.py
data/aptos_loader.py
```

---

## 4. Configuration Design

The implementation follows a dataset-specific configuration architecture.

A shared `BaseConfig` class defines common federated learning settings and hyperparameters.

Specialized configuration classes extend it:

- `CIFARConfig`
- `EMNISTConfig`
- `APTOSConfig`

Each configuration defines:

- Dataset name
- Number of classes
- Input channels
- Number of clients
- Client fraction
- Local epochs
- Communication rounds
- Batch size
- Learning rate
- Dirichlet alpha
- Noise rate
- Noise type
- Proxy dataset size
- Method switches
- Random seed

The training pipeline can be adapted by changing the selected configuration instead of modifying the core training logic.

This provides:

- Clean separation of experiment settings
- Consistent comparison between methods
- Easy hyperparameter control
- Reproducibility across seeds
- Extensibility to additional datasets

---

## 5. Baseline Methods

### 5.1 FedAvg

FedAvg is implemented as the standard federated parameter averaging baseline.

It includes:

- Random client sampling
- Local client training
- Weighted aggregation based on local dataset size
- Shared global model architecture
- Evaluation after each communication round

FedAvg is used as the primary reference method.

---

### 5.2 FedProx

FedProx extends FedAvg by adding a proximal regularization term to the local training objective:

\[
\mathcal{L}_{prox} = \frac{\mu}{2} \| w_k - w_t \|^2
\]

This term is designed to reduce client drift under heterogeneous data distributions.

FedProx is included to evaluate whether proximal regularization alone improves robustness under noisy federated settings.

---

### 5.3 Additional Method Switches

The implementation includes method switches for comparison and ablation experiments:

```text
USE_FEDPROX
USE_FEDDF
USE_R2D2
USE_SELECTIVE_FD
USE_FEDNORO
```

These switches allow different methods and components to be enabled or disabled without rewriting the training pipeline.

---

## 6. Data Partitioning and Noise Modeling

### 6.1 Non-IID Partitioning

Client data is partitioned using a Dirichlet distribution.

The main setting uses:

```text
DIRICHLET_ALPHA = 0.3
```

This creates heterogeneous class distributions across clients.

The partitioning process is controlled by the random seed, which makes experiments reproducible.

---

### 6.2 Noise Regimes

The experimental campaign was conducted under the following noise settings for each dataset:

- 0% noise
- 20% symmetric label noise
- 40% symmetric label noise
- 40% asymmetric label noise
- 40% heterogeneous client-level noise

Noise is injected after client partitioning. This is important because it allows each client to have its own local corruption pattern.

In the heterogeneous noise setting, only a subset of clients receives corrupted labels. This allows the framework to test robustness against unreliable clients.

---

## 7. R2D2-FL Framework

R2D2-FL is designed to improve robustness in noisy federated learning.

The method combines:

1. Client-side local training
2. Confidence-based sample handling
3. Soft label correction
4. Local knowledge distillation
5. Server-side reliability estimation
6. Reliability-weighted proxy distillation

Instead of relying only on direct parameter averaging, R2D2-FL uses proxy-based knowledge distillation to update the global model using information from selected client models.

---

## 7.1 Client-Side Components

Each selected client receives the current global model and performs local training.

### 1. Confidence-Based Sample Selection

For each local sample, the client estimates prediction confidence.

High-confidence samples are treated as more reliable, while low-confidence samples are handled with soft label correction.

---

### 2. Soft Label Correction

For low-confidence samples, the original label is mixed with the global teacher prediction:

\[
\tilde{y}_i = \lambda y_i + (1 - \lambda) p_{global}(x_i)
\]

This reduces the effect of incorrect labels while still preserving information from the original annotation.

---

### 3. Local Knowledge Distillation

The local model is encouraged to stay close to the global teacher distribution:

\[
\mathcal{L}_{locKD} = KL(p_{global} \| p_{local})
\]

The final local objective is:

\[
\mathcal{L}_k = \mathcal{L}_{sup} + \beta \mathcal{L}_{locKD}
\]

This helps reduce overfitting to corrupted local data.

---

## 7.2 Server-Side Components

After local training, the server receives selected client models and performs reliability-aware aggregation through proxy-based distillation.

### 1. Reliability Estimation

Using the proxy dataset, the server evaluates each selected client model.

Reliability is estimated using:

- Agreement with ensemble predictions
- Client-level reliability
- Class-level reliability
- Prediction confidence or entropy

This allows the server to reduce the influence of unreliable or corrupted clients.

---

### 2. Reliability-Weighted Ensemble Teacher

The server constructs an ensemble teacher by weighting client predictions according to reliability.

The weighting considers:

- Overall client reliability
- Class-specific reliability
- Prediction entropy

This produces a more stable teacher distribution for proxy-based distillation.

---

### 3. Proxy-Based Global Distillation

The global model is updated by minimizing the KL divergence between the ensemble teacher and the global model:

\[
\mathcal{L}_{KD} = KL(p_{teacher} \| p_{global})
\]

The purpose is to transfer reliable client knowledge into the global model while limiting the effect of noisy clients.

---

## 8. Communication Round Procedure

Each communication round follows this structure:

1. Select a subset of clients
2. Broadcast the current global model
3. Train selected clients locally
4. Collect updated local client models
5. Evaluate client predictions on the proxy dataset
6. Estimate client reliability
7. Build a reliability-weighted ensemble teacher
8. Distill the global model on proxy data
9. Evaluate global performance
10. Log metrics for analysis

This process is repeated for the configured number of communication rounds.

---

## 9. Experimental Setup

### 9.1 Datasets

The project uses three datasets:

- CIFAR-10
- EMNIST
- APTOS 2019

CIFAR-10 and EMNIST are downloaded automatically through `torchvision`. APTOS 2019 is stored locally and loaded through custom dataset loading code.

---

### 9.2 Experimental Conditions

Each dataset was evaluated under the following noise settings:

```text
0% noise
20% symmetric noise
40% symmetric noise
40% asymmetric noise
40% heterogeneous noise
```

Each experiment was repeated using three random seeds.

This seed-based evaluation was used to reduce randomness and provide more reliable comparisons.

---

### 9.3 Metrics

The framework reports:

- Global test accuracy
- Worst-client accuracy
- Macro-F1 score
- Round-wise training time
- Communication cost estimate
- Best round performance
- Convergence behavior across communication rounds

For APTOS, Macro-F1 is especially important because the dataset is imbalanced.

---

## 10. Experimental Findings

### 10.1 CIFAR-10

CIFAR-10 is used as the main controlled benchmark.

Observed behavior:

- FedAvg performs well in clean settings.
- Under strong symmetric noise, robustness becomes more challenging.
- R2D2-FL is designed to improve stability by reducing the influence of unreliable clients.
- Worst-client accuracy is important because global accuracy alone may hide poor client-level performance.

---

### 10.2 EMNIST

EMNIST provides a lighter benchmark for testing the pipeline across multiple seeds and configurations.

Observed behavior:

- Training is faster than APTOS.
- It is useful for validating method switches and ablation settings.
- Performance can still be sensitive under strong label noise.

---

### 10.3 APTOS 2019

APTOS is the most computationally expensive dataset in the project.

Observed behavior:

- Training is significantly slower than CIFAR-10 and EMNIST.
- The dataset is highly imbalanced.
- Macro-F1 is necessary for fair evaluation.
- Results can fluctuate more strongly due to medical image complexity and class imbalance.
- Running many APTOS experiments is computationally expensive, but the required three-seed experiments were still performed.

---

## 11. Ablation Study

An ablation study was conducted to analyze the contribution of the main R2D2-FL components.

The ablation study was not repeated across every noise setting. Instead, it focused on the main noisy setting in order to isolate the effect of each component.

Ablation variants include:

- Removing reliability weighting
- Removing class-level reliability
- Removing soft label correction
- Removing local knowledge distillation

Expected observations:

- Removing local KD may reduce consistency between local and global models.
- Removing soft label correction may increase sensitivity to noisy labels.
- Removing reliability weighting may allow corrupted clients to influence the global teacher more strongly.
- Removing class-level reliability may reduce robustness when specific classes are corrupted more heavily.

Robustness is expected to come from the combination of all components rather than one isolated mechanism.

---

## 12. Reproducibility

The project supports reproducibility through:

- Fixed random seeds
- Deterministic client partitioning
- Configuration-based experiment setup
- Consistent logging
- Dataset-specific configuration classes
- Clean GitHub repository without local datasets or virtual environment files

---

## 13. Current Project Status

The current project state is:

- Code repository is pushed to GitHub
- Large datasets are excluded from Git tracking
- Virtual environment is excluded from Git tracking
- `.gitignore` prevents accidental dataset uploads
- APTOS dataset remains local
- APTOS loader code is tracked in GitHub
- CIFAR-10 and EMNIST are handled through automatic download
- Main training pipeline is configurable through method switches
- Jupyter notebooks are available only for validation and Colab-based testing
- Experiments were run using the required three seeds
- Ablation was performed separately and not across all noise regimes

This repository state is appropriate because GitHub should contain the implementation and documentation, while large datasets should remain local or be downloaded through scripts.

---

## 14. Limitations

The main limitations of the current project are:

- APTOS experiments are computationally expensive.
- APTOS class imbalance makes global accuracy insufficient by itself.
- Some methods require careful hyperparameter tuning.
- Results may differ depending on device type, batch size, and available compute resources.
- APTOS requires manual local dataset placement because it cannot be automatically downloaded from GitHub.

These limitations should be clearly mentioned in the final discussion.

---

## 15. Final Conclusions

This project delivers a modular implementation of R2D2-FL for robust federated learning under noisy labels and heterogeneous client data.

The main contributions are:

- A complete federated learning training pipeline
- Support for multiple datasets
- Support for non-IID data partitioning
- Support for multiple noise regimes
- Three-seed experimental evaluation
- Implementation of baseline methods
- Reliability-aware distillation logic
- Ablation support for core R2D2-FL components
- Local APTOS preprocessing and loading support
- Clean GitHub repository prepared for evaluation and further development

The project demonstrates how reliability-aware and distillation-based techniques can be used to improve robustness in noisy federated learning.

The framework is suitable for further experimentation, result collection, ablation analysis, and final academic reporting.
