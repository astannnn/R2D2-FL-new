# Technical Report  
## R2D2-FL: Reliability-Weighted Robust Distillation for Federated Learning

---

## 1. Introduction

This technical report presents the implementation and experimental validation of **R2D2-FL (Reliability-Weighted Robust Distillation for Federated Learning)** under heterogeneous and noisy federated learning environments.

The objective of this project is to evaluate whether reliability-aware client weighting combined with proxy-based knowledge distillation can improve robustness compared to standard federated optimization and federated distillation methods in realistic non-IID and noisy-label settings.

The project includes:

- Implementation and evaluation of **FedAvg**
- Implementation and evaluation of **FedProx**
- Implementation and evaluation of **FedDF**
- Implementation and evaluation of **Selective-FD**
- Implementation and evaluation of **FedNoRo**
- Modular implementation of **R2D2-FL**
- Dirichlet-based non-IID simulation with α = 0.3
- Multiple label noise regimes
- Experiments on **CIFAR-10, EMNIST, and APTOS 2019**
- Three-seed experimental evaluation
- Ablation study for R2D2-FL components
- Reproducible configuration control
- GitHub repository containing only source code, configuration files, documentation, and validation notebooks

Large datasets, virtual environments, checkpoints, logs, and generated outputs are intentionally excluded from the repository.

---

## 2. Development and Repository Context

Earlier parts of the project were executed through GitHub and Google Colab due to limited local computational resources. After moving to a Mac environment, the project was further developed and tested locally using Cursor/VS Code.

The final repository is organized so that the project can be cloned or pulled without downloading unnecessary large files.

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

The Jupyter notebooks are included for dataset validation, Google Colab testing, and quick experimental checks. The main implementation is contained in the Python source files.

The following files and directories are excluded from GitHub:

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

The project uses three datasets: CIFAR-10, EMNIST, and APTOS 2019.

### 3.1 CIFAR-10

CIFAR-10 is downloaded automatically through `torchvision`.

It is used as the main controlled benchmark for testing robustness under synthetic label noise and non-IID client partitioning.

### 3.2 EMNIST

EMNIST is downloaded automatically through `torchvision`.

In this project, the EMNIST Digits split is used as a 10-class grayscale image classification task.

### 3.3 APTOS 2019

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

APTOS preprocessing includes image resizing, normalization, and loading labels from the CSV file.

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

The project compares R2D2-FL against several federated learning and federated distillation baselines.

The evaluated methods are:

- FedAvg
- FedProx
- FedDF
- Selective-FD
- FedNoRo
- R2D2-FL

---

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

FedProx is included to evaluate whether proximal regularization improves robustness under noisy federated settings.

---

### 5.3 FedDF

FedDF is implemented as a federated distillation baseline.

Instead of relying only on parameter averaging, FedDF performs server-side distillation using predictions from client models on a proxy dataset.

FedDF is included because it is a direct comparison point for R2D2-FL, since both methods use proxy-based distillation at the server side.

---

### 5.4 Selective-FD

Selective-FD is included as a selective federated distillation baseline.

The method is based on selective knowledge sharing, where ambiguous or unreliable predictions are filtered before distillation.

Selective-FD is useful for evaluating whether selective distillation improves robustness under noisy and heterogeneous data distributions.

---

### 5.5 FedNoRo

FedNoRo is included as a noisy-label robust federated learning baseline.

It is designed to improve learning when local client datasets contain corrupted labels or suspicious local data.

FedNoRo is particularly relevant for comparison under symmetric, asymmetric, and heterogeneous label noise settings.

---

### 5.6 Method Switches

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

Noise is injected after client partitioning. This is important because it allows each client to first receive a non-IID local dataset and then receive label corruption according to the selected setting.

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

### 7.1.1 Confidence-Based Sample Selection

For each local sample, the client estimates prediction confidence.

High-confidence samples are treated as more reliable, while low-confidence samples are handled with soft label correction.

A simplified confidence rule can be written as:

\[
m_i = \mathbf{1}[p_{w^k}(y_i | x_i) > \gamma]
\]

where:

- \(m_i\) indicates whether sample \(i\) is high-confidence
- \(p_{w^k}(y_i | x_i)\) is the probability assigned to the given label
- \(\gamma\) is the confidence threshold

---

### 7.1.2 Soft Label Correction

For low-confidence samples, the original label is mixed with the global teacher prediction:

\[
\tilde{y}_i = \lambda y_i + (1 - \lambda) p_{global}(x_i)
\]

This reduces the effect of incorrect labels while still preserving information from the original annotation.

---

### 7.1.3 Local Knowledge Distillation

The local model is encouraged to stay close to the global teacher distribution:

\[
\mathcal{L}_{locKD} = KL(p_{global} \| p_{local})
\]

The final local objective is:

\[
\mathcal{L}_k = \mathcal{L}_{sup} + \beta \mathcal{L}_{locKD}
\]

This helps reduce overfitting to corrupted local data and limits client drift under non-IID distributions.

---

## 7.2 Server-Side Components

After local training, the server receives selected client models and performs reliability-aware aggregation through proxy-based distillation.

### 7.2.1 Reliability Estimation

Using the proxy dataset, the server evaluates each selected client model.

Reliability is estimated using:

- Agreement with ensemble predictions
- Client-level reliability
- Class-level reliability
- Prediction confidence or entropy

A simplified client-level reliability score can be represented as:

\[
r_k = \mathbb{E}_{x \sim D_p}
\mathbf{1}[\arg\max p_k(x) = \arg\max p_{maj}(x)]
\]

where:

- \(D_p\) is the proxy dataset
- \(p_k(x)\) is the prediction of client \(k\)
- \(p_{maj}(x)\) is the majority or consensus prediction

This allows the server to reduce the influence of unreliable or corrupted clients.

---

### 7.2.2 Reliability-Weighted Ensemble Teacher

The server constructs an ensemble teacher by weighting client predictions according to reliability.

The weighting considers:

- Overall client reliability
- Class-specific reliability
- Prediction entropy

A simplified reliability-aware weight can be written as:

\[
\alpha_k(x) =
\sigma(r_k) \cdot \sigma(r_{k,\hat{c}(x)}) \cdot (1 - H(p_k(x)))
\]

The ensemble teacher logits are computed as:

\[
z_T(x) = \sum_{k \in S_t} \alpha_k(x) z_k(x)
\]

The teacher distribution is obtained through temperature-scaled softmax:

\[
p_T(\cdot | x) = softmax(z_T(x) / \tau)
\]

---

### 7.2.3 Proxy-Based Global Distillation

The global model is updated by minimizing the KL divergence between the ensemble teacher and the global model:

\[
\mathcal{L}_{KD} =
\mathbb{E}_{x \sim D_p}
KL(p_T(\cdot | x) \| p_w(\cdot | x))
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
- EMNIST Digits
- APTOS 2019

CIFAR-10 and EMNIST are downloaded automatically through `torchvision`. APTOS 2019 is stored locally and loaded through custom dataset loading code.

---

### 9.2 Federated Configurations

The final experimental configurations are:

| Dataset | Clients | Client Fraction | Local Epochs | Rounds | Batch Size | Learning Rate | Proxy Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 8 | 0.75 | 2 | 30 | 128 | 0.01 | 1000 |
| EMNIST | 8 | 0.75 | 2 | 20 | 64 | 0.005 | 5000 |
| APTOS 2019 | 15 | 0.50 | 1 | 30 | 8 | 0.00005 | 200 |

All datasets use Dirichlet partitioning with:

```text
DIRICHLET_ALPHA = 0.3
```

---

### 9.3 Experimental Conditions

Each dataset was evaluated under the following noise settings:

```text
0% noise
20% symmetric noise
40% symmetric noise
40% asymmetric noise
40% heterogeneous noise
```

Each experiment was repeated using three random seeds:

```text
1, 2, 3
```

This seed-based evaluation was used to reduce randomness and provide more reliable comparisons.

---

### 9.4 Metrics

The framework reports:

- Global test accuracy
- Worst-client accuracy
- Macro-F1 score
- Round-wise training time
- Communication cost estimate
- Best-round performance
- Convergence behavior across communication rounds

For APTOS, Macro-F1 is especially important because the dataset is imbalanced.

---

## 10. Experimental Findings

### 10.1 CIFAR-10

CIFAR-10 is used as the main controlled benchmark.

The final results show that R2D2-FL achieves the strongest overall performance on CIFAR-10 across all evaluated noise regimes.

Observed behavior:

- R2D2-FL achieves the best average global accuracy, Macro-F1, and worst-client accuracy across all CIFAR-10 noise settings.
- The advantage is especially clear under 40% symmetric noise and 40% heterogeneous client-level noise.
- The heterogeneous client-level noise setting aligns well with the design of R2D2-FL because only a subset of clients is corrupted.
- Reliability-aware distillation reduces the influence of unreliable clients and improves global robustness.

---

### 10.2 EMNIST

EMNIST provides a lighter grayscale benchmark for testing the pipeline across multiple seeds and configurations.

Observed behavior:

- R2D2-FL achieves the best global accuracy and Macro-F1 in most EMNIST settings.
- R2D2-FL performs best under clean, symmetric, and heterogeneous noise settings.
- Under 40% asymmetric noise, FedNoRo achieves the best global accuracy and Macro-F1.
- R2D2-FL remains competitive under asymmetric noise and achieves strong worst-client performance.
- EMNIST is easier than CIFAR-10 and APTOS, so performance differences between methods are generally smaller.

---

### 10.3 APTOS 2019

APTOS is the most computationally expensive and unstable dataset in the project.

Observed behavior:

- Training is significantly slower than CIFAR-10 and EMNIST.
- The dataset is highly imbalanced.
- Macro-F1 is necessary for fair evaluation.
- Results fluctuate more strongly due to medical image complexity and class imbalance.
- No single method dominates across all APTOS settings.
- R2D2-FL achieves the best global accuracy under 40% heterogeneous client-level noise.
- FedProx, FedDF, and Selective-FD perform better in some APTOS settings, especially for Macro-F1 and worst-client accuracy.

APTOS therefore acts as a realistic stress test for noisy and heterogeneous federated learning.

---

## 11. Ablation Study

An ablation study was conducted to analyze the contribution of the main R2D2-FL components.

The ablation study was conducted on CIFAR-10 under the 40% symmetric label noise setting.

Ablation variants include:

- Full R2D2-FL
- Removing reliability weighting
- Removing class-level reliability
- Removing soft label correction
- Removing local knowledge distillation

Observed behavior:

- Removing reliability weighting decreased global accuracy and Macro-F1 compared with the full method.
- Removing class-level reliability produced similar global accuracy but slightly different worst-client behavior.
- Removing soft label correction and removing local knowledge distillation achieved slightly higher final-round values in this specific ablation setting.
- This indicates that the contribution of individual components is not always additive.
- The interaction between reliability weighting, soft correction, and local distillation is dataset-dependent and seed-dependent.

The ablation study shows that reliability weighting has the clearest positive contribution, while soft correction and local distillation may require additional hyperparameter tuning depending on the dataset and noise regime.

---

## 12. Reproducibility

The project supports reproducibility through:

- Fixed random seeds
- Deterministic client partitioning
- Configuration-based experiment setup
- Consistent logging
- Dataset-specific configuration classes
- Identical noise settings across compared methods
- Identical model architecture per dataset
- Clean GitHub repository without local datasets or virtual environment files

The implementation automatically selects the available device:

1. CUDA
2. Apple MPS
3. CPU

This allows the same codebase to run on different hardware environments.

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
- Jupyter notebooks are available for validation and Colab-based testing
- Experiments were run using the required three seeds
- Baselines were evaluated across all required noise settings
- Ablation was performed separately on CIFAR-10 under 40% symmetric noise
- Final internship report, README, notebooks, and technical documentation were updated

This repository state is appropriate because GitHub should contain the implementation and documentation, while large datasets should remain local or be downloaded through scripts.

---

## 14. Limitations

The main limitations of the current project are:

- APTOS experiments are computationally expensive.
- APTOS class imbalance makes global accuracy insufficient by itself.
- Some methods require careful hyperparameter tuning.
- The contribution of individual R2D2-FL components is not always additive.
- Results may differ depending on device type, batch size, and available compute resources.
- APTOS requires manual local dataset placement because it cannot be automatically downloaded from GitHub.
- The current implementation uses simulated federated learning rather than a real distributed deployment.

These limitations are discussed in the final internship report.

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

The results show that R2D2-FL performs especially well on CIFAR-10 and EMNIST, while APTOS 2019 remains more challenging due to class imbalance and medical image complexity.

The framework is suitable for final academic reporting and future extensions, including additional datasets, stronger privacy mechanisms, and more systematic hyperparameter tuning.
