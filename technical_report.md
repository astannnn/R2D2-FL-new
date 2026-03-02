# Technical Report  
## R2D2-FL: Reliability-Weighted Robust Distillation for Federated Learning

---

## 1. Introduction

This report presents the final implementation and experimental validation of **R2D2-FL (Reliability-Weighted Robust Distillation for Federated Learning)** under heterogeneous and noisy federated environments.

The objective of this project is to evaluate whether reliability-aware aggregation combined with proxy-based knowledge distillation improves robustness compared to standard federated optimization methods in realistic non-IID settings.

The project includes:

- Full implementation of **FedAvg** baseline  
- Full implementation of **FedProx** baseline  
- Modular implementation of **R2D2-FL**  
- Dirichlet-based non-IID simulation (α = 0.3)  
- Multiple label noise regimes  
- Experiments on **CIFAR-10, EMNIST (Digits), and APTOS 2019**  
- Structured ablation study  
- Reproducible logging and configuration control  

All experiments were conducted using fixed random seeds and deterministic data partitioning.

---

## 2. Configuration Design

The implementation follows a **dataset-specific configuration architecture** to ensure modularity and reproducibility.

A shared `BaseConfig` class defines global training structure and common hyperparameters.

Three specialized configurations extend it:

- `CIFARConfig`
- `EMNISTConfig`
- `APTOSConfig`

Each configuration specifies:

- Number of classes  
- Input dimensions  
- Model architecture  
- Learning rate  
- Number of communication rounds  
- Local epochs  
- Noise settings  
- Proxy dataset size  

The training pipeline is fully dataset-agnostic:

```python
config = CIFARConfig()
main(config)
```

The `main(config)` function dynamically adapts to the selected dataset without modifying the training logic.

This design ensures:

- Clean separation of experiment settings  
- Consistent baseline comparison  
- Easy hyperparameter control  
- Extensibility to new datasets  

---

## 3. Baseline Methods

### 3.1 FedAvg

FedAvg is implemented as the standard federated parameter averaging algorithm with:

- Random client sampling  
- Local SGD training  
- Weighted aggregation based on local dataset size  
- Shared global model architecture  

FedAvg serves as the primary reference baseline.

---

### 3.2 FedProx

FedProx extends FedAvg by adding a proximal regularization term to the local objective:

\[
\mathcal{L}_{prox} = \frac{\mu}{2} \| w_k - w_t \|^2
\]

This term reduces client drift under heterogeneous data distributions.

FedProx is used to evaluate whether proximal regularization alone improves robustness under noise.

---

## 4. Data Partitioning and Noise Modeling

### 4.1 Non-IID Partitioning

- Dirichlet-based partitioning with α = 0.3  
- Configurable number of clients  
- Deterministic partition generation per seed  

This creates heterogeneous class distributions across clients.

---

### 4.2 Noise Regimes

The following corruption scenarios were implemented:

- **0% noise (clean setting)**
- **20% symmetric noise**
- **40% symmetric noise**
- **Asymmetric noise**
- **Heterogeneous client-level corruption**

Noise is injected **after client partitioning**, ensuring realistic local corruption patterns.

---

## 5. R2D2-FL Framework

R2D2-FL replaces direct parameter averaging with reliability-weighted proxy-based distillation.

---

## 5.1 Client-Side Components

Each client performs local training with:

### 1. Confidence-Based Sample Selection

Samples with prediction confidence above threshold γ are treated as reliable.

### 2. Soft Label Correction

For low-confidence samples:

\[
\tilde{y}_i = \lambda y_i + (1-\lambda) p_{global}(x_i)
\]

### 3. Local Knowledge Distillation

\[
\mathcal{L}_{locKD} = KL(p_{global} \| p_{local})
\]

Final local objective:

\[
\mathcal{L}_k = \mathcal{L}_{sup} + \beta \mathcal{L}_{locKD}
\]

---

## 5.2 Server-Side Components

After receiving client models, the server performs:

### 1. Reliability Estimation

Using a proxy dataset:

- Compute predictions from each client  
- Measure agreement with ensemble majority  
- Derive client-level reliability \( r_k \)  
- Compute class-level reliability \( r_{k,c} \)  

---

### 2. Reliability-Weighted Ensemble Teacher

Client logits are weighted according to:

- Client reliability  
- Class reliability  
- Prediction entropy  

This reduces influence of corrupted clients.

---

### 3. Proxy-Based Global Distillation

Instead of averaging parameters, the global model is updated by minimizing:

\[
\mathcal{L}_{KD} = KL(p_{teacher} \| p_{global})
\]

Key hyperparameters:

- Temperature τ = 2.0  
- Distillation weight β = 0.1  
- Proxy dataset size = 400 samples  

---

## 6. Communication Round Procedure

Each federated round follows:

1. Sample clients  
2. Broadcast global model  
3. Perform local training  
4. Collect local models  
5. Estimate reliability on proxy data  
6. Construct reliability-weighted ensemble  
7. Distill global model on proxy dataset  
8. Broadcast updated global model  

Global updates are driven by **distillation**, not direct parameter averaging.

---

## 7. Experimental Setup

### Datasets

- **CIFAR-10**  
- **EMNIST (Digits split)**  
- **APTOS 2019 (Retinal classification)**  

Each dataset uses its own configuration class.

### Metrics

- Global test accuracy  
- Worst-client accuracy  
- Macro-F1 (APTOS)  
- Convergence across rounds  

Each experiment:

- Repeated across multiple seeds  
- Logged per round  
- Averaged for comparison  

---

## 8. Experimental Findings

### CIFAR-10

- R2D2-FL improves worst-client robustness under symmetric noise.
- Gains are more visible at 40% corruption.
- Clean performance is comparable to FedAvg.

### EMNIST

- Improved stability in moderate noise.
- Performance becomes configuration-sensitive under extreme corruption.

### APTOS

- Results are dataset-dependent.
- FedAvg sometimes achieves higher global accuracy.
- R2D2-FL provides improved robustness consistency across clients.

---

## 9. Ablation Study

Conducted on CIFAR-10 under 40% symmetric noise.

Components removed individually:

- Reliability weighting  
- Class-level reliability  
- Soft label correction  
- Local distillation  

Observations:

- Removing local KD reduces worst-client robustness.
- Removing soft label correction increases sensitivity to label noise.
- Removing reliability weighting decreases stability under heterogeneous corruption.

Robustness emerges from the interaction of all components.

---

## 10. Reproducibility and Modularity

The framework provides:

- Modular architecture  
- Dataset-agnostic pipeline  
- Configuration-based experiment control  
- Deterministic data partitioning  
- Structured logging  
- Google Colab validation notebooks  

All experiments are reproducible from configuration files.

---

## 11. Final Conclusions

This project delivers a complete and modular implementation of **R2D2-FL**.

Key conclusions:

- Reliability-weighted distillation improves robustness under symmetric corruption.
- Performance gains are dataset-dependent.
- Proxy-based aggregation stabilizes learning under heterogeneous client corruption.
- Careful hyperparameter tuning is required for highly imbalanced medical datasets.

The framework establishes a structured robustness benchmark for noisy federated learning and provides a foundation for further research on reliability-aware aggregation.
