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

## 2. Baseline Methods

### 2.1 FedAvg

FedAvg is implemented as the standard federated parameter averaging algorithm with:

- Random client sampling  
- Local SGD training  
- Weighted aggregation based on local dataset size  
- Shared global model architecture  

FedAvg serves as the primary reference baseline for all experiments.

---

### 2.2 FedProx

FedProx extends FedAvg by adding a proximal regularization term to the local objective:

\[
\mathcal{L}_{prox} = \frac{\mu}{2} \| w_k - w_t \|^2
\]

This term reduces client drift under heterogeneous data distributions.

FedProx is used to evaluate whether proximal regularization alone is sufficient to improve robustness under noisy and non-IID conditions.

---

## 3. Data Partitioning and Noise Modeling

To simulate realistic federated learning environments, the following protocol was implemented:

### 3.1 Non-IID Partitioning

- Dirichlet-based partitioning with α = 0.3  
- Configurable number of clients  
- Deterministic partition generation per seed  

This creates heterogeneous class distributions across clients.

---

### 3.2 Noise Regimes

The following corruption scenarios were implemented:

- **0% noise (clean setting)**
- **20% symmetric noise**
- **40% symmetric noise**
- **Asymmetric noise**
- **Heterogeneous client-level corruption**

Noise is injected **after client partitioning**, ensuring realistic local corruption patterns.

This enables simulation of:

- Uniform random corruption  
- Structured label flips  
- Client-specific reliability variability  

---

## 4. R2D2-FL Framework

R2D2-FL extends standard federated learning by replacing direct parameter aggregation with reliability-weighted distillation.

---

## 4.1 Client-Side Components

Each client performs local training with:

### 1. Confidence-Based Sample Selection

Samples with prediction confidence above threshold γ are treated as reliable.

### 2. Soft Label Correction

For low-confidence samples:

\[
\tilde{y}_i = \lambda y_i + (1-\lambda) p_{global}(x_i)
\]

This reduces the impact of noisy labels.

### 3. Local Knowledge Distillation

A KL divergence term aligns local predictions with the global teacher:

\[
\mathcal{L}_{locKD} = KL(p_{global} \| p_{local})
\]

Final local objective:

\[
\mathcal{L}_k = \mathcal{L}_{sup} + \beta \mathcal{L}_{locKD}
\]

---

## 4.2 Server-Side Components

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

## 5. Communication Round Procedure

Each federated round follows:

1. Sample clients  
2. Broadcast global model  
3. Perform local training  
4. Collect local models  
5. Estimate reliability on proxy data  
6. Construct reliability-weighted ensemble  
7. Distill global model on proxy dataset  
8. Broadcast updated global model  

Unlike FedAvg, global updates are driven by **proxy-based distillation rather than direct parameter averaging**.

---

## 6. Experimental Setup

### Datasets

- **CIFAR-10** (natural image classification)  
- **EMNIST (Digits split)**  
- **APTOS 2019 (Retinal classification)**  

### Metrics

- Global test accuracy  
- Worst-client accuracy  
- Macro-F1 (APTOS)  
- Convergence behavior across rounds  

Each experiment:

- Repeated across multiple seeds  
- Logged per round  
- Averaged for final comparison  

---

## 7. Experimental Findings

### CIFAR-10

- R2D2-FL improves worst-client robustness under symmetric noise.
- Gains are more visible at 40% corruption.
- Under clean settings, performance is comparable to FedAvg.

### EMNIST

- R2D2-FL improves stability in moderate noise.
- Under extreme symmetric noise, performance becomes configuration-sensitive.
- Reliability weighting helps reduce degradation for minority classes.

### APTOS (Medical Dataset)

- Performance is dataset-dependent.
- FedAvg occasionally achieves higher global accuracy.
- R2D2-FL provides more stable behavior across clients but requires careful tuning.

---

## 8. Ablation Study

Ablation experiments were conducted on CIFAR-10 under 40% symmetric noise.

Components removed individually:

- Reliability weighting  
- Class-level reliability  
- Soft label correction  
- Local distillation  

### Observations

- Removing local KD can increase global accuracy but reduces worst-client robustness.
- Removing soft label correction increases sensitivity to noisy labels.
- Removing reliability weighting reduces stability under heterogeneous corruption.

This confirms that robustness emerges from the interaction of all components.

---

## 9. Reproducibility and Modularity

The implementation provides:

- Fully modular architecture  
- Dataset-agnostic pipeline  
- Configurable hyperparameters  
- Deterministic data partitioning  
- Round-wise logging  
- Google Colab validation notebooks  

All results are reproducible from configuration files.

---

## 10. Final Conclusions

This project delivers a complete, modular, and reproducible implementation of **R2D2-FL**.

Key conclusions:

- Reliability-weighted distillation improves robustness under symmetric corruption.
- Gains are dataset-dependent and sensitive to hyperparameter tuning.
- Proxy-based aggregation provides stability benefits under heterogeneous client corruption.
- Performance improvements are strongest in balanced multi-class datasets.

The framework establishes a structured robustness benchmark for noisy federated learning and serves as a foundation for further research in reliability-aware aggregation.
