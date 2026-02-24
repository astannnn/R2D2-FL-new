# Technical Report – R2D2-FL Implementation

## 1. Introduction

This project focuses on the implementation and experimental validation of R2D2-FL (Reliability-Weighted Robust Distillation for Federated Learning). The objective is to evaluate whether reliability-aware aggregation and proxy-based distillation improve robustness under non-IID and noisy client settings.

The work includes:
- Implementation of a FedAvg baseline
- Integration of heterogeneous noise injection
- Implementation of R2D2-FL components
- Experimental comparison under controlled settings

---

## 2. Baseline Implementation (FedAvg)

As a reference point, I implemented a standard Federated Averaging (FedAvg) algorithm.

The setup includes:
- K clients
- Random client sampling per round
- Local training for E epochs
- Weighted aggregation based on local dataset size

This baseline serves as a control to evaluate whether R2D2-FL introduces measurable improvements under noisy conditions.

---

## 3. Noise Injection Strategy

To simulate realistic federated learning conditions, the following noise settings are implemented:

- Symmetric noise
- Asymmetric noise
- Heterogeneous client-level noise

Noise is injected at the client level after Dirichlet-based data partitioning (α = configurable).

This allows simulation of:
- Uniform corruption
- Class-dependent corruption
- Client-level corruption variability

---

## 4. R2D2-FL Architecture

The R2D2-FL implementation consists of:

Client-side:
- Confidence-based sample selection
- Soft label correction
- Local knowledge distillation

Server-side:
- Client-level reliability estimation
- Class-level reliability weighting
- Reliability-weighted ensemble teacher construction
- Proxy-based global distillation

Each communication round follows:

1. Client sampling
2. Local training
3. Model upload
4. Reliability estimation on proxy data
5. Ensemble teacher construction
6. Global distillation
7. Model broadcast

---

## 5. Reliability Estimation

Client reliability is computed using a proxy dataset.

For each client:
- Predictions on proxy data are evaluated
- Agreement with majority vote is measured
- Reliability scores r_k are computed
- Class-level reliability r_{k,c} is also derived

These scores are used to weight client contributions during ensemble teacher construction.

---

## 6. Proxy-Based Distillation

A proxy dataset is maintained at the server.

Using reliability-weighted logits aggregation, a teacher distribution is constructed:

- Client logits are weighted
- Softmax with temperature τ is applied
- KL divergence loss is used to update the global model

This step is intended to:
- Reduce the impact of unreliable clients
- Improve robustness under noisy settings

---

## 7. Experimental Setup

Current experiments are conducted on:

Dataset:
- CIFAR-10

Partitioning:
- Dirichlet α = (configurable)

Noise settings:
- No noise
- 20% symmetric noise
- 40% symmetric noise

Metrics:
- Global accuracy
- Worst-client accuracy
- Convergence behavior

---

## 8. Current Observations

Preliminary experiments show:

- FedAvg behaves as expected under no noise
- Under 40% symmetric noise, FedAvg performance degrades significantly
- R2D2-FL does not yet consistently outperform FedAvg under current hyperparameter settings

Further investigation is ongoing to determine whether:
- Reliability weighting is too weak or too strong
- Proxy distillation is over-regularizing the global model
- Temperature and distillation weight require tuning

---

## 9. Ongoing Issues and Hypotheses

Current hypotheses include:

1. Reliability scores may not be sufficiently discriminative.
2. Proxy dataset size may be too small.
3. Distillation temperature may require adjustment.
4. Local KD weight β may be too high.

Next steps include:

- Controlled hyperparameter sweeps
- Ablation study (removing reliability, removing soft correction)
- Multiple-seed averaging for statistical stability
