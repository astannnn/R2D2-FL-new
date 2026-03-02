Technical Report – R2D2-FL Implementation
1. Introduction

This project implements and experimentally evaluates R2D2-FL (Reliability-Weighted Robust Distillation for Federated Learning) under non-IID and noisy federated settings.

The objective is to assess whether reliability-aware aggregation and proxy-based knowledge distillation improve robustness compared to standard federated optimization methods.

The project includes:

Implementation of FedAvg baseline

Implementation of FedProx baseline

Integration of multiple noise regimes

Full implementation of R2D2-FL

Experimental evaluation on CIFAR-10, EMNIST, and APTOS

Structured ablation study

All experiments were conducted under controlled and reproducible configurations.

2. Baseline Methods

Two baselines were implemented:

FedAvg

Standard federated parameter averaging with:

Random client sampling

Local SGD training

Weighted aggregation by dataset size

FedProx

An extension of FedAvg with a proximal regularization term to reduce client drift under heterogeneous data distributions.

These baselines serve as reference methods for robustness comparison.

3. Noise Modeling

To simulate realistic federated learning conditions, multiple noise regimes were implemented:

No noise (0%)

20% symmetric noise

40% symmetric noise

40% asymmetric noise

Heterogeneous client-level corruption

Noise was injected after Dirichlet-based data partitioning (α = 0.3) to simulate strong non-IID behavior.

This design enables evaluation under:

Uniform corruption

Structured label corruption

Client-dependent noise variability

4. R2D2-FL Architecture

The implemented R2D2-FL framework extends standard federated learning with reliability-aware distillation.

Client-Side Components

Confidence-based sample selection

Soft label correction

Local knowledge distillation

Server-Side Components

Client-level reliability estimation

Class-level reliability weighting

Reliability-weighted ensemble teacher construction

Proxy-based global distillation

Each communication round consists of:

Client sampling

Local training

Model upload

Reliability estimation on proxy data

Ensemble teacher construction

Global distillation

Model broadcast

Unlike FedAvg, global updates are performed via proxy-based distillation rather than direct parameter averaging.

5. Reliability Estimation

Client reliability scores 
𝑟
𝑘
r
k
	​

 are computed using a server-side proxy dataset.

For each client:

Predictions on proxy samples are collected

Agreement with ensemble majority is measured

Client-level reliability is derived

Class-level reliability 
𝑟
𝑘
,
𝑐
r
k,c
	​

 is computed

These reliability scores are used to weight logits during ensemble teacher construction, reducing the influence of unreliable clients.

6. Proxy-Based Distillation

A proxy dataset (size = 400 samples) is maintained at the server.

The global model is updated by minimizing KL divergence between:

Global model predictions

Reliability-weighted ensemble teacher distribution

Temperature scaling (τ = 2.0) and distillation weight (β = 0.1) are applied.

This mechanism improves robustness by filtering noisy client contributions.

7. Experimental Evaluation

Experiments were conducted on:

CIFAR-10

EMNIST

APTOS 2019 (medical dataset)

Metrics:

Global test accuracy

Worst-client accuracy

Macro-F1 (APTOS)

Results show:

On CIFAR-10, R2D2-FL consistently improves worst-client robustness under symmetric corruption.

On EMNIST, R2D2-FL improves clean performance but shows mixed behavior under extreme symmetric noise.

On APTOS, FedAvg outperforms R2D2-FL, indicating dataset-dependent behavior under severe class imbalance.

8. Ablation Study

An ablation study was conducted on CIFAR-10 under 40% symmetric noise.

Findings:

Removing local KD slightly increases global accuracy but reduces worst-client stability.

Removing soft correction significantly increases performance, suggesting sensitivity under severe symmetric noise.

Removing reliability weighting produces moderate changes.

This demonstrates that different components contribute unequally depending on noise characteristics.

9. Final Conclusions

The implementation confirms that reliability-weighted distillation improves robustness in balanced multi-class datasets under symmetric corruption.

However, performance gains are dataset-dependent and may require further tuning in highly imbalanced medical scenarios.

The project provides a complete and reproducible implementation of R2D2-FL and establishes a structured benchmark for robustness evaluation in federated learning.
