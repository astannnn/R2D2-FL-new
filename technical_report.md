Technical Report
R2D2-FL Implementation
1. Introduction

This project presents the implementation and experimental evaluation of R2D2-FL (Reliability-Weighted Robust Distillation for Federated Learning) under non-IID and noisy federated environments.

The objective is to evaluate whether reliability-aware aggregation and proxy-based knowledge distillation improve robustness compared to standard federated optimization methods.

The project includes:

Implementation of FedAvg baseline

Implementation of FedProx baseline

Integration of multiple noise regimes

Full implementation of R2D2-FL

Experimental evaluation on CIFAR-10, EMNIST, and APTOS

Structured ablation study

All experiments were conducted under controlled and reproducible configurations.

2. Baseline Methods
2.1 FedAvg

Standard federated parameter averaging with:

Random client sampling

Local SGD training

Weighted aggregation based on dataset size

2.2 FedProx

An extension of FedAvg introducing a proximal regularization term to reduce client drift under heterogeneous data distributions.

Both baselines serve as reference methods for robustness comparison.

3. Noise Modeling

To simulate realistic federated conditions, multiple noise regimes were implemented:

No noise (0%)

20% symmetric noise

40% symmetric noise

40% asymmetric noise

Heterogeneous client-level corruption

Noise is injected after Dirichlet-based data partitioning (α = 0.3).

This enables simulation of:

Uniform corruption

Structured label corruption

Client-level corruption variability

4. R2D2-FL Architecture

R2D2-FL extends standard federated learning with reliability-aware distillation.

4.1 Client-Side Components

Confidence-based sample selection

Soft label correction

Local knowledge distillation

4.2 Server-Side Components

Client-level reliability estimation

Class-level reliability weighting

Reliability-weighted ensemble teacher construction

Proxy-based global distillation

4.3 Communication Round Procedure

Each round consists of:

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

These scores are used to weight logits during ensemble teacher construction.

6. Proxy-Based Distillation

A proxy dataset (size = 400 samples) is maintained at the server.

The global model is updated by minimizing:

KL divergence between global predictions

Reliability-weighted ensemble teacher distribution

Temperature scaling (τ = 2.0) and distillation weight (β = 0.1) are applied.

This mechanism reduces the influence of unreliable clients.

7. Experimental Evaluation

Experiments were conducted on:

CIFAR-10

EMNIST

APTOS 2019 (Medical Dataset)

Metrics

Global test accuracy

Worst-client accuracy

Macro-F1 (APTOS)

Summary of Findings

On CIFAR-10, R2D2-FL consistently improves worst-client robustness under symmetric corruption.

On EMNIST, R2D2-FL improves clean performance but shows mixed behavior under extreme symmetric noise.

On APTOS, FedAvg outperforms R2D2-FL, indicating dataset-dependent robustness behavior.

8. Ablation Study

Ablation experiments were conducted on CIFAR-10 under 40% symmetric noise.

Key Observations

Removing local KD slightly increases global accuracy but reduces worst-client stability.

Removing soft label correction significantly increases performance, suggesting sensitivity under severe corruption.

Removing reliability weighting produces moderate performance variations.

This confirms that individual components contribute differently depending on noise characteristics.

9. Final Conclusions

The implementation confirms that reliability-weighted distillation improves robustness in balanced multi-class datasets under symmetric corruption.

However, performance gains are dataset-dependent and require further tuning in highly imbalanced medical scenarios.

This project provides a complete and reproducible implementation of R2D2-FL and establishes a structured robustness benchmark in federated learning.
