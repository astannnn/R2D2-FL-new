import torch


class Config:

    # ===== FL setup =====
    NUM_CLIENTS = 8
    CLIENT_FRACTION = 0.75
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 128
    ROUNDS = 30
    LR = 0.01
    DIRICHLET_ALPHA = 0.3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Noise setup =====
    NOISE_CLIENT_RATIO = 0.5
    NOISE_RATE = 0.4
    NOISE_TYPE = "symmetric"   # symmetric / asymmetric / heterogeneous

    # ===== Proxy =====
    PROXY_SIZE = 400

    # ===== Reproducibility =====
    SEED = 1

    # ===== R2D2-FL parameters =====
    USE_R2D2 = True

    TEMPERATURE = 2.0          # tau
    BETA = 0.1                 # local KD weight
    LAMBDA = 0.7               # soft label correction weight
    CONF_THRESHOLD = 0.6       # confidence threshold

    # ===== Ablation switches =====
    USE_RELIABILITY = True
    USE_CLASS_RELIABILITY = True
    USE_SOFT_CORRECTION = True
    USE_LOCAL_KD = True
