import torch


class Config:

    # ===== FL setup =====
    NUM_CLIENTS = 4
    CLIENT_FRACTION = 0.75
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 128
    ROUNDS = 30
    LR = 0.01
    DIRICHLET_ALPHA = 1.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Noise setup =====
    NOISE_CLIENT_RATIO = 0.5
    NOISE_RATE = 0.0
    NOISE_TYPE = "symmetric"   # symmetric / asymmetric / heterogeneous

    # ===== Proxy =====
    PROXY_SIZE = 150

    # ===== Reproducibility =====
    SEED = 3

    # ===== R2D2-FL parameters =====
    TEMPERATURE = 2.0
    BETA = 0.05
    LAMBDA = 0.7
    CONF_THRESHOLD = 0.6
