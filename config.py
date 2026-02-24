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
    TEMPERATURE = 2.0
    BETA = 0.1
    LAMBDA = 0.7
    CONF_THRESHOLD = 0.6
    USE_R2D2 = False
