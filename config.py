import torch


class Config:

    # =====================================================
    # DATASET
    # =====================================================
    DATASET = "aptos"
    EMNIST_SPLIT = "balanced"   # "balanced" (47 classes) или "byclass" (62)
    NUM_CLASSES = 5            # 47 для balanced
    IN_CHANNELS = 3             # EMNIST grayscale

    # =====================================================
    # FL setup
    # =====================================================
    NUM_CLIENTS = 15
    CLIENT_FRACTION = 0.5
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 8
    ROUNDS = 5
    LR = 0.0001
    DIRICHLET_ALPHA = 0.3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================================================
    # Noise setup
    # =====================================================
    NOISE_CLIENT_RATIO = 0.5
    NOISE_RATE = 0.0
    NOISE_TYPE = "symmetric"   # "symmetric" / "heterogeneous"

    # =====================================================
    # Proxy
    # =====================================================
    PROXY_SIZE = 400

    # =====================================================
    # Reproducibility
    # =====================================================
    SEED = 1

    # =====================================================
    # R2D2-FL parameters
    # =====================================================
    USE_R2D2 = False

    TEMPERATURE = 2.0
    BETA = 0.1
    LAMBDA = 0.7
    CONF_THRESHOLD = 0.6

    # =====================================================
    # Reliability
    # =====================================================
    USE_RELIABILITY = False
    USE_CLASS_RELIABILITY = False

    # =====================================================
    # Ablation switches
    # =====================================================
    USE_SOFT_CORRECTION = False
    USE_LOCAL_KD = False

    # =====================================================
    # FedProx (optional baseline)
    # =====================================================
    USE_FEDPROX = False
    MU = 0.01
