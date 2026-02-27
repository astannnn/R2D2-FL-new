import torch


class Config:

    # =====================================================
    # DATASET
    # =====================================================
    DATASET = "emnist"      # "cifar10" / "emnist"
    NUM_CLASSES = 62        # 10 for CIFAR, 62 for EMNIST
    IN_CHANNELS = 1         # 3 for CIFAR, 1 for EMNIST

    # =====================================================
    # FL setup
    # =====================================================
    NUM_CLIENTS = 1
    CLIENT_FRACTION = 1.0
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 128
    ROUNDS = 5
    LR = 0.001
    DIRICHLET_ALPHA = 0.5
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
    USE_RELIABILITY = True
    USE_CLASS_RELIABILITY = True

    # =====================================================
    # Ablation switches
    # =====================================================
    USE_SOFT_CORRECTION = True
    USE_LOCAL_KD = True

    # =====================================================
    # FedProx (optional baseline)
    # =====================================================
    USE_FEDPROX = False
    MU = 0.01
