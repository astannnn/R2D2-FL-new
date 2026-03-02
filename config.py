import torch


# =====================================================
# Base Config (только реально общие параметры)
# =====================================================

class BaseConfig:

    # Reproducibility
    SEED = 1

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Dirichlet partition
    DIRICHLET_ALPHA = 0.3

    # ================= R2D2 =================
    USE_R2D2 = False
    TEMPERATURE = 2.0
    BETA = 0.1
    LAMBDA = 0.7
    CONF_THRESHOLD = 0.6

    # ================= Reliability =================
    USE_RELIABILITY = False
    USE_CLASS_RELIABILITY = False

    # ================= Ablation switches =================
    USE_SOFT_CORRECTION = False
    USE_LOCAL_KD = False

    # ================= FedProx =================
    USE_FEDPROX = False
    MU = 0.01


# =====================================================
# CIFAR-10
# =====================================================

class CIFARConfig(BaseConfig):

    DATASET = "cifar10"

    NUM_CLASSES = 10
    IN_CHANNELS = 3

    # Federated setup
    NUM_CLIENTS = 8
    CLIENT_FRACTION = 0.75
    LOCAL_EPOCHS = 2
    ROUNDS = 30
    BATCH_SIZE = 128
    LR = 0.01

    # Noise
    NOISE_CLIENT_RATIO = 0.5
    NOISE_RATE = 0.4
    NOISE_TYPE = "symmetric"

    # Proxy
    PROXY_SIZE = 400


# =====================================================
# EMNIST
# =====================================================

class EMNISTConfig(BaseConfig):

    DATASET = "emnist"
    EMNIST_SPLIT = "digits"

    NUM_CLASSES = 10
    IN_CHANNELS = 1

    # Federated setup
    NUM_CLIENTS = 8
    CLIENT_FRACTION = 0.75
    LOCAL_EPOCHS = 2
    ROUNDS = 30
    BATCH_SIZE = 64
    LR = 0.005

    # Noise
    NOISE_CLIENT_RATIO = 0.5
    NOISE_RATE = 0.4
    NOISE_TYPE = "symmetric"

    # Proxy
    PROXY_SIZE = 5000


# =====================================================
# APTOS (Medical)
# =====================================================

class APTOSConfig(BaseConfig):

    DATASET = "aptos"

    NUM_CLASSES = 5
    IN_CHANNELS = 3

    # Federated setup
    NUM_CLIENTS = 15
    CLIENT_FRACTION = 0.5
    LOCAL_EPOCHS = 1
    ROUNDS = 15
    BATCH_SIZE = 8
    LR = 0.0001

    # Noise
    NOISE_CLIENT_RATIO = 0.5
    NOISE_RATE = 0.0
    NOISE_TYPE = "symmetric"

    # Proxy
    PROXY_SIZE = 400
