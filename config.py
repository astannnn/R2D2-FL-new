import torch


# =====================================================
# Base Config
# =====================================================

class BaseConfig:

    DISTILL_LR = 0.001

    # Reproducibility
    SEED = 1

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Dirichlet partition
    DIRICHLET_ALPHA = 0.3

    # ================= Method switches =================
    USE_R2D2 = False
    USE_FEDDF = False
    USE_FEDPROX = False
    USE_SELECTIVE_FD = False
    USE_FEDNORO = False

    # ================= R2D2 =================
    TEMPERATURE = 2.0
    BETA = 0.3
    LAMBDA = 0.7
    CONF_THRESHOLD = 0.7

    # ================= Reliability =================
    USE_RELIABILITY = False
    USE_CLASS_RELIABILITY = False

    # ================= Ablation switches =================
    USE_SOFT_CORRECTION = False
    USE_LOCAL_KD = False

    # ================= FedProx =================
    MU = 0.01

    # ================= Selective-FD =================
    SELECTIVE_TAU_CLIENT = 0.60
    SELECTIVE_TAU_SERVER = 0.80
    SELECTIVE_KD_WEIGHT = 1.0
    SELECTIVE_USE_SOFT = True
    PROXY_BATCH_SIZE = 128
    SELECTIVE_DISTILL_EPOCHS = 1

    # ================= FedNoRo =================
    FEDNORO_WARMUP_ROUNDS = 5
    FEDNORO_LABEL_CORRECTION_START = 8
    FEDNORO_CONF_THRESHOLD = 0.90
    FEDNORO_SOFT_WEIGHT = 0.7
    FEDNORO_USE_SOFT = True
    FEDNORO_KD_WEIGHT = 0.0
    FEDNORO_SUSPICIOUS_WEIGHT = 0.5


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
    NOISE_RATE = 0.0
    NOISE_TYPE = "symmetric"

    # Proxy
    PROXY_SIZE = 1000


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
    ROUNDS = 20
    BATCH_SIZE = 64
    LR = 0.005

    # Noise
    NOISE_CLIENT_RATIO = 0.5
    NOISE_RATE = 0.0
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
