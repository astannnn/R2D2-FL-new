import random
import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

from core.models import SimpleCNN
from core.partition import dirichlet_partition
from core.client import Client
from core.server import Server


# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_single_baseline(config):
    active_methods = [
        getattr(config, "USE_FEDPROX", False),
        getattr(config, "USE_FEDDF", False),
        getattr(config, "USE_R2D2", False),
        getattr(config, "USE_SELECTIVE_FD", False),
        getattr(config, "USE_FEDNORO", False),
    ]

    if sum(active_methods) > 1:
        raise ValueError("Only one baseline method can be active at a time.")


# =========================
# Noise
# =========================

def inject_symmetric_noise(dataset, indices, noise_rate, num_classes):
    targets = np.array(dataset.targets)

    n_noisy = int(len(indices) * noise_rate)
    noisy_indices = np.random.choice(indices, n_noisy, replace=False)

    for idx in noisy_indices:
        original = targets[idx]
        new_label = np.random.choice([c for c in range(num_classes) if c != original])
        targets[idx] = new_label

    dataset.targets = targets.tolist()


def inject_asymmetric_noise(dataset, indices, noise_rate, dataset_name):
    targets = np.array(dataset.targets)

    if dataset_name == "cifar10":
        mapping = {8: 0, 9: 1, 3: 5, 4: 7}

    elif dataset_name == "emnist":
        # EMNIST digits: visually similar confusions
        mapping = {
            1: 7,
            7: 1,
            3: 8,
            8: 3,
            5: 6,
            6: 5,
        }

    elif dataset_name == "aptos":
        # APTOS severity is ordinal, so asymmetric noise should mostly
        # confuse neighboring grades rather than random distant classes.
        #
        # 0 -> 1
        # 1 -> 2
        # 2 -> 3
        # 3 -> 4
        # 4 -> 3
        mapping = {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 3,
        }

    else:
        raise ValueError(f"Asymmetric noise not supported for dataset: {dataset_name}")

    candidates = [idx for idx in indices if targets[idx] in mapping]
    if len(candidates) == 0:
        return

    n_noisy = int(len(indices) * noise_rate)
    n_noisy = min(n_noisy, len(candidates))

    noisy_indices = np.random.choice(candidates, n_noisy, replace=False)
    for idx in noisy_indices:
        targets[idx] = mapping[targets[idx]]

    dataset.targets = targets.tolist()


def apply_noise(dataset, indices, config, client_id):

    if config.NOISE_TYPE == "asymmetric" and config.DATASET not in ["cifar10", "emnist", "aptos"]:
        raise ValueError("Asymmetric noise only supported for CIFAR-10, EMNIST and APTOS")

    if config.NOISE_TYPE == "symmetric":
        inject_symmetric_noise(dataset, indices, config.NOISE_RATE, config.NUM_CLASSES)

    elif config.NOISE_TYPE == "asymmetric":
        inject_asymmetric_noise(dataset, indices, config.NOISE_RATE, config.DATASET)

    elif config.NOISE_TYPE == "heterogeneous":
        if config.NUM_CLIENTS == 1:
            rate = config.NOISE_RATE
        else:
            rate = (client_id / (config.NUM_CLIENTS - 1)) * config.NOISE_RATE
        inject_symmetric_noise(dataset, indices, rate, config.NUM_CLASSES)

    else:
        raise ValueError(f"Unknown NOISE_TYPE: {config.NOISE_TYPE}")


# =========================
# Data
# =========================

def load_data(config):

    if config.DATASET == "emnist":

        def emnist_fix(x):
            x = torch.rot90(x, 1, [1, 2])
            x = torch.flip(x, [2])
            return x

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(emnist_fix),
            transforms.Normalize((0.5,), (0.5,))
        ])

        full_train = datasets.EMNIST(
            root="./data",
            split=config.EMNIST_SPLIT,
            train=True,
            download=True,
            transform=transform
        )

        np.random.seed(config.SEED)
        subset_size = min(80000, len(full_train))
        indices = np.random.choice(len(full_train), subset_size, replace=False)

        train_dataset = Subset(full_train, indices)

        test_dataset = datasets.EMNIST(
            root="./data",
            split=config.EMNIST_SPLIT,
            train=False,
            download=True,
            transform=transform
        )

        proxy_base = full_train

    elif config.DATASET == "aptos":

        from data.aptos_loader import load_aptos_raw
        train_dataset, test_dataset = load_aptos_raw(config)
        proxy_base = train_dataset

    elif config.DATASET == "cifar10":

        transform = transforms.ToTensor()

        train_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

        proxy_base = train_dataset

    else:
        raise ValueError(f"Unknown dataset: {config.DATASET}")

    proxy_dataset = None
    if getattr(config, "PROXY_SIZE", 0) > 0:
        proxy_size = min(config.PROXY_SIZE, len(proxy_base))
        proxy_indices = np.random.choice(len(proxy_base), proxy_size, replace=False)
        proxy_dataset = Subset(proxy_base, proxy_indices)

    return train_dataset, test_dataset, proxy_dataset


# =========================
# Clients
# =========================

def create_clients(train_dataset, config, model_factory):

    if isinstance(train_dataset, Subset):
        base_ds = train_dataset.dataset
        base_idxs = np.array(train_dataset.indices)
        targets = np.array(base_ds.targets)[base_idxs]
    else:
        base_ds = train_dataset
        base_idxs = None
        targets = np.array(getattr(train_dataset, "targets", getattr(train_dataset, "labels", None)))

    labels = torch.tensor(targets)

    client_indices = dirichlet_partition(
        labels.numpy(),
        config.NUM_CLIENTS,
        config.DIRICHLET_ALPHA
    )

    clients = []
    num_noisy_clients = int(config.NOISE_CLIENT_RATIO * config.NUM_CLIENTS)

    print(f"Injecting noise to {num_noisy_clients}/{config.NUM_CLIENTS} clients")
    print(f"Noise type: {config.NOISE_TYPE}")
    print(f"Noise rate: {config.NOISE_RATE}")

    for i in range(config.NUM_CLIENTS):

        idx_local = np.array(client_indices[i], dtype=int)

        if base_idxs is not None:
            idx_base = base_idxs[idx_local]
        else:
            idx_base = idx_local

        if i < num_noisy_clients and config.NOISE_RATE > 0:
            apply_noise(base_ds, idx_base, config, i)

        subset_ds = Subset(base_ds, idx_base.tolist())
        if len(subset_ds) == 0:
            continue

        loader = DataLoader(
            subset_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )

        model = model_factory()
        clients.append(Client(model, loader, config))

    return clients


# =========================
# Evaluation
# =========================

def evaluate_global(model, test_dataset, config):

    loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    acc = correct / total if total > 0 else 0.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return acc, macro_f1


def evaluate_per_client(server_model, clients, config):

    server_model.eval()
    client_accuracies = []

    with torch.no_grad():
        for client in clients:
            correct = 0
            total = 0
            for x, y in client.train_loader:
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)

                outputs = server_model(x)
                predicted = torch.argmax(outputs, dim=1)

                total += y.size(0)
                correct += (predicted == y).sum().item()

            acc = correct / total if total > 0 else 0.0
            client_accuracies.append(acc)

    return min(client_accuracies) if client_accuracies else 0.0


# =========================
# Selective-FD Round
# =========================

def selective_fd_step(selected_clients, server, proxy_dataset, config):
    if proxy_dataset is None or len(selected_clients) == 0:
        return

    client_probs_list = []
    client_masks_list = []

    for client in selected_clients:
        probs, mask = client.get_proxy_predictions(proxy_dataset)
        client_probs_list.append(probs)
        client_masks_list.append(mask)

    teacher_probs, valid_mask = server.build_selective_teacher(
        client_probs_list,
        client_masks_list,
        config
    )

    if teacher_probs is None or valid_mask is None:
        return

    for client in selected_clients:
        client.distill_on_proxy(proxy_dataset, teacher_probs, valid_mask)


# =========================
# Main
# =========================

def main(config=None):

    validate_single_baseline(config)

    print("DATASET =", config.DATASET)
    set_seed(config.SEED)

    def make_model():

        if config.DATASET == "aptos":
            import torchvision.models as models
            import torch.nn as nn

            m = models.resnet18(weights="IMAGENET1K_V1")
            m.fc = nn.Linear(m.fc.in_features, config.NUM_CLASSES)
            return m.to(config.DEVICE)

        return SimpleCNN(
            num_classes=config.NUM_CLASSES,
            in_channels=config.IN_CHANNELS
        ).to(config.DEVICE)

    print("=== CONFIG ===")
    print(f"DEVICE={config.DEVICE}")
    print(f"NUM_CLIENTS={config.NUM_CLIENTS}")
    print(f"CLIENT_FRACTION={config.CLIENT_FRACTION}")
    print(f"LOCAL_EPOCHS={config.LOCAL_EPOCHS}")
    print(f"ROUNDS={config.ROUNDS}")
    print(f"DIRICHLET_ALPHA={config.DIRICHLET_ALPHA}")
    print(f"NOISE_RATE={config.NOISE_RATE}")
    print(f"NOISE_TYPE={config.NOISE_TYPE}")
    print(f"USE_FEDPROX={config.USE_FEDPROX}")
    print(f"USE_FEDDF={config.USE_FEDDF}")
    print(f"USE_R2D2={config.USE_R2D2}")
    print(f"USE_SELECTIVE_FD={getattr(config, 'USE_SELECTIVE_FD', False)}")
    print(f"USE_FEDNORO={getattr(config, 'USE_FEDNORO', False)}")
    print("==============")

    train_dataset, test_dataset, proxy_dataset = load_data(config)
    clients = create_clients(train_dataset, config, make_model)

    print(f"Active clients created: {len(clients)}")

    global_model = make_model()
    server = Server(global_model)

    for r in range(config.ROUNDS):

        m = max(1, int(config.CLIENT_FRACTION * len(clients)))
        selected_clients = random.sample(clients, m)

        client_weights = []
        client_sizes = []

        for client in selected_clients:
            client.model.load_state_dict(server.global_model.state_dict())
            w = client.local_train(global_model=server.global_model, round_idx=r)

            client_weights.append(w)
            client_sizes.append(len(client.train_loader.dataset))

        server.aggregate(client_weights, client_sizes)

        # -------------------------
        # Selective-FD path
        # -------------------------
        if getattr(config, "USE_SELECTIVE_FD", False) and proxy_dataset is not None:
            selective_fd_step(selected_clients, server, proxy_dataset, config)

        # -------------------------
        # Existing FedDF / R2D2 path
        # -------------------------
        elif (config.USE_FEDDF or config.USE_R2D2) and proxy_dataset is not None:
            server.distill(
                [c.model for c in selected_clients],
                proxy_dataset,
                config
            )

        global_acc, macro_f1_val = evaluate_global(server.global_model, test_dataset, config)
        worst_acc = evaluate_per_client(server.global_model, clients, config)

        print(
            f"Round {r+1:03d} | "
            f"Global={global_acc:.4f} | "
            f"MacroF1={macro_f1_val:.4f} | "
            f"Worst={worst_acc:.4f}"
        )


from config import CIFARConfig  # CIFARConfig / EMNISTConfig / APTOSConfig

if __name__ == "__main__":
    config = CIFARConfig()

    # examples:
    # config.USE_FEDPROX = True
    # config.USE_FEDDF = True
    # config.USE_R2D2 = True
    # config.USE_SELECTIVE_FD = True
    # config.USE_FEDNORO = True

    main(config)
