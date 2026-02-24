import time
import csv
import random
import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from config import Config
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


# =========================
# Noise
# =========================

def inject_symmetric_noise(dataset, indices, noise_rate, num_classes=10):
    targets = np.array(dataset.targets)

    n_noisy = int(len(indices) * noise_rate)
    noisy_indices = np.random.choice(indices, n_noisy, replace=False)

    for idx in noisy_indices:
        original = targets[idx]
        new_label = np.random.choice(
            [c for c in range(num_classes) if c != original]
        )
        targets[idx] = new_label

    dataset.targets = targets.tolist()


def inject_asymmetric_noise(dataset, indices, noise_rate):
    targets = np.array(dataset.targets)

    mapping = {
        8: 0,  # ship -> airplane
        9: 1,  # truck -> automobile
        3: 5,  # cat -> dog
        4: 7,  # deer -> horse
    }

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

    if config.NOISE_TYPE == "symmetric":
        inject_symmetric_noise(dataset, indices, config.NOISE_RATE)

    elif config.NOISE_TYPE == "asymmetric":
        inject_asymmetric_noise(dataset, indices, config.NOISE_RATE)

    elif config.NOISE_TYPE == "heterogeneous":
        if config.NUM_CLIENTS == 1:
            rate = config.NOISE_RATE
        else:
            rate = (client_id / (config.NUM_CLIENTS - 1)) * config.NOISE_RATE

        inject_symmetric_noise(dataset, indices, rate)

    else:
        raise ValueError(f"Unknown NOISE_TYPE: {config.NOISE_TYPE}")


# =========================
# Data
# =========================

def load_data(config):

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

    proxy_base = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    proxy_indices = list(range(config.PROXY_SIZE))
    proxy_dataset = Subset(proxy_base, proxy_indices)

    return train_dataset, test_dataset, proxy_dataset


# =========================
# Clients
# =========================

def create_clients(train_dataset, config):

    labels = torch.tensor(train_dataset.targets)

    client_indices = dirichlet_partition(
        labels.numpy(),
        config.NUM_CLIENTS,
        config.DIRICHLET_ALPHA
    )

    clients = []

    num_noisy_clients = int(
        config.NOISE_CLIENT_RATIO * config.NUM_CLIENTS
    )

    print(f"Injecting noise to {num_noisy_clients}/{config.NUM_CLIENTS} clients")
    print(f"Noise type: {config.NOISE_TYPE}")
    print(f"Noise rate: {config.NOISE_RATE}")

    for i in range(config.NUM_CLIENTS):

        indices = client_indices[i]

        if i < num_noisy_clients and config.NOISE_RATE > 0:
            apply_noise(train_dataset, indices, config, i)

        subset = Subset(train_dataset, indices)

        loader = DataLoader(
            subset,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )

        model = SimpleCNN().to(config.DEVICE)
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

    with torch.no_grad():
        for x, y in loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            outputs = model(x)
            _, predicted = torch.max(outputs, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total


def evaluate_per_client(server_model, clients, config):

    server_model.eval()
    client_accuracies = []

    with torch.no_grad():
        for client in clients:

            correct = 0
            total = 0

            loader = client.train_loader

            for x, y in loader:
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)

                outputs = server_model(x)
                _, predicted = torch.max(outputs, 1)

                total += y.size(0)
                correct += (predicted == y).sum().item()

            acc = correct / total if total > 0 else 0.0
            client_accuracies.append(acc)

    worst_acc = min(client_accuracies)

    return worst_acc


# =========================
# Main
# =========================

def main():

    config = Config()
    set_seed(config.SEED)

    print("=== CONFIG ===")
    print(f"DEVICE={config.DEVICE}")
    print(f"NUM_CLIENTS={config.NUM_CLIENTS}")
    print(f"CLIENT_FRACTION={config.CLIENT_FRACTION}")
    print(f"LOCAL_EPOCHS={config.LOCAL_EPOCHS}")
    print(f"ROUNDS={config.ROUNDS}")
    print(f"DIRICHLET_ALPHA={config.DIRICHLET_ALPHA}")
    print(f"NOISE_CLIENT_RATIO={config.NOISE_CLIENT_RATIO}")
    print(f"NOISE_RATE={config.NOISE_RATE}")
    print(f"NOISE_TYPE={config.NOISE_TYPE}")
    print("==============")

    train_dataset, test_dataset, proxy_dataset = load_data(config)
    clients = create_clients(train_dataset, config)

    global_model = SimpleCNN().to(config.DEVICE)
    server = Server(global_model)

    log_path = f"noise_{int(config.NOISE_RATE*100)}.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "round",
                "global_accuracy",
                "worst_client_accuracy",
                "round_time_sec",
                "num_selected_clients"
            ]
        )

        print("Starting training...")

        for r in range(config.ROUNDS):

            t0 = time.time()

            m = max(1, int(config.CLIENT_FRACTION * config.NUM_CLIENTS))
            selected_clients = random.sample(clients, m)

            client_weights = []
            client_sizes = []

            for client in selected_clients:

                client.model.load_state_dict(
                    server.global_model.state_dict()
                )

                w = client.local_train(global_model=server.global_model)
                client_weights.append(w)
                client_sizes.append(len(client.train_loader.dataset))

            server.aggregate(client_weights, client_sizes)

            if config.USE_R2D2:
                server.distill(
                    [client.model for client in selected_clients],
                    proxy_dataset,
                    config
                )

            global_acc = evaluate_global(
                server.global_model,
                test_dataset,
                config
            )

            worst_acc = evaluate_per_client(
                server.global_model,
                clients,
                config
            )

            dt = time.time() - t0

            print(
                f"Round {r+1:03d} | "
                f"Global={global_acc:.4f} | "
                f"Worst={worst_acc:.4f} | "
                f"time={dt:.1f}s | "
                f"clients={m}"
            )

            writer.writerow(
                [
                    r + 1,
                    f"{global_acc:.6f}",
                    f"{worst_acc:.6f}",
                    f"{dt:.3f}",
                    m
                ]
            )

    print(f"Done. Log saved to: {log_path}")


if __name__ == "__main__":
    main()
