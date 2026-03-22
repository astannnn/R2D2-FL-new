# core/partition.py

import numpy as np


def dirichlet_partition(labels, num_clients, alpha, min_size=10, max_attempts=50):
    """
    Dirichlet partition with retry logic.
    Important for smaller datasets like APTOS, where some clients
    may otherwise receive too few or even zero samples.
    """
    labels = np.array(labels)
    num_classes = np.unique(labels).shape[0]

    best_client_indices = None
    best_min_size = -1

    for _ in range(max_attempts):
        label_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            np.random.shuffle(label_indices[c])

            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()

            split_points = (
                np.cumsum(proportions) * len(label_indices[c])
            ).astype(int)[:-1]

            split = np.split(label_indices[c], split_points)

            for i in range(num_clients):
                client_indices[i].extend(split[i].tolist())

        for i in range(num_clients):
            np.random.shuffle(client_indices[i])

        sizes = [len(idx) for idx in client_indices]
        current_min = min(sizes)

        if current_min > best_min_size:
            best_min_size = current_min
            best_client_indices = client_indices

        if current_min >= min_size:
            return client_indices

    return best_client_indices
