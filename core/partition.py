# core/partition.py

import numpy as np


def dirichlet_partition(labels, num_clients, alpha, min_size=10, max_attempts=50):
    """
    Dirichlet partition with safety check:
    retries until every client gets at least `min_size` samples.

    This is important for smaller datasets like APTOS,
    while remaining fully compatible with CIFAR/EMNIST.
    """
    labels = np.array(labels)
    num_classes = np.unique(labels).shape[0]

    for _ in range(max_attempts):
        label_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            np.random.shuffle(label_indices[c])

            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()

            split_points = (np.cumsum(proportions) * len(label_indices[c])).astype(int)[:-1]
            split = np.split(label_indices[c], split_points)

            for i in range(num_clients):
                client_indices[i].extend(split[i].tolist())

        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_size:
            for i in range(num_clients):
                np.random.shuffle(client_indices[i])
            return client_indices

    # Fallback: return best effort even if min_size was not satisfied
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices
