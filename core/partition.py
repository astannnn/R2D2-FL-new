import numpy as np


def dirichlet_partition(labels, num_clients, alpha):
    num_classes = np.unique(labels).shape[0]

    label_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        proportions = np.random.dirichlet(
            np.repeat(alpha, num_clients)
        )

        proportions = (
            np.cumsum(proportions) * len(label_indices[c])
        ).astype(int)[:-1]

        split = np.split(label_indices[c], proportions)

        for i, idx in enumerate(split):
            client_indices[i].extend(idx.tolist())

    return client_indices
