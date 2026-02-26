# core/femnist_leaf.py
import json
import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FEMNISTClientDataset(Dataset):
    """
    LEAF FEMNIST stores x as a list of 28*28 flattened grayscale pixels (0..255),
    and y as integer label in [0..61].
    """
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = torch.tensor(self.xs[idx], dtype=torch.float32).view(1, 28, 28) / 255.0
        y = torch.tensor(self.ys[idx], dtype=torch.long)
        return x, y


def _read_leaf_users(json_dir: str):
    users = []
    user_data = {}

    for fp in sorted(glob(os.path.join(json_dir, "*.json"))):
        with open(fp, "r") as f:
            obj = json.load(f)

        for u in obj["users"]:
            users.append(u)
            user_data[u] = obj["user_data"][u]

    return users, user_data


def load_leaf_femnist(
    train_dir: str,
    test_dir: str,
    num_clients: int,
    batch_size: int,
    seed: int = 0,
):
    """
    Returns:
      client_train_loaders: list[DataLoader]
      client_test_loaders:  list[DataLoader]
      client_ids: list[str]   (writer ids)
    """
    rng = np.random.RandomState(seed)

    train_users, train_data = _read_leaf_users(train_dir)
    test_users, test_data = _read_leaf_users(test_dir)

    # keep only users present in both train and test
    users = sorted(list(set(train_users).intersection(set(test_users))))
    rng.shuffle(users)

    # take subset for speed / matching NUM_CLIENTS
    users = users[:num_clients]

    client_train_loaders = []
    client_test_loaders = []

    for u in users:
        tr = train_data[u]
        te = test_data[u]

        train_ds = FEMNISTClientDataset(tr["x"], tr["y"])
        test_ds = FEMNISTClientDataset(te["x"], te["y"])

        client_train_loaders.append(
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        )
        client_test_loaders.append(
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        )

    return client_train_loaders, client_test_loaders, users
