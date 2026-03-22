# data/aptos_loader.py

import os
import random
import numpy as np

from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Subset


def create_mini_aptos():
    """
    Debug-only fallback.
    Better to remove this for final experiments, but keep it for local testing.
    """
    print("APTOS not found. Creating mini dataset...")

    for cls in range(5):
        folder = f"data/aptos/train/{cls}"
        os.makedirs(folder, exist_ok=True)

        for i in range(20):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img).save(f"{folder}/img_{i}.jpg")


class SubsetWithTargets(Subset):
    def __init__(self, dataset, indices, targets):
        super().__init__(dataset, indices)
        self.targets = list(targets)


def load_aptos_raw(config):
    """
    Returns:
        train_dataset: SubsetWithTargets
        test_dataset:  SubsetWithTargets

    Safe for:
    - Dirichlet partition
    - noise injection
    - evaluation
    """

    root = "data/aptos/train"

    if not os.path.exists(root):
        create_mini_aptos()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Build two separate base datasets so train/test transforms differ correctly
    base_train = ImageFolder(root=root, transform=train_transform)
    base_test = ImageFolder(root=root, transform=test_transform)

    n = len(base_train)
    indices = list(range(n))

    rng = random.Random(config.SEED)
    rng.shuffle(indices)

    train_size = int(0.8 * n)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_targets = [base_train.targets[i] for i in train_indices]
    test_targets = [base_test.targets[i] for i in test_indices]

    train_dataset = SubsetWithTargets(base_train, train_indices, train_targets)
    test_dataset = SubsetWithTargets(base_test, test_indices, test_targets)

    return train_dataset, test_dataset
