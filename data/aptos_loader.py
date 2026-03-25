from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split, Subset

import os
import numpy as np
from PIL import Image


def create_mini_aptos():
    print("APTOS not found. Creating mini dataset...")

    for cls in range(5):
        folder = f"data/aptos/train/{cls}"
        os.makedirs(folder, exist_ok=True)

        for i in range(5):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img).save(f"{folder}/img_{i}.jpg")


def load_aptos_raw(config):

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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

    if not os.path.exists("data/aptos/train"):
        create_mini_aptos()

    # Базовый датасет без transform, чтобы потом отдельно задать train/test transforms
    full_dataset = ImageFolder(root="data/aptos/train", transform=None)

    # 80/20 split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_subset, test_subset = random_split(
        full_dataset,
        [train_size, test_size]
    )

    # Создаем отдельные датасеты с разными transform
    train_dataset_full = ImageFolder(
        root="data/aptos/train",
        transform=train_transform
    )

    test_dataset_full = ImageFolder(
        root="data/aptos/train",
        transform=test_transform
    )

    train_dataset = Subset(train_dataset_full, train_subset.indices)
    test_dataset = Subset(test_dataset_full, test_subset.indices)

    # Восстанавливаем .targets для partitioning/noise
    train_dataset.targets = [
        full_dataset.targets[i] for i in train_subset.indices
    ]

    test_dataset.targets = [
        full_dataset.targets[i] for i in test_subset.indices
    ]

    return train_dataset, test_dataset
