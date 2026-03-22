# data/aptos_loader.py

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split

import os
import numpy as np
from PIL import Image
import torch


def create_mini_aptos():
    print("APTOS not found. Creating mini dataset...")

    for cls in range(5):
        folder = f"data/aptos/train/{cls}"
        os.makedirs(folder, exist_ok=True)

        for i in range(5):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img).save(f"{folder}/img_{i}.jpg")


def load_aptos_raw(config):

    # =========================
    # Transforms
    # =========================
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

    # =========================
    # Load dataset
    # =========================
    if not os.path.exists("data/aptos/train"):
        create_mini_aptos()

    full_dataset = ImageFolder(
        root="data/aptos/train",
        transform=train_transform
    )

    # 80/20 split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(config.SEED)

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=generator
    )

    # restore targets for compatibility
    train_dataset.targets = [
        full_dataset.targets[i] for i in train_dataset.indices
    ]

    test_dataset.targets = [
        full_dataset.targets[i] for i in test_dataset.indices
    ]

    return train_dataset, test_dataset
