# data/aptos_loader.py

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Subset

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


def stratified_split_indices(targets, test_ratio=0.2, seed=1):
    rng = np.random.default_rng(seed)

    targets = np.array(targets)
    train_indices = []
    test_indices = []

    classes = np.unique(targets)

    for c in classes:
        cls_indices = np.where(targets == c)[0]
        rng.shuffle(cls_indices)

        n_test = max(1, int(len(cls_indices) * test_ratio))
        cls_test = cls_indices[:n_test]
        cls_train = cls_indices[n_test:]

        # safety: if class is extremely small, keep at least 1 sample in train too
        if len(cls_train) == 0 and len(cls_test) > 1:
            cls_train = cls_test[-1:]
            cls_test = cls_test[:-1]

        train_indices.extend(cls_train.tolist())
        test_indices.extend(cls_test.tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices


def load_aptos_raw(config):

    # =========================
    # Transforms
    # =========================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet
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

    # Base dataset without transform, only to read labels safely
    base_dataset = ImageFolder(root="data/aptos/train", transform=None)
    targets = base_dataset.targets

    # =========================
    # Stratified 80/20 split
    # =========================
    train_indices, test_indices = stratified_split_indices(
        targets=targets,
        test_ratio=0.2,
        seed=config.SEED
    )

    # Separate datasets so train/test transforms do not mix
    train_base = ImageFolder(
        root="data/aptos/train",
        transform=train_transform
    )

    test_base = ImageFolder(
        root="data/aptos/train",
        transform=test_transform
    )

    train_dataset = Subset(train_base, train_indices)
    test_dataset = Subset(test_base, test_indices)

    # restore targets for compatibility with partition/noise pipeline
    train_dataset.targets = [targets[i] for i in train_indices]
    test_dataset.targets = [targets[i] for i in test_indices]

    return train_dataset, test_dataset
