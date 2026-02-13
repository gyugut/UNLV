from copy import deepcopy
from typing import Self

from torch.utils.data import Subset, DataLoader
from torchvision import transforms, utils

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from .base import *
from .imagenet1k import *
from .cifar100 import *
from .cifar10 import *


# Image Visualizer
def show_sample_grid(self, mean, std):
    images, targets = next(iter(self))
    grid_images = utils.make_grid(images, nrow=8, padding=10)
    np_image = np.array(grid_images).transpose((1, 2, 0))
    de_norm_image = np_image * std + mean
    plt.figure(figsize=(10, 10))
    plt.imshow(de_norm_image)

DataLoader.show_sample_grid = show_sample_grid


def dataset_split(dataset, split=0.1, new_transforms=None, random_seed=42):
    """
    Splits the dataset into train and validation sets.
    :param dataset: The dataset to split.
    :param split: The fraction of the dataset to use for validation.
    :param new_transforms: Optional new transforms to apply to the validation set.
    :param random_seed: The random seed for reproducibility.
    :return: A tuple of (train_dataset, val_dataset).
    """
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=split,
        random_state=random_seed,
        shuffle=True,
        stratify=dataset.targets if hasattr(dataset, 'targets') else None
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = deepcopy(dataset)
    if new_transforms:
        val_dataset.transform = new_transforms
    val_dataset = Subset(val_dataset, val_indices)

    return train_dataset, val_dataset


class DatasetHolder:
    def __init__(self, train, test, valid=None, attack=None):
        self.config = train.config
        self.dataset_name = train.dataset_name
        self.num_classes = train.num_classes
        self.train = train
        self.test = test
        self.valid = valid
        self.attack = attack
        self.num_classes = len(train.classes) if hasattr(train, 'classes') else len(test.classes)

    def split_train_valid(self, split=0.1, random_seed=42, new_transform=None):
        if self.valid is None:
            if new_transform is None:
                new_transform = transforms.ToTensor()
            self.train, self.valid = dataset_split(self.train, split, new_transform, random_seed)
        else:
            raise ValueError("Validation set already exists. Use a different split value.")
        return self

    def split_train_attack(self, split=0.1, random_seed=42, new_transform=None) -> Self:
        if self.attack is None:
            if new_transform is None:
                new_transform = transforms.ToTensor()
            self.train, self.attack = dataset_split(self.train, split, new_transform, random_seed)
        else:
            raise ValueError("Attack set already exists. Use a different split value.")
        return self

    def __repr__(self):
        return f"DatasetHolder(train={len(self.train)}, test={len(self.test)}, valid={len(self.valid) if self.valid else 'None'}, attack={len(self.attack) if self.attack else 'None'})"


def build_augmentation(img_size, num_ops=2, magnitude=9):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size),  # Resize Image
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)
    ])
