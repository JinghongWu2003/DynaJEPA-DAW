"""Data utilities for STL-10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from PIL import Image
from torchvision import datasets, transforms


STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2241, 0.2215, 0.2239)


def _base_transform(img_size: int = 96) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ]
    )


def _augment_transform(img_size: int = 96) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ]
    )


class DualViewTransform:
    """Create two correlated crops from the same image."""

    def __init__(self, img_size: int = 96):
        context_scale = (0.7, 1.0)
        target_scale = (0.4, 0.8)
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

        self.context_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=context_scale),
                transforms.RandomHorizontalFlip(),
                color_jitter,
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(STL10_MEAN, STL10_STD),
            ]
        )

        self.target_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=target_scale),
                transforms.RandomHorizontalFlip(),
                color_jitter,
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(STL10_MEAN, STL10_STD),
            ]
        )

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.context_transform(img), self.target_transform(img)


@dataclass
class STL10Config:
    root: str = "./data"
    split: str = "unlabeled"  # "train", "test", "unlabeled"
    dual_view: bool = False
    img_size: int = 96
    download: bool = True
    augment: bool = True


class STL10Dataset(torch.utils.data.Dataset[Tuple[torch.Tensor, ...]]):
    """Wrapper dataset that returns indices and optionally two views."""

    def __init__(self, config: STL10Config):
        super().__init__()
        self.config = config
        base_transform = _augment_transform(config.img_size) if config.augment else _base_transform(config.img_size)
        self.transform: Callable[[Image.Image], torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]
        if config.dual_view:
            self.transform = DualViewTransform(config.img_size)
        else:
            self.transform = base_transform

        self.dataset = datasets.STL10(
            root=config.root,
            split=config.split,
            download=config.download,
            transform=self.transform,
        )

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if self.config.split == "unlabeled":
            if self.config.dual_view:
                context, target = sample[0]
                return context, target, index
            return sample[0], index
        # labeled split
        img, label = sample
        if self.config.dual_view:
            context, target = img
            return context, target, label, index
        return img, label, index


def build_dataloader(
    split: str,
    batch_size: int,
    dual_view: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
    img_size: int = 96,
    root: str = "./data",
    augment: bool = True,
) -> torch.utils.data.DataLoader:
    config = STL10Config(root=root, split=split, dual_view=dual_view, img_size=img_size, augment=augment)
    dataset = STL10Dataset(config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader

