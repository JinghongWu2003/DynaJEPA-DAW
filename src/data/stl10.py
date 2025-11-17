import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class DualViewSTL10(Dataset):
    def __init__(self, root: str, split: str, image_size: int = 96, download: bool = True):
        base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        # For JEPA we just create two independently augmented views from same base transform.
        self.transform = base_transform
        self.dataset = datasets.STL10(root=root, split=split, download=download, transform=base_transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        # Create two views; here they are just identical transforms, but could be different.
        view1 = self.transform(img)
        view2 = self.transform(img)
        return (view1, view2), target, idx


def get_unlabeled_loader(data_dir: str = "data", batch_size: int = 256, num_workers: int = 4, image_size: int = 96):
    dataset = DualViewSTL10(root=data_dir, split="unlabeled", image_size=image_size, download=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def get_labeled_loaders(batch_size: int = 256, num_workers: int = 4, image_size: int = 96):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_set = datasets.STL10(root="data", split="train", download=True, transform=transform)
    test_set = datasets.STL10(root="data", split="test", download=True, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader
