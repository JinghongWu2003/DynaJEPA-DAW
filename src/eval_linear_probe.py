"""Linear probe evaluation on STL-10 using a frozen encoder."""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
from torch import nn, optim
from tqdm import tqdm

from src.data import build_dataloader
from src.models.autoencoder import ConvAutoencoder
from src.models.jepa import JEPAModel
from src.utils import get_device, set_seed


def load_encoder(model_type: str, checkpoint: str, feature_dim: int, device: torch.device) -> nn.Module:
    if model_type == "jepa":
        model = JEPAModel(feature_dim=feature_dim)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"])
        encoder = model.online_encoder
    elif model_type == "autoencoder":
        model = ConvAutoencoder(feature_dim=feature_dim)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"])
        encoder = model.encoder
    else:
        raise ValueError("model must be 'jepa' or 'autoencoder'")

    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def extract_embeddings(encoder: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = []
    labels = []
    with torch.no_grad():
        for images, targets, _ in tqdm(loader, desc="Extracting", leave=False):
            images = images.to(device)
            emb = encoder(images)
            feats.append(emb.cpu())
            labels.append(targets)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def linear_probe(train_feats: torch.Tensor, train_labels: torch.Tensor, test_feats: torch.Tensor, test_labels: torch.Tensor, epochs: int = 20, lr: float = 1e-2) -> float:
    device = train_feats.device
    classifier = nn.Linear(train_feats.size(1), 10).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        perm = torch.randperm(train_feats.size(0), device=device)
        feats = train_feats[perm]
        labels = train_labels[perm]
        classifier.train()
        logits = classifier(feats)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_feats)
        preds = logits.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()
    return acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe on STL-10")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--model", type=str, default="jepa", choices=["jepa", "autoencoder"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    encoder = load_encoder(args.model, args.checkpoint, args.feature_dim, device)

    train_loader = build_dataloader(
        split="train",
        batch_size=args.batch_size,
        dual_view=False,
        shuffle=False,
        num_workers=args.num_workers,
        img_size=args.img_size,
        root=args.data_root,
        augment=False,
    )
    test_loader = build_dataloader(
        split="test",
        batch_size=args.batch_size,
        dual_view=False,
        shuffle=False,
        num_workers=args.num_workers,
        img_size=args.img_size,
        root=args.data_root,
        augment=False,
    )

    train_feats, train_labels = extract_embeddings(encoder, train_loader, device)
    test_feats, test_labels = extract_embeddings(encoder, test_loader, device)

    train_feats = train_feats.to(device)
    test_feats = test_feats.to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    acc = linear_probe(train_feats, train_labels, test_feats, test_labels, epochs=args.epochs, lr=args.lr)
    print(f"Linear probe accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()

