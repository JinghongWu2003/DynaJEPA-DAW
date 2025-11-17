import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import get_labeled_loaders
from src.models import JEPAModel
from src.utils.checkpoint import load_checkpoint
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probe for pretrained JEPA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_features(model: JEPAModel, loader, device: str):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for images, targets, _ in tqdm(loader, desc="Extract", leave=False):
            images = images.to(device)
            f = model.online_encoder(images)
            feats.append(f.cpu())
            labels.append(targets)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger("linear_probe")

    train_loader, test_loader = get_labeled_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size
    )

    device = args.device
    model = JEPAModel()
    ckpt = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    train_feats, train_labels = extract_features(model, train_loader, device)
    test_feats, test_labels = extract_features(model, test_loader, device)

    linear_head = nn.Linear(train_feats.size(1), 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_head.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        linear_head.train()
        for batch_idx in tqdm(range(0, len(train_feats), args.batch_size), desc=f"Train epoch {epoch+1}", leave=False):
            x = train_feats[batch_idx : batch_idx + args.batch_size].to(device)
            y = train_labels[batch_idx : batch_idx + args.batch_size].to(device)
            logits = linear_head(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        linear_head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx in range(0, len(test_feats), args.batch_size):
                x = test_feats[batch_idx : batch_idx + args.batch_size].to(device)
                y = test_labels[batch_idx : batch_idx + args.batch_size].to(device)
                logits = linear_head(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / total
        logger.info(f"Epoch {epoch+1}: Linear probe accuracy {acc:.2f}%")

    logger.info("Linear probe finished")


if __name__ == "__main__":
    main()
