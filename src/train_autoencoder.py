"""Training script for convolutional autoencoder on STL-10."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from src.data import build_dataloader
from src.models.autoencoder import ConvAutoencoder
from src.utils import AverageMeter, count_parameters, get_device, set_seed


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()

    loader = build_dataloader(
        split="unlabeled",
        batch_size=args.batch_size,
        dual_view=False,
        shuffle=True,
        num_workers=args.num_workers,
        img_size=args.img_size,
        root=args.data_root,
        augment=True,
    )

    model = ConvAutoencoder(feature_dim=args.model_dim).to(device)
    print(f"Autoencoder parameters: {count_parameters(model):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for images, _ in progress:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), images.size(0))
            progress.set_postfix(loss=loss_meter.avg, lr=optimizer.param_groups[0]["lr"])

        scheduler.step()
        print(f"Epoch {epoch}: loss={loss_meter.avg:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        if epoch % args.checkpoint_freq == 0 or epoch == args.epochs:
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
            }
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, Path(args.checkpoint_dir) / "autoencoder_latest.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train convolutional autoencoder on STL-10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

