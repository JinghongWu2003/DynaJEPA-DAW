"""Training script for JEPA with Difficulty-Aware Weighting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from tqdm import tqdm

from src.data import build_dataloader
from src.models.jepa import JEPAModel
from src.utils import AverageMeter, count_parameters, get_device, set_seed


def compute_weights(
    per_sample_loss: torch.Tensor,
    indices: torch.Tensor,
    epoch: int,
    max_epochs: int,
    difficulty_bank: torch.Tensor,
    use_daw: bool,
    use_curriculum: bool,
) -> torch.Tensor:
    if not use_daw:
        return torch.ones_like(per_sample_loss)

    eps = 1e-6
    stored_difficulty = difficulty_bank[indices].to(per_sample_loss.device)
    difficulty = torch.where(stored_difficulty > 0, stored_difficulty, per_sample_loss.detach())

    stage = epoch / max_epochs if max_epochs > 0 else 1.0
    if not use_curriculum:
        alpha = 1.0
    elif stage < 0.33:
        alpha = -0.7  # focus on easier samples
    elif stage < 0.66:
        alpha = -0.1  # near-uniform
    else:
        alpha = 0.7  # focus on harder samples

    weights = torch.pow(difficulty + eps, alpha)
    weights = torch.clamp(weights, 0.5, 2.0)
    weights = weights / weights.mean().clamp(min=eps)
    return weights.detach()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()

    loader = build_dataloader(
        split="unlabeled",
        batch_size=args.batch_size,
        dual_view=True,
        shuffle=True,
        num_workers=args.num_workers,
        img_size=args.img_size,
        root=args.data_root,
        augment=True,
    )

    model = JEPAModel(feature_dim=args.model_dim, projector_hidden=args.projector_hidden, predictor_hidden=args.predictor_hidden, ema_decay=args.ema_decay)
    model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    difficulty_bank = torch.zeros(len(loader.dataset))

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for context, target, indices in progress:
            context = context.to(device)
            target = target.to(device)
            indices = indices.to(device)

            per_sample_loss, mean_loss = model(context, target)
            weights = compute_weights(
                per_sample_loss=per_sample_loss.detach(),
                indices=indices,
                epoch=epoch,
                max_epochs=args.epochs,
                difficulty_bank=difficulty_bank,
                use_daw=args.use_daw,
                use_curriculum=args.use_curriculum,
            ).to(device)

            weighted_loss = (weights * per_sample_loss).sum() / weights.sum()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            model.update_moving_average()

            with torch.no_grad():
                difficulty_bank[indices.cpu()] = 0.9 * difficulty_bank[indices.cpu()] + 0.1 * per_sample_loss.cpu()

            loss_meter.update(weighted_loss.item(), context.size(0))
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
            torch.save(ckpt, Path(args.checkpoint_dir) / "jepa_latest.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JEPA with DAW on STL-10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--projector-hidden", type=int, default=512)
    parser.add_argument("--predictor-hidden", type=int, default=512)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--use-daw", action="store_true", help="Enable difficulty-aware weighting")
    parser.add_argument("--use-curriculum", action="store_true", help="Enable curriculum schedule over epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

