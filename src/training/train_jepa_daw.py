import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import get_unlabeled_loader
from src.daw import DifficultyBuffer, compute_weights
from src.models import JEPAModel
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import str2bool
from src.utils.logging import get_logger
from src.utils.seed import set_seed


MODES = ["baseline", "daw_instant", "daw_ema"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train JEPA with/without DAW on STL-10")
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--daw-alpha", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--w-min", type=float, default=0.5)
    parser.add_argument("--w-max", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--tensorboard", type=str2bool, default=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--print-freq", type=int, default=50)
    return parser.parse_args()


def train_one_epoch(
    model: JEPAModel,
    loader,
    optimizer: optim.Optimizer,
    device: str,
    args,
    logger,
    tb_writer: SummaryWriter | None,
    buffer: DifficultyBuffer | None,
    global_step: int,
) -> int:
    model.train()
    pbar = tqdm(loader, desc="Training", leave=False)
    for step, (views, indices) in enumerate(pbar):
        context_view, target_view = views
        context_view = context_view.to(device)
        target_view = target_view.to(device)
        indices = indices.to(device)

        per_sample_loss, stats = model(context_view, target_view)
        if args.mode == "baseline":
            weights = torch.ones_like(per_sample_loss, device=device)
        elif args.mode == "daw_instant":
            weights = compute_weights(per_sample_loss.detach(), gamma=args.gamma, w_min=args.w_min, w_max=args.w_max)
            if buffer is not None:
                buffer.update(indices, per_sample_loss.detach())
        elif args.mode == "daw_ema":
            if buffer is None:
                raise ValueError("DifficultyBuffer required for EMA mode")
            buffer.update(indices, per_sample_loss.detach())
            difficulty = buffer.get_difficulty(indices, mode="ema")
            weights = compute_weights(difficulty, gamma=args.gamma, w_min=args.w_min, w_max=args.w_max)
        else:
            raise ValueError(f"Unknown mode {args.mode}")

        loss = (weights * per_sample_loss).sum() / (weights.sum() + 1e-6)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target()

        with torch.no_grad():
            weight_stats: Dict[str, float] = {
                "weight_mean": weights.mean().item(),
                "weight_min": weights.min().item(),
                "weight_max": weights.max().item(),
                "per_sample_mean": per_sample_loss.mean().item(),
            }

        if step % args.print_freq == 0:
            msg = {
                "loss": loss.item(),
                **weight_stats,
                **{k: v for k, v in stats.items()},
            }
            logger.info(f"Step {step}: {msg}")

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", loss.item(), global_step)
            tb_writer.add_scalar("train/weight_mean", weight_stats["weight_mean"], global_step)
            tb_writer.add_scalar("train/weight_min", weight_stats["weight_min"], global_step)
            tb_writer.add_scalar("train/weight_max", weight_stats["weight_max"], global_step)
            tb_writer.add_scalar("train/per_sample_mean", weight_stats["per_sample_mean"], global_step)
            global_step += 1

    return global_step


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = get_logger("train", log_file=Path(args.log_dir) / "train.log")
    tb_writer = SummaryWriter(log_dir=args.log_dir) if args.tensorboard else None

    device = args.device
    loader = get_unlabeled_loader(
        data_dir="data",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    model = JEPAModel(momentum=args.ema_momentum).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    buffer = None
    if args.mode in {"daw_instant", "daw_ema"}:
        dataset_size = len(loader.dataset)
        buffer = DifficultyBuffer(dataset_size=dataset_size, alpha=args.daw_alpha, device=device)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, device=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if buffer is not None and "buffer" in ckpt:
            buffer.load_state_dict(ckpt["buffer"])
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        global_step = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            args=args,
            logger=logger,
            tb_writer=tb_writer,
            buffer=buffer,
            global_step=global_step,
        )

        save_path = os.path.join(args.save_dir, f"jepa_{args.mode}_epoch{epoch+1}.pt")
        save_checkpoint(
            save_path,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch + 1,
            global_step=global_step,
            buffer_state=buffer.state_dict() if buffer is not None else None,
        )
        logger.info(f"Saved checkpoint to {save_path}")

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
