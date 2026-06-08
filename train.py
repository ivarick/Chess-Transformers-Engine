"""Train the transformer chess position evaluator from PGN files."""

from __future__ import annotations

import argparse
import gc
import os
import random
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataloader import ChessDataset
from model import TransformerChessModel, count_parameters


def configure_runtime(threads: int | None = None, high_priority: bool = False) -> None:
    """Configure optional performance settings without forcing debug modes."""

    if threads:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
        torch.set_num_threads(threads)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if high_priority:
        with suppress(AttributeError, psutil.Error):
            psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, list[float], list[float]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint)
        return 0, float("inf"), [], []

    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    train_losses = list(checkpoint.get("train_losses", []))
    val_losses = list(checkpoint.get("val_losses", []))
    return start_epoch, best_val_loss, train_losses, val_losses


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    best_val_loss: float,
    train_losses: list[float],
    val_losses: list[float],
    config: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": config,
        },
        checkpoint_path,
    )


def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validation", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / max(1, total_samples)


def train_model(args: argparse.Namespace) -> None:
    configure_runtime(args.threads, args.high_priority)
    set_seed(args.seed)

    device = resolve_device(args.device)
    model = TransformerChessModel(
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Using device: {device}")
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")

    dataset = ChessDataset(
        args.pgn_files,
        max_games_per_file=args.max_games_per_file,
        max_plies_per_game=args.max_plies_per_game,
        min_fullmove_number=args.min_fullmove_number,
        show_progress=not args.quiet_indexing,
    )

    if len(dataset) < 2:
        raise ValueError("Need at least two indexed positions to create train/validation splits.")

    train_size = int((1.0 - args.val_fraction) * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError("The validation split produced an empty dataset. Adjust --val-fraction.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
    )
    criterion = nn.MSELoss()
    use_amp = device.type == "cuda" and not args.disable_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_path = Path(args.checkpoint)
    best_checkpoint_path = Path(args.best_checkpoint)
    start_epoch = 0
    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    if args.resume and checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        start_epoch, best_val_loss, train_losses, val_losses = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

    config = vars(args).copy()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs).squeeze(-1)
                smoothed_targets = targets * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                loss = criterion(outputs, smoothed_targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = inputs.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = total_train_loss / max(1, total_train_samples)
        val_loss = run_validation(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val_loss,
                train_losses,
                val_losses,
                config,
            )
            print(f"Saved best model: {best_checkpoint_path} ({best_val_loss:.4f})")

        save_checkpoint(
            checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_val_loss,
            train_losses,
            val_losses,
            config,
        )
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_files", nargs="+", help="One or more PGN files to index for training.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-games-per-file", type=int, default=2_000_000)
    parser.add_argument("--max-plies-per-game", type=int, default=100)
    parser.add_argument("--min-fullmove-number", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint", default="checkpoints/ultron.pth")
    parser.add_argument("--best-checkpoint", default="checkpoints/ultron_best.pth")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--high-priority", action="store_true")
    parser.add_argument("--quiet-indexing", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pct-start", type=float, default=0.05)
    parser.add_argument("--div-factor", type=float, default=25.0)
    parser.add_argument("--final-div-factor", type=float, default=1000.0)
    return parser.parse_args()


def main() -> int:
    train_model(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
