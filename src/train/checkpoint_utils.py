"""Checkpoint save/load utilities shared by train_loop.py and eval scripts.

Every checkpoint carries enough metadata (epoch, seed, experiment_version,
optimizer state) to (a) resume training if interrupted, and (b) run analysis
on ANY epoch's checkpoint, not just the final one — this is what makes
provisional (early-epoch) results possible.
"""

from pathlib import Path

import torch


def checkpoint_path(checkpoint_dir, experiment_version: str, seed: int, epoch: int) -> Path:
    """Standard checkpoint filename: {version}_seed{seed}_epoch{epoch:03d}.pt."""
    return Path(checkpoint_dir) / f"{experiment_version}_seed{seed}_epoch{epoch:03d}.pt"


def save_checkpoint(
    path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    experiment_version: str,
    extra: dict = None,
) -> Path:
    """Save a checkpoint with everything needed to resume or evaluate later.

    Args:
        path: destination file path.
        model: the model whose state_dict to save.
        optimizer: the optimizer whose state_dict to save (None if not training).
        epoch: epoch number just completed (1-indexed).
        seed: the run's random seed.
        experiment_version: "v1" or "v2".
        extra: any additional metadata to store (e.g. {"avg_loss": ...}).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "seed": seed,
        "experiment_version": experiment_version,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path, map_location: str = "cpu") -> dict:
    """Load a checkpoint dict (does NOT restore it into a model/optimizer —
    see `restore_model_and_optimizer` for that)."""
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_model_and_optimizer(checkpoint: dict, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """Load a checkpoint's state into an existing model (+ optimizer, if given)."""
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer
