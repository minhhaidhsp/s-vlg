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


# Keys bitsandbytes attaches for 4-bit (NF4) quantized base-model weights when
# state_dict() is called on a --real (QLoRA) model. These represent the FROZEN
# base Qwen weights (never change during training -- only LoRA/ViT/fusion do),
# but a freshly-constructed model's own state_dict() doesn't always expose them
# under the same strict-matching contract load_state_dict expects, so strict
# loading spuriously fails even though nothing meaningful is actually missing.
_BNB_QUANT_KEY_SUFFIXES = (
    ".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map",
    ".quant_state.bitsandbytes__nf4",
)


def restore_model_and_optimizer(checkpoint: dict, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """Load a checkpoint's state into an existing model (+ optimizer, if given).

    Uses strict=False and then verifies any key mismatch is ONLY the expected
    bitsandbytes quantization-metadata pattern (see _BNB_QUANT_KEY_SUFFIXES) --
    a real missing/unexpected key still raises, so an actual bug won't
    silently pass.
    """
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    unexpected_real = [k for k in load_result.unexpected_keys if not k.endswith(_BNB_QUANT_KEY_SUFFIXES)]
    missing_real = [k for k in load_result.missing_keys if not k.endswith(_BNB_QUANT_KEY_SUFFIXES)]
    if unexpected_real or missing_real:
        raise RuntimeError(
            f"Loading checkpoint left real (non-quantization-metadata) key "
            f"mismatches -- unexpected={unexpected_real[:5]}, missing={missing_real[:5]}"
        )
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer
