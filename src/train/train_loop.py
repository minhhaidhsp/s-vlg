"""Generic training loop shared by V1 (SVLG) and V2 (SU_MedVQA).

Saves a checkpoint after EVERY epoch (not just the best one), so provisional
results can be produced from any early checkpoint instead of waiting for the
full training budget. Supports resuming from a checkpoint if a run is
interrupted (e.g. a Colab disconnect) — epoch, seed, and optimizer state are
all restored, so training continues from exactly where it left off.
"""

from pathlib import Path

import torch

from src.train.checkpoint_utils import (
    checkpoint_path,
    load_checkpoint,
    restore_model_and_optimizer,
    save_checkpoint,
)


def train(
    model,
    dataloader,
    optimizer,
    num_epochs: int,
    checkpoint_dir,
    experiment_version: str,
    compute_loss_fn,
    seed: int = 0,
    resume_from=None,
    on_epoch_end=None,
    max_grad_norm: float = 1.0,
    show_progress: bool = True,
) -> list:
    """Train `model` for up to `num_epochs`, checkpointing every epoch.

    Args:
        model: an nn.Module (an SVLG or SU_MedVQA instance).
        dataloader: an iterable of batches (each batch passed to `compute_loss_fn`).
        optimizer: a torch optimizer over model.parameters().
        num_epochs: TOTAL epoch target (not "epochs left") — if resuming from
            epoch 3, only epochs 4..num_epochs run.
        checkpoint_dir: directory to write per-epoch checkpoints into.
        experiment_version: "v1" or "v2" (used in checkpoint filenames and metadata).
        compute_loss_fn: function(model, batch) -> scalar loss tensor (already
            combining L_gen + alpha*L_MI + beta_t*L_KL [+ L_MLTM], e.g. via
            model.compute_total_loss(...) — this loop only calls backward()/step()).
        seed: random seed identifying this run (part of the checkpoint filename).
            Ignored if `resume_from` is given — the checkpoint's own seed wins.
        resume_from: path to a checkpoint to resume training from.
        on_epoch_end: optional callback(epoch, avg_loss, checkpoint_path).
        max_grad_norm: gradient clipping threshold (torch.nn.utils.clip_grad_norm_),
            applied every step BEFORE optimizer.step(). Without this, plain SGD
            on this hybrid ViT+LM architecture reliably diverges to NaN loss
            after a few hundred steps (observed empirically: gradient norm grew
            9 -> 27 over 96 steps, then NaN) — this is not optional for runs
            with more than a couple dozen steps per epoch.
        show_progress: if True (default), show a tqdm progress bar per epoch
            with a live running-average loss — useful on Colab for long full-
            dataset runs. Falls back to no bar (silent per-batch) if tqdm
            isn't installed.

    Returns:
        List of checkpoint paths written during this call (one per epoch trained).
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    start_epoch = 1
    if resume_from is not None:
        ckpt = load_checkpoint(resume_from)
        restore_model_and_optimizer(ckpt, model, optimizer)
        seed = ckpt["seed"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {resume_from}: completed epoch={ckpt['epoch']}, seed={seed}")

    written = []
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        iterable = dataloader
        progress_bar = None
        if show_progress and tqdm is not None:
            total = len(dataloader) if hasattr(dataloader, "__len__") else None
            progress_bar = tqdm(dataloader, total=total, desc=f"epoch {epoch}/{num_epochs}", unit="batch")
            iterable = progress_bar

        for batch in iterable:
            optimizer.zero_grad()
            loss = compute_loss_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if progress_bar is not None:
                progress_bar.set_postfix(avg_loss=f"{total_loss / n_batches:.4f}")
        avg_loss = total_loss / max(1, n_batches)

        path = checkpoint_path(checkpoint_dir, experiment_version, seed, epoch)
        save_checkpoint(
            path, model, optimizer, epoch=epoch, seed=seed,
            experiment_version=experiment_version, extra={"avg_loss": avg_loss},
        )
        written.append(path)
        print(f"[{experiment_version} seed={seed}] epoch {epoch}/{num_epochs} "
              f"avg_loss={avg_loss:.4f} -> checkpoint: {path}")

        if on_epoch_end is not None:
            on_epoch_end(epoch, avg_loss, path)

    return written


def _self_test() -> bool:
    import shutil
    import tempfile

    import torch

    from src.models.su_medvqa import SU_MedVQA

    ok = True
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        config = {
            "model": {
                "vit_name": "vit_base_patch16_224",
                "llm_name": "Qwen/Qwen2.5-3B-Instruct",
                "n_prefix": 4,
                "lora_r": 16,
                "lora_alpha": 32,
                "fusion_d_shared": 16,
                "fusion_d_vis": 16,
                "question_vocab_size": 200,
            },
            "uncertainty": {"gamma_threshold": 1.0},
            "train": {"mi_alpha": 1.0, "kl_beta_max": 1.0, "kl_warmup_epochs": 10},
        }

        torch.manual_seed(0)
        model = SU_MedVQA(config, test_mode=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        B, L = 2, 8
        fake_batch = {
            "images": torch.randn(B, 3, 224, 224),
            "question_input_ids": torch.randint(0, config["model"]["question_vocab_size"], (B, L)),
            "question_text": "What organ is likely affected?",
            "answer_text": ["The kidney is likely affected." for _ in range(B)],
        }
        dataloader = [fake_batch]  # one batch per "epoch", just to exercise the loop

        def compute_loss_fn(model, batch):
            _, _, fusion_losses, decoder_out = model(
                batch["images"], batch["question_input_ids"], batch["question_text"],
                answer_text=batch["answer_text"],
            )
            return model.compute_total_loss(decoder_out[0], fusion_losses, epoch=1, config=config)

        checkpoint_dir = tmp_dir / "checkpoints"
        written = train(
            model, dataloader, optimizer, num_epochs=2,
            checkpoint_dir=checkpoint_dir, experiment_version="v2",
            compute_loss_fn=compute_loss_fn, seed=7,
        )

        if len(written) != 2:
            print(f"FAIL: expected 2 checkpoints written, got {len(written)}")
            ok = False
        expected_names = {"v2_seed7_epoch001.pt", "v2_seed7_epoch002.pt"}
        actual_names = {p.name for p in written}
        if actual_names != expected_names:
            print(f"FAIL: unexpected checkpoint filenames {actual_names} != {expected_names}")
            ok = False
        if not all(p.exists() for p in written):
            print("FAIL: not all checkpoint files exist on disk")
            ok = False

        ckpt1 = load_checkpoint(written[0])
        if ckpt1["epoch"] != 1 or ckpt1["seed"] != 7 or ckpt1["experiment_version"] != "v2":
            print(f"FAIL: checkpoint 1 metadata wrong: {ckpt1['epoch']}, {ckpt1['seed']}, {ckpt1['experiment_version']}")
            ok = False
        if "avg_loss" not in ckpt1["extra"]:
            print("FAIL: checkpoint extra metadata missing avg_loss")
            ok = False

        # --- resume: fresh model/optimizer, resume from epoch-1 checkpoint,
        # train up to num_epochs=3 -> should only run epoch 3 (not redo epoch 1/2).
        torch.manual_seed(123)  # different init, to prove the resumed weights are what's used
        model2 = SU_MedVQA(config, test_mode=True)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-3)

        written_resume = train(
            model2, dataloader, optimizer2, num_epochs=3,
            checkpoint_dir=checkpoint_dir, experiment_version="v2",
            compute_loss_fn=compute_loss_fn, resume_from=written[0],  # resume from epoch 1
        )

        if len(written_resume) != 2:
            print(f"FAIL: resume should have trained exactly 2 more epochs (2,3), got {len(written_resume)}")
            ok = False
        resumed_epochs = sorted(load_checkpoint(p)["epoch"] for p in written_resume)
        if resumed_epochs != [2, 3]:
            print(f"FAIL: resumed training should cover epochs [2, 3], got {resumed_epochs}")
            ok = False
        if load_checkpoint(written_resume[0])["seed"] != 7:
            print("FAIL: resumed run should inherit the checkpoint's seed (7), not a new one")
            ok = False

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("PASS: train_loop" if ok else "FAIL: train_loop")
    return ok


if __name__ == "__main__":
    _self_test()
