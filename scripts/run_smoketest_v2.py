"""Full-pipeline smoke test for Version 2 (SU-MedVQA), local CPU / test_mode.

Trains a tiny SU_MedVQA on a small VQA-RAD + SLAKE subset, evaluates it, runs
a short ablation, and compiles everything into outputs/PAPER_DATA_v2.md.
Numbers will be BAD (tiny model, few epochs, tiny subset) — that is expected
and fine: the goal is to verify the pipeline runs end to end and produces
correctly-shaped, correctly-labeled ("provisional") numbers, not good scores.

Every number is written through ResultsLogger (status="provisional") — none
are hardcoded. Anything that genuinely cannot be measured locally (real GPU
compute cost) is left absent so compile_paper_data.py reports it as [THIẾU].

Usage:
  python scripts/run_smoketest_v2.py --n 200 --epochs 2
  python scripts/run_smoketest_v2.py --n 16 --epochs 1   # fast sanity check
"""

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.vqa_dataset import build_dataloader, build_question_tokenizer, load_vqa_records, split_records
from src.eval.metrics import bleu_score, breakdown_by_category, closed_question_prf1_auc, exact_match, risk_coverage_curve, vqa_accuracy
from src.models.su_medvqa import SU_MedVQA
from src.train.checkpoint_utils import load_checkpoint
from src.train.train_loop import train
from src.utils.config import load_version_config
from src.utils.results_logger import ResultsLogger

SYSTEM_TEXT = "You are a helpful medical VQA assistant."


def build_config(n_samples_hint: int, vocab_size: int) -> dict:
    config = load_version_config("v2")
    config["model"]["question_vocab_size"] = vocab_size
    return config


def make_model(config: dict, variant: str, seed: int) -> SU_MedVQA:
    torch.manual_seed(seed)
    if variant == "no_rpr":
        return SU_MedVQA(config, test_mode=True, use_rel_pos_bias=False)
    if variant == "no_disentangle":
        return SU_MedVQA(config, test_mode=True, disentangle_deterministic=True)
    # "full" and "no_gate" share the exact same architecture — the gate
    # (Eq. 32) only ever applies at inference (see decoder.py), so "no_gate"
    # trains identically and differs only in the gamma passed to generate().
    return SU_MedVQA(config, test_mode=True)


def make_loss_fn(config: dict, epoch_for_kl_anneal: int = 1):
    def compute_loss_fn(model, batch):
        _, _, fusion_losses, decoder_out = model(
            batch["images"], batch["question_input_ids"], batch["question_text"],
            system_text=SYSTEM_TEXT, answer_text=batch["answer_text"],
        )
        return model.compute_total_loss(decoder_out[0], fusion_losses, epoch=epoch_for_kl_anneal, config=config)
    return compute_loss_fn


@torch.no_grad()
def evaluate(model, dataloader, gamma_override: float = None) -> dict:
    """Runs evaluation over every batch in `dataloader`, returns per-sample
    lists: preds, refs, answer_types, question_types, sources, uncertainties, correct_flags.

    CLOSED (yes/no) questions are answered via decoder.predict_closed — a
    forced binary choice comparing next-token log-probability for "yes" vs
    "no" directly — instead of free-form generate(). Free generation can
    degenerate on a small/under-trained model (greedy decoding collapsing to
    the SAME output regardless of input, observed with the tiny test_mode
    backbone), which silently zeroes out CLOSED-question accuracy and looks
    like a scoring bug rather than a generation failure mode. OPEN questions
    still use generate() (no such forced-choice shortcut exists for them).
    The uncertainty gate (Eq. 32) is still applied uniformly, before this
    OPEN/CLOSED dispatch.
    """
    model.eval()
    preds, refs, answer_types, question_types, sources, uncertainties = [], [], [], [], [], []

    for batch in dataloader:
        z_final, U, _, _ = model(
            batch["images"], batch["question_input_ids"], batch["question_text"],
            system_text=SYSTEM_TEXT, answer_text=None,
        )
        gamma = gamma_override if gamma_override is not None else model.decoder.gamma
        B = z_final.shape[0]
        needs_review = U > gamma

        batch_preds = [None] * B

        gated_idx = [i for i in range(B) if needs_review[i]]
        for i in gated_idx:
            batch_preds[i] = model.decoder.CAUTIOUS_ANSWER

        closed_idx = [i for i in range(B) if not needs_review[i] and batch["answer_type"][i] == "CLOSED"]
        if closed_idx:
            closed_preds = model.decoder.predict_closed(
                z_final[closed_idx], SYSTEM_TEXT, "", [batch["question_text"][i] for i in closed_idx]
            )
            for i, p in zip(closed_idx, closed_preds):
                batch_preds[i] = p

        open_idx = [i for i in range(B) if not needs_review[i] and batch["answer_type"][i] != "CLOSED"]
        if open_idx:
            open_results = model.decoder.generate(
                z_final[open_idx], U[open_idx], system_text=SYSTEM_TEXT, evidence_text="",
                question_text=[batch["question_text"][i] for i in open_idx], gamma=gamma, max_new_tokens=8,
            )
            for i, r in zip(open_idx, open_results):
                batch_preds[i] = r["answer"]

        preds.extend(batch_preds)
        refs.extend(batch["answer_text"])
        answer_types.extend(batch["answer_type"])
        question_types.extend(batch["question_type"])
        sources.extend(batch["source"])
        uncertainties.extend(U.tolist())

    correct_flags = [exact_match(p, r) for p, r in zip(preds, refs)]
    return {
        "preds": preds, "refs": refs, "answer_types": answer_types,
        "question_types": question_types, "sources": sources,
        "uncertainties": uncertainties, "correct_flags": correct_flags,
    }


def log_table6_and_table7(logger: ResultsLogger, eval_out: dict, seed: int, epoch: int, config: dict):
    preds, refs = eval_out["preds"], eval_out["refs"]

    def _metrics_for(idx):
        p = [preds[i] for i in idx]
        r = [refs[i] for i in idx]
        at = [eval_out["answer_types"][i] for i in idx]
        m = {
            "vqa_acc": vqa_accuracy(p, r),
            "exact_match": vqa_accuracy(p, r),
            "bleu1": bleu_score(p, r, max_ngram_order=1),
            "bleu4": bleu_score(p, r, max_ngram_order=4),
        }
        m.update(closed_question_prf1_auc(p, r, at))
        return m

    # Table 6: combined, and per-dataset (this doubles as "Table 8, VQA-RAD/
    # SLAKE riêng" for V2 — see PROJECT_STATE.md/MANIFEST.md, V2 reports
    # per-dataset numbers as extra rows of Table 6 rather than a separate file).
    all_idx = list(range(len(preds)))
    logger.log_metrics(
        table_id="table6_overall", model_name="SU-MedVQA", metrics_dict=_metrics_for(all_idx),
        seed=seed, dataset="vqa-rad+slake", config=config, status="provisional", epochs_trained=epoch,
    )
    for source in sorted(set(eval_out["sources"])):
        idx = [i for i, s in enumerate(eval_out["sources"]) if s == source]
        if not idx:
            continue
        logger.log_metrics(
            table_id="table6_overall", model_name="SU-MedVQA", metrics_dict=_metrics_for(idx),
            seed=seed, dataset=source, config=config, status="provisional", epochs_trained=epoch,
        )

    # Table 7: breakdown by answer_type (OPEN/CLOSED) and by question_type (SLAKE only).
    for label, categories in (("answer_type", eval_out["answer_types"]), ("question_type", eval_out["question_types"])):
        breakdown = breakdown_by_category(preds, refs, categories)
        for category, stats in breakdown.items():
            idx = [i for i, c in enumerate(categories) if (c or "unknown") == category]
            metrics = {"vqa_acc": stats["vqa_acc"], "exact_match": stats["exact_match"]}
            closed_stats = closed_question_prf1_auc(
                [preds[i] for i in idx], [refs[i] for i in idx], [eval_out["answer_types"][i] for i in idx]
            )
            if closed_stats["f1"] is not None:
                metrics["f1"] = closed_stats["f1"]
            logger.log_metrics(
                table_id="table7_by_category", model_name="SU-MedVQA", metrics_dict=metrics,
                seed=seed, dataset=f"{label}:{category}", config=config, status="provisional", epochs_trained=epoch,
            )


def log_risk_coverage(logger: ResultsLogger, eval_out: dict, seed: int, epoch: int, config: dict):
    rc = risk_coverage_curve(eval_out["uncertainties"], eval_out["correct_flags"])
    logger.log_risk_coverage(
        config_name="SU-MedVQA", coverage_points=rc["coverage_points"], risk_values=rc["risk_values"],
        auc=rc["auc"], seed=seed, dataset="vqa-rad+slake (full val+test of smoketest subset)",
        config=config, status="provisional", epochs_trained=epoch,
    )


def log_attention_heatmap(logger: ResultsLogger, model, test_records: list, tokenizer, seed: int, epoch: int):
    """Figure 10 data: for a few localization-type questions (question_type
    in {Organ, Position}), log the [196] flattened 14x14 attention map. Fits
    log_curve_data's generic (x, y, label) shape: x = patch index, y = weight;
    reshape y to (14, 14) when plotting.
    """
    from src.data.vqa_dataset import VQADataset

    candidates = [r for r in test_records if r.question_type in ("Organ", "Position")][:3]
    if not candidates:
        print("  (no Organ/Position samples available in this subset for Figure 10 — skipping)")
        return

    dataset = VQADataset(candidates, tokenizer)
    model.eval()
    with torch.no_grad():
        for i, record in enumerate(candidates):
            item = dataset[i]
            images = item["image"].unsqueeze(0)
            question_input_ids = item["question_input_ids"].unsqueeze(0)
            _, _, _, _, attn = model(
                images, question_input_ids, item["question_text"],
                system_text=SYSTEM_TEXT, answer_text=None, return_attn=True,
            )
            patch_weights = attn[0].mean(dim=0)  # average over question tokens -> [N_p]
            logger.log_curve_data(
                curve_id="fig10_attention",
                x=list(range(patch_weights.shape[0])),
                y=patch_weights.tolist(),
                label=f"{record.source}: {record.question[:50]}",
                seed=seed, status="provisional", epochs_trained=epoch,
            )


def run_ablation(config: dict, train_dataloader, test_dataloader, seed: int, checkpoint_root: Path, logger: ResultsLogger):
    variants = ["full", "no_rpr", "no_gate", "no_disentangle"]
    for variant in variants:
        print(f"\n--- Ablation variant: {variant} ---")
        model = make_model(config, variant, seed=seed)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_fn = make_loss_fn(config)

        written = train(
            model, train_dataloader, optimizer, num_epochs=1,
            checkpoint_dir=checkpoint_root / variant, experiment_version="v2",
            compute_loss_fn=loss_fn, seed=seed,
        )
        epoch = load_checkpoint(written[-1])["epoch"]

        gamma_override = float("inf") if variant == "no_gate" else None
        eval_out = evaluate(model, test_dataloader, gamma_override=gamma_override)
        metrics = {
            "vqa_acc": vqa_accuracy(eval_out["preds"], eval_out["refs"]),
            "exact_match": vqa_accuracy(eval_out["preds"], eval_out["refs"]),
        }
        closed_stats = closed_question_prf1_auc(eval_out["preds"], eval_out["refs"], eval_out["answer_types"])
        if closed_stats["f1"] is not None:
            metrics["f1"] = closed_stats["f1"]
        if closed_stats["auc_roc"] is not None:
            metrics["auc_roc"] = closed_stats["auc_roc"]

        logger.log_ablation(
            variant_name=variant, metrics_dict=metrics, seed=seed, dataset="vqa-rad+slake",
            config=config, status="provisional", epochs_trained=epoch,
        )

        # Figure 9: same metrics, as a bar-chart-ready (x=metric name, y=value) series.
        metric_names = list(metrics.keys())
        logger.log_curve_data(
            curve_id="fig9_ablation", x=metric_names, y=[metrics[m] for m in metric_names],
            label=variant, seed=seed, dataset="vqa-rad+slake", config=config,
            status="provisional", epochs_trained=epoch,
        )

        print(f"  {variant}: vqa_acc={metrics['vqa_acc']:.3f} (epoch={epoch}, status=provisional)")


def log_efficiency(logger: ResultsLogger, model, train_time_seconds: float, test_dataloader, seed: int, epoch: int, config: dict):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch = next(iter(test_dataloader))
    single = {k: (v[:1] if isinstance(v, list) else v[:1]) for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        z_final, U, _, _ = model(single["images"], single["question_input_ids"], single["question_text"],
                                  system_text=SYSTEM_TEXT, answer_text=None)
        model.decoder.generate(z_final, U, system_text=SYSTEM_TEXT, evidence_text="",
                                question_text=single["question_text"], gamma=model.decoder.gamma, max_new_tokens=8)
        inference_latency_ms = (time.time() - t0) * 1000

    metrics = {
        "train_time_hours": train_time_seconds / 3600.0,
        "inference_latency_ms": inference_latency_ms,
        "num_params": num_params,
        # gpu_mem_gb intentionally omitted: cannot be measured on CPU — see
        # PAPER_DATA_MAP.md, Table 11 stays [THIẾU] for this column until a
        # real Colab GPU run measures it.
    }
    logger.log_metrics(
        table_id="table11_efficiency", model_name="SU-MedVQA", metrics_dict=metrics, seed=seed,
        dataset="cpu-local (do tren CPU local; can do lai tren GPU Colab de co so FINAL, "
                "va num_params la cua mo hinh tiny test_mode, khong phai Qwen+LoRA that)",
        config=config, status="provisional", epochs_trained=epoch,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="Samples per dataset (VQA-RAD, SLAKE)")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs for the full model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    checkpoint_root = PROJECT_ROOT / "outputs" / "checkpoints" / "v2_smoketest"

    print("=== Step a: load subset ===")
    tokenizer, vocab_size = build_question_tokenizer(test_mode=True)
    config = build_config(args.n, vocab_size)

    vqa_rad = load_vqa_records("vqa-rad", n=args.n, seed=args.seed)
    slake = load_vqa_records("slake", n=args.n, seed=args.seed)
    vqa_rad_splits = split_records(vqa_rad, seed=args.seed)
    slake_splits = split_records(slake, seed=args.seed)
    print(f"VQA-RAD: {len(vqa_rad)} samples -> train={len(vqa_rad_splits['train'])}, "
          f"val={len(vqa_rad_splits['val'])}, test={len(vqa_rad_splits['test'])}")
    print(f"SLAKE:   {len(slake)} samples -> train={len(slake_splits['train'])}, "
          f"val={len(slake_splits['val'])}, test={len(slake_splits['test'])}")

    train_records = vqa_rad_splits["train"] + slake_splits["train"]
    val_records = vqa_rad_splits["val"] + slake_splits["val"]
    test_records = vqa_rad_splits["test"] + slake_splits["test"]
    val_test_records = val_records + test_records  # for risk-coverage: FULL val+test, no further subsetting

    train_dataloader = build_dataloader(train_records, tokenizer, batch_size=args.batch_size, shuffle=True)
    test_dataloader = build_dataloader(test_records, tokenizer, batch_size=args.batch_size, shuffle=False)
    val_test_dataloader = build_dataloader(val_test_records, tokenizer, batch_size=args.batch_size, shuffle=False)

    logger = ResultsLogger(experiment_version="v2")

    print("\n=== Step b: train the full model ===")
    full_model = make_model(config, "full", seed=args.seed)
    optimizer = torch.optim.SGD(full_model.parameters(), lr=1e-3)
    loss_fn = make_loss_fn(config)

    t0 = time.time()
    written = train(
        full_model, train_dataloader, optimizer, num_epochs=args.epochs,
        checkpoint_dir=checkpoint_root / "full", experiment_version="v2",
        compute_loss_fn=loss_fn, seed=args.seed,
    )
    train_time_seconds = time.time() - t0
    final_epoch = load_checkpoint(written[-1])["epoch"]
    print(f"Trained {final_epoch} epoch(s) in {train_time_seconds:.1f}s -> {written[-1]}")

    print("\n=== Step c: evaluate final checkpoint -> Table 6, 7, 10, Figure 8, Figure 10 ===")
    test_eval = evaluate(full_model, test_dataloader)
    log_table6_and_table7(logger, test_eval, seed=args.seed, epoch=final_epoch, config=config)

    val_test_eval = evaluate(full_model, val_test_dataloader)
    log_risk_coverage(logger, val_test_eval, seed=args.seed, epoch=final_epoch, config=config)

    log_attention_heatmap(logger, full_model, test_records, tokenizer, seed=args.seed, epoch=final_epoch)

    log_efficiency(logger, full_model, train_time_seconds, test_dataloader, seed=args.seed, epoch=final_epoch, config=config)

    print("\n=== Step d: ablation (Table 9) — 1 epoch per variant ===")
    run_ablation(config, train_dataloader, test_dataloader, seed=args.seed, checkpoint_root=checkpoint_root, logger=logger)

    print("\n=== Step f: compile paper data ===")
    import subprocess
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "compile_paper_data.py"), "--version", "v2"], check=True)

    print("\n=== Summary ===")
    print("Bảng 6 (tổng thể): có số (provisional)")
    print("Bảng 7 (phân rã nhóm): có số (provisional)")
    print("Bảng 8 (VQA-RAD/SLAKE riêng): gộp vào Bảng 6 dưới dạng các dòng dataset riêng (xem MANIFEST.md) — có số (provisional)")
    print("Bảng 9 (ablation): có số cho 4 biến thể (provisional)")
    print("Bảng 10 + Hình 8 (risk-coverage): có số, chạy trên TOÀN BỘ val+test của subset (provisional)")
    print("Hình 10 (attention heatmap): có số nếu subset có câu hỏi Organ/Position (SLAKE)")
    print("Bảng 11 (chi phí tính toán): có num_params + train_time_hours + inference_latency_ms đo trên CPU local "
          "(provisional, cần đo lại trên GPU Colab); gpu_mem_gb để trống -> [THIẾU]")
    print(f"\nFile tổng hợp: {PROJECT_ROOT / 'outputs' / 'PAPER_DATA_v2.md'}")


if __name__ == "__main__":
    main()
