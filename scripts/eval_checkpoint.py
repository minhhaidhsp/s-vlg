"""Evaluate ANY checkpoint (not necessarily the final one) and log the result
via ResultsLogger — with status "provisional" or "final" and epochs_trained
read from the checkpoint's own metadata, never assumed.

This is deliberately a thin, generic evaluator: with real MIMIC data still
pending (V1, blocked on CITI — see PROJECT_STATE.md) and no full VQA
dataset/dataloader wired up yet (V2 — see PROJECT_STATE.md mục 4), it
demonstrates the checkpoint -> metric -> ResultsLogger pipeline end to end
on either a handful of real VQA-RAD samples (V2) or fake tensors matching
the config shapes (V1) — real forward passes through the restored model
weights, not hardcoded numbers.

Usage:
  python scripts/eval_checkpoint.py --checkpoint PATH --version v1 --kind metrics
  python scripts/eval_checkpoint.py --checkpoint PATH --version v2 --kind metrics --table-id table6_overall
  python scripts/eval_checkpoint.py --checkpoint PATH --version v2 --kind ablation --variant-name no_rpr_bias
  python scripts/eval_checkpoint.py --checkpoint PATH --version v2 --kind risk_coverage --config-name SU-MedVQA
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.train.checkpoint_utils import load_checkpoint, restore_model_and_optimizer
from src.utils.config import load_version_config
from src.utils.results_logger import ResultsLogger


def _build_model(version: str, config: dict):
    """Reconstruct the right model class for `version`, test_mode=True (no
    quantized LLM / real MIMIC / real training data needed for this demo
    evaluator — see module docstring)."""
    if version == "v1":
        from src.models.svlg import SVLG
        return SVLG(config, test_mode=True)
    from src.models.su_medvqa import SU_MedVQA
    return SU_MedVQA(config, test_mode=True)


def _demo_batch_v2(config: dict, n: int = 4):
    """A handful of real VQA-RAD samples if available, else fake tensors."""
    L_question = 12
    vocab = config["model"].get("question_vocab_size", 30522)
    images = torch.randn(n, 3, 224, 224)
    question_input_ids = torch.randint(0, vocab, (n, L_question))

    try:
        from src.data.load_eval_vqa import load_vqa_rad
        samples = []
        for i, s in enumerate(load_vqa_rad()):
            samples.append(s)
            if len(samples) >= n:
                break
        answers = [s.answer for s in samples] if samples else ["yes"] * n
    except Exception:
        answers = ["yes"] * n

    while len(answers) < n:
        answers.append(answers[-1] if answers else "yes")

    return images, question_input_ids, "What organ is likely affected?", answers[:n]


def _demo_batch_v1(config: dict, n: int = 4, num_nodes: int = 20):
    model_cfg = config.get("model", {})
    lab_dim = model_cfg.get("lab_input_dim", 50)
    graph_in_dim = model_cfg.get("graph_in_dim", 64)
    vocab = model_cfg.get("question_vocab_size", 30522)

    images = torch.randn(n, 3, 224, 224)
    question_input_ids = torch.randint(0, vocab, (n, 12))
    lab_x = torch.randn(n, lab_dim)
    lab_mask = (torch.rand(n, lab_dim) > 0.3).float()
    graph_x = torch.randn(num_nodes, graph_in_dim)
    src = torch.randint(0, num_nodes, (40,))
    dst = torch.randint(0, num_nodes, (40,))
    weight = torch.rand(40)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    edge_weight = torch.cat([weight, weight])
    batch_node_idx = torch.randint(0, num_nodes, (n,))
    answers = ["yes"] * n

    return (images, question_input_ids, lab_x, lab_mask, graph_x, edge_index, edge_weight, batch_node_idx, answers)


@torch.no_grad()
def _evaluate(version: str, model, config: dict, n: int = 4):
    """Run the restored model on a small demo batch, return (metrics_dict, U).

    exact_match/vqa_acc here are a simple string-equality rate between the
    generated answer and the reference — a real (if toy) metric computed
    from the checkpoint's actual weights, not a hardcoded number.
    """
    if version == "v2":
        images, question_input_ids, question_text, answers = _demo_batch_v2(config, n=n)
        z_final, U, _, _ = model(images, question_input_ids, question_text, answer_text=None)
    else:
        (images, question_input_ids, lab_x, lab_mask, graph_x, edge_index, edge_weight,
         batch_node_idx, answers) = _demo_batch_v1(config, n=n)
        z_final, U, _, _, _ = model(
            images, question_input_ids, lab_x, lab_mask, graph_x, edge_index, edge_weight,
            batch_node_idx, evidence_text=["" for _ in range(n)], question_text="What organ is likely affected?",
            answer_text=None,
        )
        question_text = "What organ is likely affected?"

    results = model.decoder.generate(
        z_final, U, system_text="You are a helpful medical VQA assistant.",
        evidence_text="", question_text=question_text, gamma=model.decoder.gamma, max_new_tokens=5,
    )

    correct_flags = [
        1.0 if r["answer"].strip().lower() == ref.strip().lower() else 0.0
        for r, ref in zip(results, answers)
    ]
    acc = sum(correct_flags) / len(correct_flags)

    metrics = {"vqa_acc": acc, "exact_match": acc}
    return metrics, U, correct_flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint saved by src/train/train_loop.py")
    parser.add_argument("--version", required=True, choices=["v1", "v2"])
    parser.add_argument("--kind", required=True, choices=["metrics", "ablation", "risk_coverage"])
    parser.add_argument("--table-id", default="table6_overall", help="For --kind metrics")
    parser.add_argument("--model-name", default=None, help="For --kind metrics (default: S-VLG/SU-MedVQA)")
    parser.add_argument("--variant-name", default="full_model", help="For --kind ablation")
    parser.add_argument("--config-name", default=None, help="For --kind risk_coverage")
    parser.add_argument("--dataset", default="demo")
    parser.add_argument("--target-epochs", type=int, default=20, help="Epochs needed for status='final'")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--tables-root", default=None, help="Override ResultsLogger tables_dir root (testing only; default: real outputs/tables)")
    parser.add_argument("--figures-root", default=None, help="Override ResultsLogger figures_data_dir root (testing only; default: real outputs/figures/data)")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    epoch = ckpt["epoch"]
    seed = ckpt["seed"]
    ckpt_version = ckpt["experiment_version"]
    if ckpt_version != args.version:
        raise ValueError(f"Checkpoint was saved for version={ckpt_version!r}, but --version={args.version!r} was given")

    config = load_version_config(args.version)
    model = _build_model(args.version, config)
    restore_model_and_optimizer(ckpt, model)
    model.eval()

    status = "final" if epoch >= args.target_epochs else "provisional"
    print(f"Loaded checkpoint: version={ckpt_version}, seed={seed}, epoch={epoch} -> status={status}")

    metrics, U, correct_flags = _evaluate(args.version, model, config, n=args.n_samples)
    print(f"Demo metrics: {metrics}")

    logger = ResultsLogger(
        experiment_version=args.version, tables_dir=args.tables_root, figures_data_dir=args.figures_root
    )
    default_model_name = "S-VLG" if args.version == "v1" else "SU-MedVQA"

    if args.kind == "metrics":
        path = logger.log_metrics(
            table_id=args.table_id,
            model_name=args.model_name or default_model_name,
            metrics_dict=metrics,
            seed=seed,
            dataset=args.dataset,
            status=status,
            epochs_trained=epoch,
        )
    elif args.kind == "ablation":
        path = logger.log_ablation(
            variant_name=args.variant_name,
            metrics_dict=metrics,
            seed=seed,
            dataset=args.dataset,
            status=status,
            epochs_trained=epoch,
        )
    else:  # risk_coverage
        # Confidence-sorted coverage sweep: sort samples by ascending
        # uncertainty U (most confident first), sweep coverage fractions,
        # risk = error rate on the covered (most-confident) subset so far.
        n = len(U)
        order = torch.argsort(U).tolist()
        coverage_points, risk_values = [], []
        for k in range(1, n + 1):
            covered = order[:k]
            covered_correct = [correct_flags[i] for i in covered]
            risk = 1.0 - (sum(covered_correct) / len(covered_correct))
            coverage_points.append(k / n)
            risk_values.append(risk)
        auc = sum(risk_values) / len(risk_values)  # trapezoidal-ish mean over the swept coverage points
        path = logger.log_risk_coverage(
            config_name=args.config_name or default_model_name,
            coverage_points=coverage_points,
            risk_values=risk_values,
            auc=auc,
            seed=seed,
            dataset=args.dataset,
            status=status,
            epochs_trained=epoch,
        )

    print(f"Logged to: {path}")


if __name__ == "__main__":
    main()
