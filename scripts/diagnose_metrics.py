"""Diagnostic: print raw (question, ref, generated answer, correct?) samples
from a trained V2 checkpoint to find where vqa_acc=0.0 comes from — the
matching function or the generation itself.
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.vqa_dataset import build_dataloader, build_question_tokenizer, load_vqa_records, split_records
from src.eval.metrics import exact_match, normalize_answer
from src.models.su_medvqa import SU_MedVQA
from src.train.checkpoint_utils import load_checkpoint, restore_model_and_optimizer
from src.utils.config import load_version_config

SYSTEM_TEXT = "You are a helpful medical VQA assistant."
CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "v2_smoketest" / "full" / "v2_seed0_epoch002.pt"


def main():
    tokenizer, vocab_size = build_question_tokenizer(test_mode=True)
    config = load_version_config("v2")
    config["model"]["question_vocab_size"] = vocab_size

    model = SU_MedVQA(config, test_mode=True)
    ckpt = load_checkpoint(CKPT)
    restore_model_and_optimizer(ckpt, model)
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt['epoch']}")

    vqa_rad = load_vqa_records("vqa-rad", n=50, seed=0)
    slake = load_vqa_records("slake", n=50, seed=0)
    test_records = split_records(vqa_rad, seed=0)["test"] + split_records(slake, seed=0)["test"]
    dataloader = build_dataloader(test_records, tokenizer, batch_size=4, shuffle=False)

    n_printed = 0
    with torch.no_grad():
        for batch in dataloader:
            z_final, U, _, _ = model(
                batch["images"], batch["question_input_ids"], batch["question_text"],
                system_text=SYSTEM_TEXT, answer_text=None,
            )
            results = model.decoder.generate(
                z_final, U, system_text=SYSTEM_TEXT, evidence_text="",
                question_text=batch["question_text"], gamma=model.decoder.gamma, max_new_tokens=8,
            )
            for i, r in enumerate(results):
                pred_raw = r["answer"]
                ref_raw = batch["answer_text"][i]
                pred_norm = normalize_answer(pred_raw)
                ref_norm = normalize_answer(ref_raw)
                correct = exact_match(pred_raw, ref_raw)
                print(f"[{n_printed+1:02d}] Q: {batch['question_text'][i]!r}")
                print(f"     type={batch['answer_type'][i]}  ref={ref_raw!r} (norm={ref_norm!r})")
                print(f"     pred_raw={pred_raw!r}")
                print(f"     pred_norm={pred_norm!r}  correct={bool(correct)}  needs_review={r['needs_expert_review']}")
                print()
                n_printed += 1
                if n_printed >= 20:
                    return


if __name__ == "__main__":
    main()
