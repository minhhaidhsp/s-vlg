"""Unified loader for public Medical VQA evaluation datasets (VQA-RAD, SLAKE).

Provides a common interface — (image_path, question, answer, answer_type) —
across both datasets, so downstream eval code (and later the MIMIC-derived
dataset) can share one code path. Run scripts/download_eval_datasets.py first
to populate data/raw/vqa-rad/ and data/raw/slake/.
"""

import json
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


class VQASample(NamedTuple):
    image_path: Path
    question: str
    answer: str
    answer_type: Optional[str]  # "OPEN" | "CLOSED" | None
    question_type: Optional[str]
    split: Optional[str]
    source: str


def _load_metadata(dataset_dir: Path) -> list:
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run scripts/download_eval_datasets.py first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vqa_rad(split: Optional[str] = None) -> Iterator[VQASample]:
    dataset_dir = RAW_DIR / "vqa-rad"
    for r in _load_metadata(dataset_dir):
        if split is not None and r.get("split") not in (split, None):
            continue
        yield VQASample(
            image_path=dataset_dir / r["image"],
            question=r["question"],
            answer=r["answer"],
            answer_type=r.get("answer_type"),
            question_type=r.get("question_type"),
            split=r.get("split"),
            source="vqa-rad",
        )


def load_slake(split: Optional[str] = None) -> Iterator[VQASample]:
    dataset_dir = RAW_DIR / "slake"
    for r in _load_metadata(dataset_dir):
        if split is not None and r.get("split") != split:
            continue
        yield VQASample(
            image_path=dataset_dir / r["image"],
            question=r["question"],
            answer=r["answer"],
            answer_type=r.get("answer_type"),
            question_type=r.get("question_type"),
            split=r.get("split"),
            source="slake",
        )


_LOADERS = {
    "vqa-rad": load_vqa_rad,
    "vqa_rad": load_vqa_rad,
    "slake": load_slake,
}


def load_eval_vqa(name: str, split: Optional[str] = None) -> Iterator[VQASample]:
    key = name.lower()
    if key not in _LOADERS:
        raise ValueError(f"Unknown eval dataset: {name!r} (expected one of {sorted(set(_LOADERS))})")
    yield from _LOADERS[key](split=split)


if __name__ == "__main__":
    for dataset_name in ("vqa-rad", "slake"):
        print(f"--- {dataset_name}: first 5 samples ---")
        for i, sample in enumerate(load_eval_vqa(dataset_name)):
            if i >= 5:
                break
            print(
                f"[{sample.source}] {sample.image_path.name} | "
                f"Q: {sample.question} | A: {sample.answer} "
                f"(answer_type={sample.answer_type}, question_type={sample.question_type})"
            )
