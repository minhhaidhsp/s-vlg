"""Dataset/DataLoader for VQA-RAD and SLAKE, built on top of
src/data/load_eval_vqa.py — shared by both project versions, but exercised
first for V2 (SU-MedVQA), which trains directly on these public datasets.

Images are resized to 224x224 and ImageNet-normalized (matching the ViT
backbone in src/models/vision_encoder.py). Questions are tokenized with the
SAME tokenizer used by the decoder LLM (see build_question_tokenizer), so the
token ids fed into the co-attention question_embedding table are consistent
with the decoder's own vocabulary. Any dataset lacking its own val split
(VQA-RAD has only train/test) gets one carved out of train, seeded and
reproducible; SLAKE already ships train/validation/test and is used as-is.
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.load_eval_vqa import load_eval_vqa

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_question_tokenizer(test_mode: bool = True, tiny_model_name: str = "sshleifer/tiny-gpt2",
                             llm_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Same tokenizer selection logic as src/models/decoder.py::VQADecoder,
    so question token ids stay consistent with the decoder's own vocabulary.
    Returns (tokenizer, vocab_size).
    """
    from transformers import AutoTokenizer

    model_name = tiny_model_name if test_mode else llm_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, len(tokenizer)


def load_vqa_records(source: str, n: int = None, seed: int = 0, fraction: float = None) -> list:
    """Load all VQASample records for `source` ("vqa-rad"/"slake"), optionally
    capped to a seeded random subset.

    Args:
        n: absolute sample cap (for smoke tests). Ignored if `fraction` is given.
        fraction: take this fraction of THIS dataset's own size (e.g. 0.2 for ~1/5).
            Use this instead of `n` when subsetting multiple datasets of very
            different sizes together (VQA-RAD ~2.2k vs SLAKE ~7k) -- a single
            shared `n` would distort their natural size ratio in the combined
            set, while `fraction` preserves it (each dataset shrinks by the
            same proportion).
    """
    records = list(load_eval_vqa(source))
    if fraction is not None:
        n = round(len(records) * fraction)
    if n is not None and n < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, n)
    return records


def split_records(records: list, seed: int = 0, val_ratio: float = 0.1) -> dict:
    """Group records into {"train", "val", "test"}.

    If the source dataset already provides its own validation split (SLAKE),
    that's used directly. Otherwise (VQA-RAD: only train/test), a val split is
    carved out of "train" via a seeded shuffle — reproducible per seed, and
    "test" is left untouched (never used to derive val, to avoid leaking test
    information into model selection).
    """
    has_own_val = any(r.split == "validation" for r in records)
    if has_own_val:
        return {
            "train": [r for r in records if r.split == "train"],
            "val": [r for r in records if r.split == "validation"],
            "test": [r for r in records if r.split == "test"],
        }

    train_records = [r for r in records if r.split == "train"]
    test_records = [r for r in records if r.split == "test"]

    rng = random.Random(seed)
    shuffled = train_records[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    return {
        "train": shuffled[n_val:],
        "val": shuffled[:n_val],
        "test": test_records,
    }


class VQADataset(Dataset):
    """Wraps a list of VQASample records into a torch Dataset."""

    def __init__(self, records: list, tokenizer, image_size: int = 224, max_question_len: int = 32):
        self.records = records
        self.tokenizer = tokenizer
        self.transform = build_image_transform(image_size)
        self.max_question_len = max_question_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        image = Image.open(r.image_path).convert("RGB")
        image = self.transform(image)

        enc = self.tokenizer(r.question, truncation=True, max_length=self.max_question_len, return_tensors="pt")
        question_input_ids = enc["input_ids"][0]

        return {
            "image": image,
            "question_input_ids": question_input_ids,
            "question_text": r.question,
            "answer_text": r.answer,
            "answer_type": r.answer_type,
            "question_type": r.question_type,
            "source": r.source,
        }


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch: list) -> dict:
        images = torch.stack([b["image"] for b in batch])
        max_len = max(b["question_input_ids"].shape[0] for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["question_input_ids"].shape[0]
            input_ids[i, :L] = b["question_input_ids"]
        return {
            "images": images,
            "question_input_ids": input_ids,
            "question_text": [b["question_text"] for b in batch],
            "answer_text": [b["answer_text"] for b in batch],
            "answer_type": [b["answer_type"] for b in batch],
            "question_type": [b["question_type"] for b in batch],
            "source": [b["source"] for b in batch],
        }
    return collate_fn


def build_dataloader(records: list, tokenizer, batch_size: int = 4, shuffle: bool = False,
                      image_size: int = 224) -> DataLoader:
    dataset = VQADataset(records, tokenizer, image_size=image_size)
    collate_fn = make_collate_fn(pad_token_id=tokenizer.pad_token_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _self_test() -> bool:
    ok = True
    tokenizer, vocab_size = build_question_tokenizer(test_mode=True)

    for source in ("vqa-rad", "slake"):
        records = load_vqa_records(source, n=8, seed=0)
        if len(records) == 0:
            print(f"FAIL: no records loaded for {source} (did you run scripts/download_eval_datasets.py?)")
            ok = False
            continue

        splits = split_records(records, seed=0)
        total = sum(len(v) for v in splits.values())
        if total != len(records):
            print(f"FAIL: {source} split sizes don't add up: {total} != {len(records)}")
            ok = False

        dataloader = build_dataloader(splits["train"] or records, tokenizer, batch_size=2)
        batch = next(iter(dataloader))
        if batch["images"].shape[1:] != (3, 224, 224):
            print(f"FAIL: {source} image shape {batch['images'].shape} != (B,3,224,224)")
            ok = False
        if batch["question_input_ids"].shape[0] != batch["images"].shape[0]:
            print(f"FAIL: {source} batch size mismatch between images and question_input_ids")
            ok = False
        if torch.isnan(batch["images"]).any():
            print(f"FAIL: NaN in {source} images")
            ok = False

    print("PASS: vqa_dataset" if ok else "FAIL: vqa_dataset")
    return ok


if __name__ == "__main__":
    _self_test()
