"""Download public (non-credentialed) Medical VQA datasets used for external
evaluation: VQA-RAD (Lau et al. 2018) and SLAKE (Liu et al. 2021, English only).

Both datasets are public and require no credentials. They are stored under
data/raw/, which is git-ignored (see .gitignore) — do NOT remove them from
version control exclusion.

Sources:
  VQA-RAD (primary) : https://huggingface.co/datasets/flaviagiammarino/vqa-rad
  VQA-RAD (fallback) : https://osf.io/89kps/  (original release, via OSF public API)
  SLAKE              : https://huggingface.co/datasets/BoKelvin/SLAKE

Usage:
  python scripts/download_eval_datasets.py
"""

import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STATS_PATH = PROJECT_ROOT / "outputs" / "tables" / "eval_datasets_stats.json"

VQA_RAD_DIR = RAW_DIR / "vqa-rad"
VQA_RAD_HF_ID = "flaviagiammarino/vqa-rad"
VQA_RAD_OSF_NODE = "89kps"
VQA_RAD_OSF_HOMEPAGE = "https://osf.io/89kps/"

SLAKE_DIR = RAW_DIR / "slake"
SLAKE_HF_ID = "BoKelvin/SLAKE"


def _is_yes_no(answer: str) -> bool:
    return answer.strip().lower() in {"yes", "no"}


# ============================================================== VQA-RAD ===

def download_vqa_rad_hf(out_dir: Path) -> bool:
    """Primary path: load via the `datasets` library from Hugging Face.

    The HF release only exposes (image, question, answer) — no original
    question_type/answer_type annotations. We infer answer_type with the
    standard VQA-RAD heuristic (yes/no answers -> CLOSED, else OPEN);
    question_type is left as None since it isn't recoverable from this copy.
    Images are content-hashed to filenames so duplicate images (the dataset
    card notes a few duplicate triplets) are stored once.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[VQA-RAD] `datasets` library not installed, skipping HF path.")
        return False

    images_dir = out_dir / "images"
    meta_path = out_dir / "metadata.json"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds_dict = load_dataset(VQA_RAD_HF_ID)
    except Exception as e:
        print(f"[VQA-RAD] Failed to load from Hugging Face: {e}")
        return False

    records = []
    for split, ds in ds_dict.items():
        for ex in ds:
            buf = io.BytesIO()
            ex["image"].convert("RGB").save(buf, format="JPEG")
            raw = buf.getvalue()
            image_name = hashlib.md5(raw).hexdigest() + ".jpg"
            image_path = images_dir / image_name
            if not image_path.exists():
                image_path.write_bytes(raw)

            answer = ex["answer"]
            records.append({
                "image": f"images/{image_name}",
                "question": ex["question"],
                "answer": answer,
                "answer_type": "CLOSED" if _is_yes_no(answer) else "OPEN",
                "question_type": None,
                "split": split,
            })

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    n_images = len(list(images_dir.glob("*.jpg")))
    print(f"[VQA-RAD][HF] Saved {len(records)} QA pairs over {n_images} unique images.")
    return True


def download_vqa_rad_osf(out_dir: Path) -> bool:
    """Fallback path: fetch the original release directly from OSF via its
    public API (https://api.osf.io/v2/nodes/89kps/...). This copy includes
    the original question_type / answer_type annotations from Lau et al.
    """
    images_dir = out_dir / "images"
    meta_path = out_dir / "metadata.json"
    images_dir.mkdir(parents=True, exist_ok=True)

    api_root = f"https://api.osf.io/v2/nodes/{VQA_RAD_OSF_NODE}/files/osfstorage/"
    try:
        listing = requests.get(api_root, timeout=30).json()
    except Exception as e:
        print(f"[VQA-RAD][OSF] API request failed: {e}")
        return False

    qa_json_url = None
    image_folder_url = None
    for item in listing.get("data", []):
        name = item["attributes"]["name"]
        if name == "VQA_RAD Dataset Public.json":
            qa_json_url = item["links"]["download"]
        elif name == "VQA_RAD Image Folder":
            image_folder_url = item["relationships"]["files"]["links"]["related"]["href"]

    if not qa_json_url or not image_folder_url:
        print("[VQA-RAD][OSF] Could not locate expected files on OSF project 89kps.")
        return False

    records_raw = requests.get(qa_json_url, timeout=60).json()

    downloaded = 0
    next_url = image_folder_url
    while next_url:
        page = requests.get(next_url, timeout=30).json()
        for item in page.get("data", []):
            name = item["attributes"]["name"]
            dest = images_dir / name
            if dest.exists():
                continue
            resp = requests.get(item["links"]["download"], timeout=60)
            dest.write_bytes(resp.content)
            downloaded += 1
        next_url = page.get("links", {}).get("next")

    records = []
    for r in records_raw:
        image_name = r["image_name"]
        if not (images_dir / image_name).exists():
            continue
        records.append({
            "image": f"images/{image_name}",
            "question": r["question"],
            "answer": r["answer"],
            "answer_type": (r.get("answer_type") or "").upper() or None,
            "question_type": r.get("question_type"),
            "split": None,
        })

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[VQA-RAD][OSF] Downloaded {downloaded} new images, saved {len(records)} QA pairs.")
    return True


def download_vqa_rad(out_dir: Path = VQA_RAD_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        print(f"[VQA-RAD] {meta_path} already exists, skip download.")
        return

    print(f"[VQA-RAD] Source (primary, Hugging Face): https://huggingface.co/datasets/{VQA_RAD_HF_ID}")
    if download_vqa_rad_hf(out_dir):
        return

    print(f"[VQA-RAD] Falling back to OSF: {VQA_RAD_OSF_HOMEPAGE}")
    if not download_vqa_rad_osf(out_dir):
        print("[VQA-RAD] FAILED to download from both HF and OSF.")


# ================================================================ SLAKE ===

def download_slake(out_dir: Path = SLAKE_DIR) -> None:
    """SLAKE is bilingual (en/zh); only English (q_lang == 'en') rows are kept."""
    print(f"[SLAKE] Source: https://huggingface.co/datasets/{SLAKE_HF_ID}")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        print(f"[SLAKE] {meta_path} already exists, skip download.")
        return

    from huggingface_hub import hf_hub_download

    records_all = []
    for split_file, split_name in [
        ("train.json", "train"),
        ("validation.json", "validation"),
        ("test.json", "test"),
    ]:
        path = hf_hub_download(repo_id=SLAKE_HF_ID, filename=split_file, repo_type="dataset")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            r["split"] = split_name
        records_all.extend(data)

    records_en = [r for r in records_all if r.get("q_lang") == "en"]

    images_dir = out_dir / "imgs"
    if not images_dir.exists():
        zip_path = hf_hub_download(repo_id=SLAKE_HF_ID, filename="imgs.zip", repo_type="dataset")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        macosx_junk = out_dir / "__MACOSX"
        if macosx_junk.exists():
            shutil.rmtree(macosx_junk)

    records = []
    for r in records_en:
        img_rel = r["img_name"]
        if not (images_dir / img_rel).exists():
            continue
        records.append({
            "image": f"imgs/{img_rel}",
            "question": r["question"],
            "answer": r["answer"],
            "answer_type": r.get("answer_type"),
            "question_type": r.get("content_type"),
            "split": r["split"],
        })

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(
        f"[SLAKE] Saved {len(records)} English QA pairs "
        f"(out of {len(records_all)} total en+zh rows)."
    )


# ================================================================ stats ===

def compute_stats(name: str, dataset_dir: Path) -> dict:
    meta_path = dataset_dir / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    images = {r["image"] for r in records}
    answer_types = Counter(r.get("answer_type") or "UNKNOWN" for r in records)
    question_types = Counter(r.get("question_type") or "UNKNOWN" for r in records)
    n = len(records)
    n_closed = answer_types.get("CLOSED", 0)
    n_open = answer_types.get("OPEN", 0)

    stats = {
        "num_images": len(images),
        "num_qa_pairs": n,
        "answer_type_counts": dict(answer_types),
        "open_ratio": round(n_open / n, 4) if n else None,
        "closed_ratio": round(n_closed / n, 4) if n else None,
        "question_type_counts": dict(question_types),
    }

    print(f"\n=== {name} stats ===")
    print(f"  images     : {stats['num_images']}")
    print(f"  QA pairs   : {stats['num_qa_pairs']}")
    print(f"  open/closed: {stats['open_ratio']} / {stats['closed_ratio']} "
          f"(counts: {stats['answer_type_counts']})")
    print(f"  question types: {stats['question_type_counts']}")

    return stats


def main() -> None:
    download_vqa_rad()
    download_slake()

    all_stats = {
        "vqa-rad": compute_stats("VQA-RAD", VQA_RAD_DIR),
        "slake": compute_stats("SLAKE (English)", SLAKE_DIR),
    }

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\nStats written to {STATS_PATH}")


if __name__ == "__main__":
    main()
