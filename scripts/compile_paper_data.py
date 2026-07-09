"""Compile ALL logged results for one project version into a single
manuscript-ready file: outputs/PAPER_DATA_{version}.md.

Every cell is labeled [FINAL], [TẠM epoch=N], or [THIẾU] so it's obvious at a
glance what's still provisional and what still needs to be run. Re-run this
script any time new results are logged (e.g. after scripts/eval_checkpoint.py)
— it overwrites the same file, so it always reflects the latest numbers.

Table/figure specs here are kept in sync with outputs/tables/MANIFEST.md and
PAPER_DATA_MAP.md — update all three together if the paper's table/figure
list changes.

Usage:
  python scripts/compile_paper_data.py --version v1
  python scripts/compile_paper_data.py --version v2
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DATA_DIR = PROJECT_ROOT / "outputs" / "figures" / "data"

TABLE_SPECS = {
    "v1": [
        {"title": "Bảng 6 — Hiệu năng tổng thể", "filename": "table6_overall.json",
         "key_field": "model_name",
         "metric_names": ["vqa_acc", "exact_match", "bleu4", "precision", "recall", "f1", "auc_roc"]},
        {"title": "Bảng 7 — Phân rã theo nhóm câu hỏi", "filename": "table7_by_category.json",
         "key_field": "model_name", "metric_names": ["vqa_acc", "exact_match", "f1"]},
        {"title": "Bảng 8 — Đánh giá ngoại vi (VQA-RAD / SLAKE)", "filename": "table8_external.json",
         "key_field": "model_name", "metric_names": ["vqa_acc", "exact_match", "bleu4", "f1", "auc_roc"]},
        {"title": "Bảng 9 — Ablation (8 biến thể)", "filename": "table9_ablation.json",
         "key_field": "variant_name", "metric_names": ["vqa_acc", "exact_match", "f1", "auc_roc"]},
        {"title": "Bảng 10 — Risk-coverage", "filename": "table10_risk_coverage.json",
         "key_field": "config_name", "metric_names": ["auc"]},
        {"title": "Bảng 11 — Chi phí tính toán", "filename": "table11_efficiency.json",
         "key_field": "model_name",
         "metric_names": ["train_time_hours", "gpu_mem_gb", "inference_latency_ms", "num_params"]},
    ],
    "v2": [
        {"title": "Bảng 6 — Hiệu năng tổng thể", "filename": "table6_overall.json",
         "key_field": "model_name",
         "metric_names": ["vqa_acc", "exact_match", "bleu4", "precision", "recall", "f1", "auc_roc"]},
        {"title": "Bảng 7 — Phân rã theo nhóm câu hỏi", "filename": "table7_by_category.json",
         "key_field": "model_name", "metric_names": ["vqa_acc", "exact_match", "f1"]},
        {"title": "Bảng 9 — Ablation (RPR-CoAttention + uncertainty)", "filename": "table9_ablation.json",
         "key_field": "variant_name", "metric_names": ["vqa_acc", "exact_match", "f1", "auc_roc"]},
        {"title": "Bảng 10 — Risk-coverage", "filename": "table10_risk_coverage.json",
         "key_field": "config_name", "metric_names": ["auc"]},
        {"title": "Bảng 11 — Chi phí tính toán", "filename": "table11_efficiency.json",
         "key_field": "model_name",
         "metric_names": ["train_time_hours", "gpu_mem_gb", "inference_latency_ms", "num_params"]},
    ],
}

FIGURE_SPECS = {
    "v1": [
        {"title": "Hình 7 — PR / ROC", "filename": "fig7_pr_roc.json"},
        {"title": "Hình 8 — Risk-coverage", "filename": "fig8_risk_coverage.json"},
        {"title": "Hình 9 — Ablation bar chart", "filename": "fig9_ablation.json"},
        {"title": "Hình 10 — Attention heatmap", "filename": "fig10_attention.json"},
        {"title": "Hình 11 — Đồ thị bằng chứng Graph-RAG", "filename": "fig11_evidence_graph.json"},
    ],
    "v2": [
        {"title": "Hình 8 — Risk-coverage", "filename": "fig8_risk_coverage.json"},
        {"title": "Hình 9 — Ablation bar chart", "filename": "fig9_ablation.json"},
        {"title": "Hình 10 — Attention heatmap", "filename": "fig10_attention.json"},
    ],
}


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _status_label(status: str, epochs_trained) -> str:
    if status == "final":
        return "[FINAL]"
    if epochs_trained is not None:
        return f"[TẠM epoch={epochs_trained}]"
    return "[TẠM]"


def render_table(spec: dict, version: str):
    """Returns (markdown_text, overall_status) where overall_status is one of
    "final" (every logged group is final), "provisional" (some data, not all
    final), or "missing" (no data at all)."""
    path = TABLES_DIR / version / spec["filename"]
    key_field = spec["key_field"]
    metric_names = spec["metric_names"]
    n_cols = 4 + len(metric_names)
    header = f"| {key_field} | dataset | seeds | trạng thái | " + " | ".join(metric_names) + " |"
    sep = "|" + "---|" * n_cols

    lines = [f"\n### {spec['title']}\n"]

    if not path.exists():
        lines.append(f"_(chưa có file: `outputs/tables/{version}/{spec['filename']}`)_\n")
        lines.append(header)
        lines.append(sep)
        lines.append("| " + " | ".join(["[THIẾU]"] * n_cols) + " |")
        return "\n".join(lines), "missing"

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    aggregates = data.get("aggregates", [])

    if not records:
        lines.append("_(file tồn tại nhưng chưa có bản ghi nào)_\n")
        lines.append(header)
        lines.append(sep)
        lines.append("| " + " | ".join(["[THIẾU]"] * n_cols) + " |")
        return "\n".join(lines), "missing"

    lines.append(header)
    lines.append(sep)

    saw_final, saw_provisional = False, False
    aggregated_keys = {(a[key_field], a.get("dataset")) for a in aggregates}

    for a in aggregates:
        group_records = [
            r for r in records if r.get(key_field) == a[key_field] and r.get("dataset") == a.get("dataset")
        ]
        statuses = {r.get("status", "provisional") for r in group_records}
        epochs = [r.get("epochs_trained") for r in group_records if r.get("epochs_trained") is not None]
        is_final = statuses == {"final"}
        label = "[FINAL]" if is_final else (f"[TẠM epoch={min(epochs)}]" if epochs else "[TẠM]")
        saw_final = saw_final or is_final
        saw_provisional = saw_provisional or not is_final

        row = [str(a[key_field]), str(a.get("dataset") or "?"), str(a["n_seeds"]), label]
        for m in metric_names:
            row.append(f"{a['mean'][m]:.4f} ± {a['std'][m]:.4f}" if m in a["mean"] else "[THIẾU]")
        lines.append("| " + " | ".join(row) + " |")

    single_groups = {}
    for r in records:
        key = (r.get(key_field), r.get("dataset"))
        if key in aggregated_keys:
            continue
        single_groups[key] = r  # overwrite so the LAST (most recent) record for this key wins

    for (key_value, dataset), r in single_groups.items():
        status = r.get("status", "provisional")
        label = _status_label(status, r.get("epochs_trained"))
        saw_final = saw_final or (status == "final")
        saw_provisional = saw_provisional or (status != "final")

        row = [str(key_value), str(dataset or "?"), "1", label]
        for m in metric_names:
            row.append(_fmt(r.get("metrics", {}).get(m, "[THIẾU]")))
        lines.append("| " + " | ".join(row) + " |")

    if saw_provisional:
        overall = "provisional"
    elif saw_final:
        overall = "final"
    else:
        overall = "missing"
    return "\n".join(lines), overall


def render_figure(spec: dict, version: str):
    path = FIGURES_DATA_DIR / version / spec["filename"]
    lines = [f"\n### {spec['title']}\n"]

    if not path.exists():
        lines.append(f"[THIẾU] — chưa có file: `outputs/figures/data/{version}/{spec['filename']}`\n")
        return "\n".join(lines), "missing"

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not records:
        lines.append("[THIẾU] — file tồn tại nhưng chưa có đường/điểm dữ liệu nào.\n")
        return "\n".join(lines), "missing"

    lines.append(f"Đã có **{len(records)}** đường/series dữ liệu:\n")
    any_final, any_provisional = False, False
    for r in records:
        status = r.get("status", "provisional")
        label_tag = _status_label(status, r.get("epochs_trained"))
        any_final = any_final or (status == "final")
        any_provisional = any_provisional or (status != "final")
        lines.append(f"- **{r.get('label')}** {label_tag}: {len(r.get('x', []))} điểm dữ liệu")

    lines.append(f"\nFile dữ liệu thô để vẽ: `outputs/figures/data/{version}/{spec['filename']}`\n")
    overall = "provisional" if any_provisional else ("final" if any_final else "missing")
    return "\n".join(lines), overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, choices=["v1", "v2"])
    args = parser.parse_args()
    version = args.version
    version_name = "S-VLG" if version == "v1" else "SU-MedVQA"

    progress = []  # (title, status)
    table_sections = []
    for spec in TABLE_SPECS[version]:
        text, status = render_table(spec, version)
        table_sections.append(text)
        progress.append((spec["title"], status))

    figure_sections = []
    for spec in FIGURE_SPECS[version]:
        text, status = render_figure(spec, version)
        figure_sections.append(text)
        progress.append((spec["title"], status))

    n_final = sum(1 for _, s in progress if s == "final")
    n_prov = sum(1 for _, s in progress if s == "provisional")
    n_missing = sum(1 for _, s in progress if s == "missing")

    out = []
    out.append(f"# PAPER_DATA_{version} — {version_name} ({version})\n")
    out.append(
        f"_Sinh tự động bởi `python scripts/compile_paper_data.py --version {version}`. "
        f"Chạy lại lệnh này bất cứ khi nào có số liệu mới — file này sẽ cập nhật theo, "
        f"không cần chỉnh tay._\n"
    )

    out.append("## TÓM TẮT TIẾN ĐỘ\n")
    out.append(f"- ✅ **Đã đủ (FINAL)**: {n_final}/{len(progress)}")
    out.append(f"- 🟡 **Có tạm (provisional)**: {n_prov}/{len(progress)}")
    out.append(f"- ⬜ **Còn thiếu (THIẾU)**: {n_missing}/{len(progress)}\n")
    icon = {"final": "✅", "provisional": "🟡", "missing": "⬜"}
    for title, status in progress:
        out.append(f"- {icon[status]} {title} — {status.upper()}")

    out.append("\n---\n## Bảng\n")
    out.extend(table_sections)

    out.append("\n---\n## Hình\n")
    out.extend(figure_sections)

    out.append(
        "\n---\nXem `PAPER_DATA_MAP.md` để biết mỗi bảng/hình cần train tối thiểu bao "
        "nhiêu mới có nghĩa, và `outputs/tables/MANIFEST.md` để biết schema JSON đầy đủ."
    )

    out_path = PROJECT_ROOT / "outputs" / f"PAPER_DATA_{version}.md"
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Tiến độ: {n_final} final, {n_prov} provisional, {n_missing} thiếu (trên {len(progress)} mục).")


if __name__ == "__main__":
    main()
