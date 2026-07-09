"""Render every paper table from outputs/tables/*.json into manuscript-ready
markdown. Cells with no data yet are printed as "[…]" so it's obvious what
experiments are still outstanding. See outputs/tables/MANIFEST.md for the
authoritative table/figure -> JSON file mapping.

Usage:
  python scripts/build_paper_tables.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _placeholder_row(n: int) -> str:
    return "| " + " | ".join(["[…]"] * n) + " |"


def render_grouped_table(
    title: str, filename: str, key_field: str, metric_names: list, version: str = None
) -> None:
    """Render a ResultsLogger-produced table (log_metrics/log_ablation output):
    mean +/- std where >= 2 seeds were aggregated, else the latest single-seed
    raw value, else "[…]" for anything not logged yet.

    `version` ("v1"/"v2") looks under outputs/tables/{version}/{filename},
    matching ResultsLogger(experiment_version=...); None uses the legacy
    unversioned outputs/tables/{filename} (dataset-level artifacts only).
    """
    rel_path = f"{version}/{filename}" if version else filename
    path = TABLES_DIR / rel_path
    print(f"\n### {title}\n")

    n_cols = 3 + len(metric_names)
    header = f"| {key_field} | dataset | seeds | " + " | ".join(metric_names) + " |"
    sep = "|" + "---|" * n_cols

    if not path.exists():
        print(f"_(no data yet — expected file: `outputs/tables/{rel_path}`)_\n")
        print(header)
        print(sep)
        print(_placeholder_row(n_cols))
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    aggregates = data.get("aggregates", [])

    if not records:
        print("_(file exists but has no records yet)_\n")
        print(header)
        print(sep)
        print(_placeholder_row(n_cols))
        return

    print(header)
    print(sep)

    aggregated_keys = {(a[key_field], a.get("dataset")) for a in aggregates}
    for a in aggregates:
        row = [str(a[key_field]), str(a.get("dataset") or "[…]"), str(a["n_seeds"])]
        for m in metric_names:
            if m in a["mean"]:
                row.append(f"{a['mean'][m]:.4f} ± {a['std'][m]:.4f}")
            else:
                row.append("[…]")
        print("| " + " | ".join(row) + " |")

    single_groups = {}
    for r in records:
        key = (r.get(key_field), r.get("dataset"))
        if key in aggregated_keys:
            continue
        single_groups[key] = r  # overwrite so the LAST (most recent) record for this key wins

    for (key_value, dataset), r in single_groups.items():
        row = [str(key_value), str(dataset or "[…]"), "1"]
        for m in metric_names:
            row.append(_fmt(r.get("metrics", {}).get(m, "[…]")))
        print("| " + " | ".join(row) + " |")


def render_eval_datasets_stats() -> None:
    path = TABLES_DIR / "eval_datasets_stats.json"
    print("\n### Table 2 (external eval dataset statistics) — eval_datasets_stats.json\n")

    header = "| dataset | num_images | num_qa_pairs | open_ratio | closed_ratio |"
    sep = "|---|---|---|---|---|"

    if not path.exists():
        print(f"_(no data yet — run scripts/download_eval_datasets.py)_\n")
        print(header)
        print(sep)
        print(_placeholder_row(5))
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    print(header)
    print(sep)
    for name, stats in data.items():
        print(
            f"| {name} | {stats.get('num_images', '[…]')} | {stats.get('num_qa_pairs', '[…]')} | "
            f"{stats.get('open_ratio', '[…]')} | {stats.get('closed_ratio', '[…]')} |"
        )


def render_tbd_table(title: str, filename: str, columns: list, note: str) -> None:
    """Placeholder rendering for tables with no producing script yet."""
    path = TABLES_DIR / filename
    print(f"\n### {title}\n")
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "---|" * len(columns)

    if not path.exists():
        print(f"_(no data yet — {note})_\n")
        print(header)
        print(sep)
        print(_placeholder_row(len(columns)))
        return

    # If the file has been created by some future script, dump its raw rows generically.
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", data if isinstance(data, list) else [])
    print(header)
    print(sep)
    if not records:
        print(_placeholder_row(len(columns)))
        return
    for r in records:
        print("| " + " | ".join(_fmt(r.get(c, "[…]")) for c in columns) + " |")


def main() -> None:
    print("# S-VLG — Paper Tables (auto-generated from outputs/tables/*.json)")
    print("\nSee outputs/tables/MANIFEST.md for the full table/figure mapping.")

    render_tbd_table(
        "Table 2 (MIMIC cohort statistics)",
        "dataset_stats.json",
        ["num_patients", "num_studies", "num_images", "num_qa_pairs"],
        "no MIMIC cohort-building script has produced this file yet",
    )

    render_eval_datasets_stats()

    render_tbd_table(
        "Table 2 (observation window scan)",
        "window_scan.json",
        ["window_hours", "num_patients_retained", "num_qa_pairs"],
        "no window-length ablation script has produced this file yet",
    )

    # V1 (S-VLG) — Table 6-11, under outputs/tables/v1/
    print("\n## Version 1 (S-VLG) tables")

    render_grouped_table(
        "Table 6 (overall performance) — V1",
        "table6_overall.json",
        key_field="model_name",
        metric_names=["vqa_acc", "exact_match", "bleu4", "precision", "recall", "f1", "auc_roc"],
        version="v1",
    )

    render_grouped_table(
        "Table 7 (performance by question category) — V1",
        "table7_by_category.json",
        key_field="model_name",
        metric_names=["vqa_acc", "exact_match", "f1"],
        version="v1",
    )

    render_grouped_table(
        "Table 8 (external eval: VQA-RAD / SLAKE) — V1",
        "table8_external.json",
        key_field="model_name",
        metric_names=["vqa_acc", "exact_match", "bleu4", "f1", "auc_roc"],
        version="v1",
    )

    render_grouped_table(
        "Table 9 (ablation, 8 variants) — V1",
        "table9_ablation.json",
        key_field="variant_name",
        metric_names=["vqa_acc", "exact_match", "f1", "auc_roc"],
        version="v1",
    )

    render_grouped_table(
        "Table 10 (risk-coverage) — V1",
        "table10_risk_coverage.json",
        key_field="config_name",
        metric_names=["auc"],
        version="v1",
    )

    render_grouped_table(
        "Table 11 (compute cost) — V1",
        "table11_efficiency.json",
        key_field="model_name",
        metric_names=["train_time_hours", "gpu_mem_gb", "inference_latency_ms", "num_params"],
        version="v1",
    )

    # V2 (SU-MedVQA) — only overall perf, by-category, ablation (RPR + uncertainty),
    # risk-coverage; no Table 8 (its own primary data already is VQA-RAD/SLAKE) or
    # Table 11 (not part of the V2 manuscript scope) — see PROJECT_STATE.md.
    print("\n## Version 2 (SU-MedVQA) tables")

    render_grouped_table(
        "Table 6 (overall performance) — V2",
        "table6_overall.json",
        key_field="model_name",
        metric_names=["vqa_acc", "exact_match", "bleu4", "precision", "recall", "f1", "auc_roc"],
        version="v2",
    )

    render_grouped_table(
        "Table 7 (performance by question category) — V2",
        "table7_by_category.json",
        key_field="model_name",
        metric_names=["vqa_acc", "exact_match", "f1"],
        version="v2",
    )

    render_grouped_table(
        "Table 9 (ablation: RPR + uncertainty) — V2",
        "table9_ablation.json",
        key_field="variant_name",
        metric_names=["vqa_acc", "exact_match", "f1", "auc_roc"],
        version="v2",
    )

    render_grouped_table(
        "Table 10 (risk-coverage) — V2",
        "table10_risk_coverage.json",
        key_field="config_name",
        metric_names=["auc"],
        version="v2",
    )

    print("\n---")
    print("Legend: mean ± std shown when >= 2 seeds are logged for the same "
          "(model/variant/config, dataset); otherwise the single logged value; "
          "\"[…]\" means no experiment has logged that number yet.")


if __name__ == "__main__":
    main()
