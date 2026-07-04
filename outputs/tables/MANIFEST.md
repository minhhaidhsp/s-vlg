# Results Manifest — Paper Table/Figure → JSON File Mapping

Every number appearing in either manuscript MUST be traceable to one of the
files below, written exclusively via `src/utils/results_logger.ResultsLogger`
(never hand-typed into the manuscript, never hardcoded in any script). Each
record carries a timestamp, best-effort git commit hash, `experiment_version`,
dataset name, seed, and a config snapshot (`window_hours`,
`jaccard_threshold_tau`, `retrieval_M`, `gamma_threshold`, `epoch`, `seed`)
for reproducibility.

This codebase produces **two papers** from one shared codebase — see
`PROJECT_STATE.md` for the full rationale:
  - **V1 = S-VLG** (three modalities: image + text + patient/EHR knowledge graph)
  - **V2 = SU-MedVQA** (two modalities: image + text)

Model-result tables/figures (log_metrics/log_ablation/log_risk_coverage) are
version-separated by `ResultsLogger(experiment_version="v1"|"v2")`, which
writes under `outputs/tables/{version}/` and `outputs/figures/data/{version}/`
so V1 and V2 numbers never overwrite each other. Dataset-level statistics
(corpus stats, not model results) are NOT version-namespaced since they
describe the raw data, not a model run.

Run `python scripts/build_paper_tables.py` to render every table below in
manuscript-ready markdown (split into a V1 section and a V2 section), with
real numbers filled in where available and `[…]` for anything still missing.

## Shared, unversioned dataset statistics (`outputs/tables/`)

| Paper item | Version | JSON file | Written by | Required fields |
|---|---|---|---|---|
| Table 2 — MIMIC cohort statistics | V1 only | `dataset_stats.json` | (MIMIC cohort-building script, TBD) | num_patients, num_studies, num_images, num_qa_pairs, split sizes (train/val/test), label prevalence |
| Table 2 — external eval dataset statistics | V1 (Table 8 context) + V2 (primary data) | `eval_datasets_stats.json` | `scripts/download_eval_datasets.py` | per dataset (vqa-rad, slake): num_images, num_qa_pairs, answer_type_counts, open_ratio, closed_ratio, question_type_counts |
| Table 2 — observation window scan | V1 only | `window_scan.json` | (window-length ablation script, TBD) | window_hours, num_patients_retained, num_qa_pairs |

## Version 1 (S-VLG) tables (`outputs/tables/v1/`)

| Paper item | JSON file | Written by | Required fields |
|---|---|---|---|
| Table 6 — overall performance | `v1/table6_overall.json` | `ResultsLogger(experiment_version="v1").log_metrics(table_id="table6_overall", ...)` | model_name, dataset, seed, vqa_acc, exact_match, bleu4, precision, recall, f1, auc_roc |
| Table 7 — performance by question category | `v1/table7_by_category.json` | `log_metrics(table_id="table7_by_category", ...)` | model_name, dataset, seed, question_category, vqa_acc, exact_match, f1 |
| Table 8 — external eval (VQA-RAD / SLAKE) | `v1/table8_external.json` | `log_metrics(table_id="table8_external", ...)` | model_name, dataset ("vqa-rad"/"slake"), seed, vqa_acc, exact_match, bleu4, f1, auc_roc |
| Table 9 — ablation (8 variants) | `v1/table9_ablation.json` | `log_ablation(...)` | variant_name, dataset, seed, vqa_acc, exact_match, f1, auc_roc |
| Table 10 — risk-coverage | `v1/table10_risk_coverage.json` | `log_risk_coverage(...)` | config_name, dataset, seed, coverage_points, risk_values, auc |
| Table 11 — compute cost | `v1/table11_efficiency.json` | `log_metrics(table_id="table11_efficiency", ...)` | model_name, dataset, seed, train_time_hours, gpu_mem_gb, inference_latency_ms, num_params |

## Version 1 (S-VLG) figures (`outputs/figures/data/v1/`)

| Paper item | JSON file | Written by | Required fields |
|---|---|---|---|
| Figure 7 — PR / ROC curves | `v1/fig7_pr_roc.json` | `log_curve_data(curve_id="fig7_pr_roc", ...)` | label, x (recall or fpr), y (precision or tpr), seed |
| Figure 8 — risk-coverage curve | `v1/fig8_risk_coverage.json` | auto-mirrored by `log_risk_coverage(...)` (never call directly) | label (=config_name), x (coverage_points), y (risk_values), seed |
| Figure 9 — ablation bar chart | `v1/fig9_ablation.json` | `log_curve_data(curve_id="fig9_ablation", ...)` | label (=variant_name), x (metric names), y (metric values) |
| Figure 10 — attention heatmap (qualitative example) | `v1/fig10_attention.json` | (qualitative-example export script, TBD) | image_id, question, patch_grid (14×14 attention weights) |
| Figure 11 — Graph-RAG evidence subgraph (qualitative example) | `v1/fig11_evidence_graph.json` | (Graph-RAG qualitative export script, TBD) | query_patient_id, retrieved candidates (patient_id, cosine_score), shared ICD/CPT codes |

## Version 2 (SU-MedVQA) tables (`outputs/tables/v2/`)

V2 has a smaller scope than V1 — no graph/tabular branches, no Graph-RAG, and
no separate Table 8 (V2's *primary* data already is VQA-RAD/SLAKE — reported
as extra per-dataset rows of Table 6 instead of a secondary
external-generalization check). Table 11 (compute cost) IS included for V2
(CPU-local numbers, explicitly labeled as needing a GPU re-measurement for
final — see `PAPER_DATA_MAP.md`).

| Paper item | JSON file | Written by | Required fields |
|---|---|---|---|
| Table 6 — overall performance (+ per-dataset rows for "VQA-RAD/SLAKE riêng") | `v2/table6_overall.json` | `ResultsLogger(experiment_version="v2").log_metrics(table_id="table6_overall", ...)` | model_name, dataset ("vqa-rad+slake" combined, or "vqa-rad"/"slake" per-dataset), seed, vqa_acc, exact_match, bleu4, precision, recall, f1, auc_roc |
| Table 7 — performance by question category | `v2/table7_by_category.json` | `log_metrics(table_id="table7_by_category", ...)` | model_name, dataset (encodes category as "answer_type:X"/"question_type:X"), seed, vqa_acc, exact_match, f1 |
| Table 9 — ablation (RPR-CoAttention + uncertainty control) | `v2/table9_ablation.json` | `log_ablation(...)` | variant_name ("full", "no_rpr", "no_gate", "no_disentangle"), dataset, seed, vqa_acc, exact_match, f1, auc_roc |
| Table 10 — risk-coverage | `v2/table10_risk_coverage.json` | `log_risk_coverage(...)` | config_name, dataset, seed, coverage_points, risk_values, auc |
| Table 11 — compute cost | `v2/table11_efficiency.json` | `log_metrics(table_id="table11_efficiency", ...)` | model_name, dataset (notes CPU-local caveat), seed, train_time_hours, gpu_mem_gb (left absent until measured on GPU), inference_latency_ms, num_params |

## Version 2 (SU-MedVQA) figures (`outputs/figures/data/v2/`)

| Paper item | JSON file | Written by | Required fields |
|---|---|---|---|
| Figure 8 — risk-coverage curve | `v2/fig8_risk_coverage.json` | auto-mirrored by `log_risk_coverage(...)` | label (=config_name), x (coverage_points), y (risk_values), seed |
| Figure 9 — ablation bar chart | `v2/fig9_ablation.json` | `log_curve_data(curve_id="fig9_ablation", ...)` | label (=variant_name), x (metric names), y (metric values) |
| Figure 10 — attention heatmap (qualitative example) | `v2/fig10_attention.json` | (qualitative-example export script, TBD) | image_id, question, patch_grid (14×14 attention weights) |

Files marked "(..., TBD)" have no producing script yet — they are still
missing entirely, and `build_paper_tables.py` will show them as empty
placeholder tables until a script is written to populate them.
