# PAPER_DATA_v1 — S-VLG (v1)

_Sinh tự động bởi `python scripts/compile_paper_data.py --version v1`. Chạy lại lệnh này bất cứ khi nào có số liệu mới — file này sẽ cập nhật theo, không cần chỉnh tay._

## TÓM TẮT TIẾN ĐỘ

- ✅ **Đã đủ (FINAL)**: 0/11
- 🟡 **Có tạm (provisional)**: 0/11
- ⬜ **Còn thiếu (THIẾU)**: 11/11

- ⬜ Bảng 6 — Hiệu năng tổng thể — MISSING
- ⬜ Bảng 7 — Phân rã theo nhóm câu hỏi — MISSING
- ⬜ Bảng 8 — Đánh giá ngoại vi (VQA-RAD / SLAKE) — MISSING
- ⬜ Bảng 9 — Ablation (8 biến thể) — MISSING
- ⬜ Bảng 10 — Risk-coverage — MISSING
- ⬜ Bảng 11 — Chi phí tính toán — MISSING
- ⬜ Hình 7 — PR / ROC — MISSING
- ⬜ Hình 8 — Risk-coverage — MISSING
- ⬜ Hình 9 — Ablation bar chart — MISSING
- ⬜ Hình 10 — Attention heatmap — MISSING
- ⬜ Hình 11 — Đồ thị bằng chứng Graph-RAG — MISSING

---
## Bảng


### Bảng 6 — Hiệu năng tổng thể

_(chưa có file: `outputs/tables/v1/table6_overall.json`)_

| model_name | dataset | seeds | trạng thái | vqa_acc | exact_match | bleu4 | precision | recall | f1 | auc_roc |
|---|---|---|---|---|---|---|---|---|---|---|
| [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] |

### Bảng 7 — Phân rã theo nhóm câu hỏi

_(chưa có file: `outputs/tables/v1/table7_by_category.json`)_

| model_name | dataset | seeds | trạng thái | vqa_acc | exact_match | f1 |
|---|---|---|---|---|---|---|
| [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] |

### Bảng 8 — Đánh giá ngoại vi (VQA-RAD / SLAKE)

_(chưa có file: `outputs/tables/v1/table8_external.json`)_

| model_name | dataset | seeds | trạng thái | vqa_acc | exact_match | bleu4 | f1 | auc_roc |
|---|---|---|---|---|---|---|---|---|
| [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] |

### Bảng 9 — Ablation (8 biến thể)

_(chưa có file: `outputs/tables/v1/table9_ablation.json`)_

| variant_name | dataset | seeds | trạng thái | vqa_acc | exact_match | f1 | auc_roc |
|---|---|---|---|---|---|---|---|
| [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] |

### Bảng 10 — Risk-coverage

_(chưa có file: `outputs/tables/v1/table10_risk_coverage.json`)_

| config_name | dataset | seeds | trạng thái | auc |
|---|---|---|---|---|
| [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] |

### Bảng 11 — Chi phí tính toán

_(chưa có file: `outputs/tables/v1/table11_efficiency.json`)_

| model_name | dataset | seeds | trạng thái | train_time_hours | gpu_mem_gb | inference_latency_ms | num_params |
|---|---|---|---|---|---|---|---|
| [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] | [THIẾU] |

---
## Hình


### Hình 7 — PR / ROC

[THIẾU] — chưa có file: `outputs/figures/data/v1/fig7_pr_roc.json`


### Hình 8 — Risk-coverage

[THIẾU] — chưa có file: `outputs/figures/data/v1/fig8_risk_coverage.json`


### Hình 9 — Ablation bar chart

[THIẾU] — chưa có file: `outputs/figures/data/v1/fig9_ablation.json`


### Hình 10 — Attention heatmap

[THIẾU] — chưa có file: `outputs/figures/data/v1/fig10_attention.json`


### Hình 11 — Đồ thị bằng chứng Graph-RAG

[THIẾU] — chưa có file: `outputs/figures/data/v1/fig11_evidence_graph.json`


---
Xem `PAPER_DATA_MAP.md` để biết mỗi bảng/hình cần train tối thiểu bao nhiêu mới có nghĩa, và `outputs/tables/MANIFEST.md` để biết schema JSON đầy đủ.