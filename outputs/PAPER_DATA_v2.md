# PAPER_DATA_v2 — SU-MedVQA (v2)

_Sinh tự động bởi `python scripts/compile_paper_data.py --version v2`. Chạy lại lệnh này bất cứ khi nào có số liệu mới — file này sẽ cập nhật theo, không cần chỉnh tay._

## TÓM TẮT TIẾN ĐỘ

- ✅ **Đã đủ (FINAL)**: 0/8
- 🟡 **Có tạm (provisional)**: 8/8
- ⬜ **Còn thiếu (THIẾU)**: 0/8

- 🟡 Bảng 6 — Hiệu năng tổng thể — PROVISIONAL
- 🟡 Bảng 7 — Phân rã theo nhóm câu hỏi — PROVISIONAL
- 🟡 Bảng 9 — Ablation (RPR-CoAttention + uncertainty) — PROVISIONAL
- 🟡 Bảng 10 — Risk-coverage — PROVISIONAL
- 🟡 Bảng 11 — Chi phí tính toán — PROVISIONAL
- 🟡 Hình 8 — Risk-coverage — PROVISIONAL
- 🟡 Hình 9 — Ablation bar chart — PROVISIONAL
- 🟡 Hình 10 — Attention heatmap — PROVISIONAL

---
## Bảng


### Bảng 6 — Hiệu năng tổng thể

| model_name | dataset | seeds | trạng thái | vqa_acc | exact_match | bleu4 | precision | recall | f1 | auc_roc |
|---|---|---|---|---|---|---|---|---|---|---|
| SU-MedVQA | vqa-rad+slake | 1 | [TẠM epoch=2] | 0.2386 | 0.2386 | 0.1481 | 0.4500 | 0.2093 | 0.2857 | None |
| SU-MedVQA | slake | 1 | [TẠM epoch=2] | 0.1892 | 0.1892 | 0.0000 | 0.6667 | 0.1176 | 0.2000 | None |
| SU-MedVQA | vqa-rad | 1 | [TẠM epoch=2] | 0.2745 | 0.2745 | 0.2721 | 0.4118 | 0.2692 | 0.3256 | None |

### Bảng 7 — Phân rã theo nhóm câu hỏi

| model_name | dataset | seeds | trạng thái | vqa_acc | exact_match | f1 |
|---|---|---|---|---|---|---|
| SU-MedVQA | answer_type:CLOSED | 1 | [TẠM epoch=2] | 0.4516 | 0.4516 | 0.2857 |
| SU-MedVQA | answer_type:OPEN | 1 | [TẠM epoch=2] | 0.0000 | 0.0000 | [THIẾU] |
| SU-MedVQA | question_type:unknown | 1 | [TẠM epoch=2] | 0.2745 | 0.2745 | 0.3256 |
| SU-MedVQA | question_type:Organ | 1 | [TẠM epoch=2] | 0.3333 | 0.3333 | 0.0000 |
| SU-MedVQA | question_type:Position | 1 | [TẠM epoch=2] | 0.1538 | 0.1538 | 0.8000 |
| SU-MedVQA | question_type:KG | 1 | [TẠM epoch=2] | 0.1538 | 0.1538 | 0.0000 |
| SU-MedVQA | question_type:Plane | 1 | [TẠM epoch=2] | 0.0000 | 0.0000 | 0.0000 |
| SU-MedVQA | question_type:Size | 1 | [TẠM epoch=2] | 0.0000 | 0.0000 | 0.0000 |
| SU-MedVQA | question_type:Abnormality | 1 | [TẠM epoch=2] | 0.4545 | 0.4545 | 0.0000 |
| SU-MedVQA | question_type:Quantity | 1 | [TẠM epoch=2] | 0.0000 | 0.0000 | [THIẾU] |
| SU-MedVQA | question_type:Color | 1 | [TẠM epoch=2] | 0.0000 | 0.0000 | 0.0000 |
| SU-MedVQA | question_type:Modality | 1 | [TẠM epoch=2] | 0.0000 | 0.0000 | 0.0000 |
| SU-MedVQA | question_type:Shape | 1 | [TẠM epoch=2] | 0.0000 | 0.0000 | [THIẾU] |

### Bảng 9 — Ablation (RPR-CoAttention + uncertainty)

| variant_name | dataset | seeds | trạng thái | vqa_acc | exact_match | f1 | auc_roc |
|---|---|---|---|---|---|---|---|
| full | vqa-rad+slake | 1 | [TẠM epoch=1] | 0.2216 | 0.2216 | 0.3333 | [THIẾU] |
| no_rpr | vqa-rad+slake | 1 | [TẠM epoch=1] | 0.2216 | 0.2216 | 0.3333 | [THIẾU] |
| no_gate | vqa-rad+slake | 1 | [TẠM epoch=1] | 0.2216 | 0.2216 | 0.3333 | [THIẾU] |
| no_disentangle | vqa-rad+slake | 1 | [TẠM epoch=1] | 0.2216 | 0.2216 | 0.1724 | [THIẾU] |

### Bảng 10 — Risk-coverage

| config_name | dataset | seeds | trạng thái | auc |
|---|---|---|---|---|
| SU-MedVQA | vqa-rad+slake (full val+test of smoketest subset) | 1 | [TẠM epoch=2] | 0.7679 |

### Bảng 11 — Chi phí tính toán

| model_name | dataset | seeds | trạng thái | train_time_hours | gpu_mem_gb | inference_latency_ms | num_params |
|---|---|---|---|---|---|---|---|
| SU-MedVQA | cpu-local (do tren CPU local; can do lai tren GPU Colab de co so FINAL, va num_params la cua mo hinh tiny test_mode, khong phai Qwen+LoRA that) | 1 | [TẠM epoch=2] | 0.1402 | [THIẾU] | 146.8713 | 128708012 |

---
## Hình


### Hình 8 — Risk-coverage

Đã có **1** đường/series dữ liệu:

- **SU-MedVQA** [TẠM epoch=2]: 300 điểm dữ liệu

File dữ liệu thô để vẽ: `outputs/figures/data/v2/fig8_risk_coverage.json`


### Hình 9 — Ablation bar chart

Đã có **4** đường/series dữ liệu:

- **full** [TẠM epoch=1]: 3 điểm dữ liệu
- **no_rpr** [TẠM epoch=1]: 3 điểm dữ liệu
- **no_gate** [TẠM epoch=1]: 3 điểm dữ liệu
- **no_disentangle** [TẠM epoch=1]: 3 điểm dữ liệu

File dữ liệu thô để vẽ: `outputs/figures/data/v2/fig9_ablation.json`


### Hình 10 — Attention heatmap

Đã có **3** đường/series dữ liệu:

- **slake: Is this a study of the abdomen?** [TẠM epoch=2]: 196 điểm dữ liệu
- **slake: Which part of the body does this image belong to?** [TẠM epoch=2]: 196 điểm dữ liệu
- **slake: Where is the brain edema located?** [TẠM epoch=2]: 196 điểm dữ liệu

File dữ liệu thô để vẽ: `outputs/figures/data/v2/fig10_attention.json`


---
Xem `PAPER_DATA_MAP.md` để biết mỗi bảng/hình cần train tối thiểu bao nhiêu mới có nghĩa, và `outputs/tables/MANIFEST.md` để biết schema JSON đầy đủ.