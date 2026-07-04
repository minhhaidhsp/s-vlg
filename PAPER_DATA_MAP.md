# PAPER_DATA_MAP.md — Kịch bản số liệu cho bản thảo

Bản đồ chi tiết: mỗi bảng/hình trong bản thảo cần TRƯỜNG SỐ LIỆU gì, script nào
sinh ra, file JSON nào lưu, và **train tối thiểu bao nhiêu thì con số đó mới có
nghĩa**. Danh sách lấy từ `outputs/tables/MANIFEST.md`, sắp theo thứ tự xuất
hiện trong bài (Bảng 2 → 11, Hình 7 → 11).

Trạng thái ("Chưa có" / "Tạm — provisional" / "Chính thức — final") được đọc
tự động từ dữ liệu thật bằng:
```
python scripts/compile_paper_data.py --version v1
python scripts/compile_paper_data.py --version v2
```
File này (PAPER_DATA_MAP.md) mô tả **kịch bản** (cái gì cần, khi nào đủ nghĩa);
`outputs/PAPER_DATA_{version}.md` là **kết quả thực tế đọc được** tại thời
điểm chạy script — luôn ưu tiên tin vào file đó hơn là suy đoán từ đây.

---

## Bảng 2 — Thống kê dữ liệu

| | |
|---|---|
| **Thuộc** | Bảng 2a: V1 riêng · Bảng 2b (eval datasets): V1 + V2 dùng chung · Bảng 2c: V1 riêng |
| **Trường số liệu (2a — cohort MIMIC)** | num_patients, num_studies, num_images, num_qa_pairs, kích thước split (train/val/test), tỷ lệ nhãn (label prevalence) |
| **Trường số liệu (2b — eval datasets)** | mỗi bộ (vqa-rad, slake): num_images, num_qa_pairs, answer_type_counts, open_ratio, closed_ratio, question_type_counts |
| **Trường số liệu (2c — window scan)** | window_hours, num_patients_retained, num_qa_pairs |
| **Script sinh** | 2a: (script dựng cohort MIMIC thật, **CHƯA VIẾT** — chờ CITI) · 2b: `scripts/download_eval_datasets.py` (**ĐÃ CHẠY**) · 2c: (script quét window_hours, **CHƯA VIẾT**) |
| **File JSON** | 2a: `outputs/tables/dataset_stats.json` (không versioned) · 2b: `outputs/tables/eval_datasets_stats.json` (không versioned) · 2c: `outputs/tables/window_scan.json` (không versioned) |
| **Trạng thái hiện tại** | 2a: Chưa có · 2b: **Chính thức** (đã chạy thật, số liệu cố định vì là thống kê corpus, không phụ thuộc epoch) · 2c: Chưa có |
| **Train tối thiểu để có nghĩa** | 2a, 2c: không cần train mô hình — chỉ cần chạy script tiền xử lý dữ liệu. 2b: đã xong, không cần gì thêm. |

---

## Bảng 6 — Hiệu năng tổng thể

| | |
|---|---|
| **Thuộc** | V1 và V2 (mỗi bản một file riêng) |
| **Trường số liệu** | model_name, dataset, seed, vqa_acc, exact_match, bleu4, precision, recall, f1, auc_roc |
| **Script sinh** | script đánh giá checkpoint (V1: `scripts/eval_metrics.py --version v1 --table-id table6_overall`; V2: cùng script `--version v2`) — **CHƯA CÓ dữ liệu thật**, hạ tầng đã sẵn sàng |
| **File JSON** | `outputs/tables/v1/table6_overall.json` · `outputs/tables/v2/table6_overall.json` |
| **Trạng thái hiện tại** | Chưa có (cả hai) |
| **Train tối thiểu để có nghĩa** | **>= 1 epoch** trên tập train + eval trên tập val/test — con số tạm (provisional) đã có ý nghĩa tham khảo xu hướng ngay từ epoch 1. Cần đủ **20 epoch × 3 seed** (theo `configs/config.yaml: train.epochs`, `train.num_seeds`) để lên **final** (mean ± std qua 3 seed). |

---

## Bảng 7 — Phân rã theo nhóm câu hỏi

| | |
|---|---|
| **Thuộc** | V1 và V2 |
| **Trường số liệu** | model_name, dataset, seed, question_category (đóng/mở, hoặc theo content_type như Organ/Modality/...), vqa_acc, exact_match, f1 |
| **Script sinh** | script đánh giá checkpoint, nhóm theo `answer_type`/`question_type` đã có sẵn trong metadata VQA-RAD/SLAKE (`eval_datasets_stats.json`) |
| **File JSON** | `outputs/tables/v1/table7_by_category.json` · `outputs/tables/v2/table7_by_category.json` |
| **Trạng thái hiện tại** | Chưa có (cả hai) |
| **Train tối thiểu để có nghĩa** | Cùng checkpoint với Bảng 6 (chạy chung 1 lượt eval) — **>= 1 epoch**. Vì đây là nhóm nhỏ theo category, cỡ mẫu mỗi nhóm nhỏ hơn, nên số liệu chỉ *ổn định* (ít nhiễu) khi mô hình đã hội tụ tương đối — final cần cùng số epoch × seed như Bảng 6. |

---

## Bảng 8 — Đánh giá ngoại vi VQA-RAD/SLAKE

| | |
|---|---|
| **Thuộc** | **V1 riêng** (V2 không có — VQA-RAD/SLAKE đã là dữ liệu chính của V2, xem Bảng 6/7 của V2) |
| **Trường số liệu** | model_name, dataset ("vqa-rad"/"slake"), seed, vqa_acc, exact_match, bleu4, f1, auc_roc |
| **Script sinh** | script đánh giá checkpoint V1 chạy trên VQA-RAD/SLAKE (zero-shot/generalization test) — **CHƯA VIẾT riêng**, có thể tái dùng `scripts/eval_metrics.py --version v1 --dataset vqa-rad` |
| **File JSON** | `outputs/tables/v1/table8_external.json` |
| **Trạng thái hiện tại** | Chưa có |
| **Train tối thiểu để có nghĩa** | Cần model V1 đã train xong trên MIMIC ở mức tối thiểu hội tụ (**>= vài epoch**, khuyến nghị không lấy từ epoch 1 vì đây là test ngoài phân phối — số liệu quá sớm dễ gây hiểu lầm là mô hình generalize kém trong khi thực ra chỉ là chưa train đủ). Final cần cùng epoch × seed với Bảng 6. |

---

## Bảng 9 — Ablation (8 biến thể V1 / RPR+uncertainty V2)

| | |
|---|---|
| **Thuộc** | V1 (8 biến thể) và V2 (biến thể RPR-CoAttention + uncertainty gate) |
| **Trường số liệu** | variant_name, dataset, seed, vqa_acc, exact_match, f1, auc_roc |
| **Script sinh** | script đánh giá checkpoint, gọi `ResultsLogger.log_ablation(...)` mỗi biến thể — **CHƯA CÓ dữ liệu thật** |
| **File JSON** | `outputs/tables/v1/table9_ablation.json` · `outputs/tables/v2/table9_ablation.json` |
| **Trạng thái hiện tại** | Chưa có (cả hai) |
| **Train tối thiểu để có nghĩa** | **Mỗi biến thể phải train riêng >= 1 epoch** (ablation không chia sẻ checkpoint với mô hình đầy đủ — đổi kiến trúc/tắt module thì phải train lại từ đầu). Đây là phần TỐN THỜI GIAN NHẤT: V1 cần 8 lần train, V2 cần vài lần tùy số biến thể — xem mục "Thứ tự ưu tiên" bên dưới. |

---

## Bảng 10 — Risk-coverage

| | |
|---|---|
| **Thuộc** | V1 và V2 |
| **Trường số liệu** | config_name, dataset, seed, coverage_points, risk_values, auc |
| **Script sinh** | script đánh giá checkpoint, quét ngưỡng bất định trên **toàn bộ val+test** để dựng đường risk-coverage — `ResultsLogger.log_risk_coverage(...)` (tự động mirror sang Hình 8) |
| **File JSON** | `outputs/tables/v1/table10_risk_coverage.json` · `outputs/tables/v2/table10_risk_coverage.json` |
| **Trạng thái hiện tại** | Chưa có (cả hai) |
| **Train tối thiểu để có nghĩa** | Khác Bảng 6/7 (có thể tạm dùng 1 phần dữ liệu) — risk-coverage **BẮT BUỘC eval xong toàn bộ val+test** của checkpoint đó để đường cong không bị thiên lệch do cỡ mẫu nhỏ. Có thể lấy tạm ở checkpoint epoch sớm (vẫn đúng "coverage toàn tập" tại epoch đó), nhưng KHÔNG được lấy trên một tập con eval để tiết kiệm thời gian — đó là nguồn sai số dễ gây hiểu lầm nhất trong toàn bộ hệ thống số liệu này. |

---

## Bảng 11 — Chi phí tính toán

| | |
|---|---|
| **Thuộc** | **V1 riêng** (V2 không nằm trong scope bản thảo V2) |
| **Trường số liệu** | model_name, dataset, seed, train_time_hours, gpu_mem_gb, inference_latency_ms, num_params |
| **Script sinh** | đo trực tiếp trong `train_loop.py` (thời gian/epoch × số epoch, GPU mem peak) + script đo latency suy diễn trên checkpoint — **CHƯA CÓ dữ liệu thật** |
| **File JSON** | `outputs/tables/v1/table11_efficiency.json` |
| **Trạng thái hiện tại** | Chưa có |
| **Train tối thiểu để có nghĩa** | train_time_hours/gpu_mem_gb đo được ngay từ **1 epoch** (ngoại suy tuyến tính ra 20 epoch, ghi rõ là ước lượng cho tới khi có full run). inference_latency_ms/num_params đo được với **bất kỳ checkpoint nào** (kể cả epoch 1), không phụ thuộc train nhiều hay ít. |

---

## Hình 7 — PR/ROC

| | |
|---|---|
| **Thuộc** | **V1 riêng** |
| **Dữ liệu cần có** | các đường (label) với mảng x (recall hoặc fpr), y (precision hoặc tpr) |
| **Script sinh** | script đánh giá checkpoint (nhị phân hóa theo answer_type hoặc theo ngưỡng uncertainty) → `log_curve_data(curve_id="fig7_pr_roc", ...)` |
| **File JSON** | `outputs/figures/data/v1/fig7_pr_roc.json` |
| **Trạng thái hiện tại** | Chưa có |
| **Train tối thiểu để có nghĩa** | Cùng checkpoint với Bảng 6, eval trên toàn bộ test set (đường cong cần đủ điểm ngưỡng để mượt) — >= 1 epoch cho bản tạm, hội tụ đầy đủ cho bản final. |

## Hình 8 — Risk-coverage

| | |
|---|---|
| **Thuộc** | V1 và V2 |
| **Dữ liệu cần có** | tự động mirror từ Bảng 10 — KHÔNG gọi `log_curve_data` trực tiếp cho hình này |
| **File JSON** | `outputs/figures/data/v1/fig8_risk_coverage.json` · `.../v2/fig8_risk_coverage.json` |
| **Trạng thái hiện tại** | Chưa có (cả hai) |
| **Train tối thiểu để có nghĩa** | Giống hệt Bảng 10. |

## Hình 9 — Ablation bar chart

| | |
|---|---|
| **Thuộc** | V1 và V2 |
| **Dữ liệu cần có** | label=variant_name, x=tên metric, y=giá trị metric |
| **Script sinh** | sau khi có đủ Bảng 9, gọi `log_curve_data(curve_id="fig9_ablation", ...)` cho mỗi biến thể |
| **File JSON** | `outputs/figures/data/v1/fig9_ablation.json` · `.../v2/fig9_ablation.json` |
| **Trạng thái hiện tại** | Chưa có (cả hai) |
| **Train tối thiểu để có nghĩa** | Giống Bảng 9 — cần MỌI biến thể đã train xong ít nhất 1 epoch để vẽ được biểu đồ so sánh có ý nghĩa (thiếu 1 biến thể thì biểu đồ vẫn vẽ được nhưng thiếu cột, không nên coi là final). |

## Hình 10 — Attention heatmap (ca định tính)

| | |
|---|---|
| **Thuộc** | V1 và V2 |
| **Dữ liệu cần có** | image_id, question, patch_grid (trọng số attention lưới 14×14 từ RPR-CoAttention) |
| **Script sinh** | script export ca định tính từ checkpoint bất kỳ — **CHƯA VIẾT** |
| **File JSON** | `outputs/figures/data/v1/fig10_attention.json` · `.../v2/fig10_attention.json` |
| **Trạng thái hiện tại** | Chưa có (cả hai) |
| **Train tối thiểu để có nghĩa** | Về mặt kỹ thuật chạy được ngay ở **epoch 1** (chỉ cần forward pass lấy attention weights ra), nhưng để minh họa có ý nghĩa (attention "nhìn đúng chỗ") nên lấy từ checkpoint đã hội tụ tương đối tốt — khuyến nghị lấy cùng lúc với checkpoint final của Bảng 6. |

## Hình 11 — Đồ thị bằng chứng Graph-RAG (ca định tính)

| | |
|---|---|
| **Thuộc** | **V1 riêng** (V2 không có Graph-RAG) |
| **Dữ liệu cần có** | query_patient_id, các ứng viên truy hồi (patient_id, cosine_score), mã ICD/CPT chung |
| **Script sinh** | `src/graph/retrieval.retrieve()` + `src/graph/linearize.py` trên một bệnh nhân ví dụ — hạ tầng đã có (`scripts/test_graph_rag.py` minh họa với đồ thị giả), chỉ cần chạy trên đồ thị MIMIC thật và export ra JSON |
| **File JSON** | `outputs/figures/data/v1/fig11_evidence_graph.json` |
| **Trạng thái hiện tại** | Chưa có (cần đồ thị MIMIC thật, chờ CITI) |
| **Train tối thiểu để có nghĩa** | **Không cần train mô hình** — chỉ cần `PatientKnowledgeGraph` dựng từ dữ liệu MIMIC thật + GraphSAGE (có thể dùng trọng số ngẫu nhiên/mới train sơ bộ, vì đây là minh họa cơ chế truy hồi, không phải đánh giá hiệu năng). |

---

## Thứ tự ưu tiên lấy số

Nguyên tắc: **lấy số từ 1 lần train chính trước, số cần train riêng (ablation)
lấy sau cùng** — vì ablation tốn gấp nhiều lần công sức train.

### Ưu tiên 1 — từ MỘT lần train mô hình đầy đủ (full model, seed bất kỳ, checkpoint sớm)
1. Bảng 6 (hiệu năng tổng thể) — chỉ cần 1 epoch + eval để có số tạm đầu tiên.
2. Bảng 7 (phân rã theo nhóm câu hỏi) — cùng lượt eval với Bảng 6.
3. Bảng 11 (chi phí tính toán) — đo được ngay cả ở epoch 1 (latency/num_params không đổi theo epoch; train_time/GPU mem ngoại suy được).
4. Hình 10 (attention heatmap) — chạy được ngay từ checkpoint bất kỳ.
5. Bảng 8 (VQA-RAD/SLAKE ngoại vi, V1) — nên đợi vài epoch (không lấy ngay epoch 1) để tránh hiểu lầm.
6. Bảng 10 + Hình 8 (risk-coverage) — cần eval xong TOÀN BỘ val+test, nên chạy sau khi checkpoint đã tương đối ổn định (không bắt buộc phải là checkpoint cuối, nhưng phải quét hết tập chứ không phải tập con).
7. Hình 7 (PR/ROC, V1) — cùng lượt eval với Bảng 10.
8. Hình 11 (đồ thị bằng chứng, V1) — không phụ thuộc quá trình train mô hình, có thể lấy song song bất cứ lúc nào có đồ thị MIMIC thật.

### Ưu tiên 2 — cần train RIÊNG cho từng cấu hình (tốn thời gian nhất, làm sau cùng)
9. Bảng 9 + Hình 9 (ablation) — V1: 8 biến thể × >=1 epoch mỗi biến thể; V2: ít biến thể hơn (RPR bật/tắt, uncertainty gate bật/tắt). Chỉ bắt đầu sau khi Ưu tiên 1 đã cho số liệu tạm ổn định cho biến thể "full model", để biết baseline so sánh.

### Ưu tiên 3 — chỉ có ý nghĩa khi train xong đủ 20 epoch × 3 seed (FINAL)
10. Nâng mọi bảng ở Ưu tiên 1 & 2 từ "provisional" lên "final" bằng
    `ResultsLogger.mark_final(...)` sau khi đủ epoch × seed theo
    `configs/config.yaml (train.epochs, train.num_seeds)`.

---

Xem `outputs/tables/MANIFEST.md` để biết mapping schema JSON chi tiết, và
`PROJECT_STATE.md` mục "Quy trình lấy số liệu tạm" để biết luồng thao tác
từng bước (train → eval checkpoint sớm → compile_paper_data → điền bản thảo
→ lặp lại).
