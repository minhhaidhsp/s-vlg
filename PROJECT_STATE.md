# PROJECT_STATE.md — Nguồn sự thật duy nhất về trạng thái dự án S-VLG

> **Đọc file này ĐẦU TIÊN** trước khi làm bất cứ việc gì trong dự án — dù bạn
> là người hay là AI (Claude trong phiên chat mới, hoặc Claude Code). Xem mục
> "HƯỚNG DẪN CHO PHIÊN LÀM VIỆC MỚI" ở cuối file trước khi bắt đầu, và cập
> nhật lại mục 2 + mục 8 (changelog) trước khi kết thúc phiên.

Ngày tạo file: **2026-07-04**.

---

## 1. Tổng quan dự án

Mục tiêu: xây dựng một mô hình Medical Visual Question Answering (VQA), phát
triển thành **HAI bài báo khoa học** từ **chung một codebase**.

### Version 1 — S-VLG (Structured Vision-Language-Graph)

- **Ba mô thức**: ảnh (X-quang/CT/MRI) + văn bản (câu hỏi/câu trả lời) + đồ
  thị tri thức bệnh nhân/EHR (ICD, CPT, tương đồng bệnh nhân).
- **Đóng góp chính**:
  1. RPR-CoAttention — đồng chú ý nhận thức vị trí tương đối 2D giữa
     câu hỏi và patch ảnh.
  2. Đồ thị tri thức bệnh nhân + Graph-RAG — truy hồi ca bệnh tương đồng
     làm bằng chứng cho câu trả lời.
  3. Hợp nhất biến phân tách biệt (disentangled fusion) + kiểm soát bất định
     (uncertainty gate) trong sinh câu trả lời.
- **Dữ liệu**: MIMIC-CXR-VQA + MIMIC-IV + MIMIC-CXR-JPG (credentialed, xem
  mục 5).
- **Đích nhắm**: IEEE Access / tạp chí SCIE Q1.

### Version 2 — SU-MedVQA (Spatially-aware Uncertainty-controlled MedVQA)

- **Hai mô thức**: ảnh + văn bản (không có đồ thị/EHR).
- **Đóng góp chính**: RPR-CoAttention (trục chính) + sinh ngôn ngữ có kiểm
  soát bất định (đóng góp phụ).
- **Dữ liệu**: VQA-RAD + SLAKE (công khai, không cần credential).
- **Đích nhắm**: hội nghị chuyên ngành hoặc tạp chí Scopus Q2/Q3.

---

## 2. Trạng thái hiện tại (cập nhật lần cuối: 2026-07-04)

| | Trạng thái | Chi tiết |
|---|---|---|
| **Version 1 (S-VLG)** | **PENDING** tại ranh giới Giai đoạn 5/6 | Đã xong Giai đoạn 1-5 (khung dự án, framework tiền xử lý, toàn bộ module mô hình, Graph-RAG, ResultsLogger + tách version). **ĐANG CHỜ**: phê duyệt CITI trên PhysioNet để được cấp quyền tải MIMIC-IV/MIMIC-CXR-JPG thật. Khi có dữ liệu thật, tiếp tục Giai đoạn 6 (tiền xử lý MIMIC thật, dựng cohort, xây patient knowledge graph từ dữ liệu thật) rồi Giai đoạn 7-8. |
| **Version 2 (SU-MedVQA)** | **ĐANG LÀM** | Chạy được ngay vì dùng dữ liệu công khai VQA-RAD/SLAKE (đã tải xong tại `data/raw/vqa-rad/`, `data/raw/slake/`). Đã xong Giai đoạn 1-3, 5 (module lõi dùng chung + `su_medvqa.py` + hạ tầng kết quả). Kế hoạch: chạy trọn Giai đoạn 6-7-8 (không có Giai đoạn 4 — V2 không dùng Graph-RAG). |

---

## 3. Lộ trình 8 giai đoạn

| Giai đoạn | Nội dung | V1 (S-VLG) | V2 (SU-MedVQA) |
|---|---|---|---|
| 1 | Khung dự án + framework tiền xử lý (cấu trúc thư mục, `.gitignore`, `requirements.txt`, `config.yaml`, tải bộ dữ liệu eval công khai VQA-RAD/SLAKE) | ✅ Done | ✅ Done (dùng chung) |
| 2 | Toàn bộ module mô hình lõi (vision_encoder, rpr_coattention, mltm, graph_sage, disentangled_fusion, decoder) | ✅ Done | ✅ Done (dùng chung phần áp dụng được) |
| 3 | Ráp mô hình end-to-end (`svlg.py` / `su_medvqa.py`) | ✅ Done | ✅ Done |
| 4 | Graph-RAG (patient_graph, retrieval, linearize) | ✅ Done | ➖ N/A (V2 không dùng đồ thị) |
| 5 | Hạ tầng kết quả (ResultsLogger có tách version, MANIFEST.md, build_paper_tables.py) | ✅ Done | ✅ Done (dùng chung) |
| 6 | Tiền xử lý dữ liệu THẬT (V1: dựng cohort MIMIC thật; V2: tiền xử lý/split VQA-RAD+SLAKE cho huấn luyện) | ⏳ Pending (chờ CITI) | ⏳ Pending (sẵn sàng bắt đầu ngay) |
| 7 | Huấn luyện + chọn siêu tham số qua validation (tau, M, gamma, LoRA, v.v.) | ⏳ Pending | ⏳ Pending |
| 8 | Đánh giá toàn diện + điền số liệu bảng/hình + viết bản thảo | ⏳ Pending | ⏳ Pending |

---

## 4. Cấu trúc code: dùng chung vs riêng

### Module lõi dùng chung (PHẢI linh hoạt, KHÔNG hardcode cho một phiên bản)

| Module | Vai trò | Ghi chú linh hoạt |
|---|---|---|
| `src/models/vision_encoder.py` | ViT patch feature extractor (Eq. 4) | Không hardcode số mô thức; chỉ xử lý ảnh. |
| `src/models/rpr_coattention.py` | Đồng chú ý vị trí tương đối 2D (Eq. 9-14) | Nhận `d`, `k` tổng quát; không giả định V1 hay V2. |
| `src/models/disentangled_fusion.py` | Hợp nhất biến phân tách biệt (Eq. 21-28) | Đã refactor nhận `branch_dims: list[int]` — `num_branches=1` cho V2 (chỉ vision), `num_branches=3` cho V1 (vision+tabular+graph). Self-test PASS cho cả hai. |
| `src/models/decoder.py` | Soft-prefix LLM + cổng bất định (Eq. 29-33) | Nhận `z_final_dim` tổng quát, không hardcode số nhánh nguồn gốc của `z_final`. |
| `src/utils/results_logger.py` | Ghi kết quả thực nghiệm | Nhận `experiment_version="v1"/"v2"`, ghi vào `outputs/tables/{version}/...`. |
| `src/data/vqa_dataset.py` *(kế hoạch, CHƯA tạo)* | Dataset/dataloader thống nhất cho VQA (ảnh, câu hỏi, câu trả lời, evidence) | Sẽ cần khi bước vào Giai đoạn 6-7 huấn luyện thật; hiện `src/data/load_eval_vqa.py` mới chỉ phục vụ đọc VQA-RAD/SLAKE thô. |
| `src/utils/metrics.py` *(kế hoạch, CHƯA tạo)* | Tính vqa_acc, exact_match, BLEU-4, precision/recall/F1, AUC-ROC dùng chung cho cả 2 bài báo | Sẽ ghi trực tiếp qua `ResultsLogger`, không in console rồi chép tay. |

### Riêng Version 1 (S-VLG)

- `src/models/mltm.py` — Masked Lab-Test Modeling (nhánh bảng biểu).
- `src/models/graph_sage.py` — GraphSAGE tự viết (nhánh đồ thị).
- `src/graph/patient_graph.py`, `src/graph/retrieval.py`, `src/graph/linearize.py` — Graph-RAG.
- `src/models/svlg.py` — lắp ráp 3 mô thức (`SVLG`).
- `configs/config_v1.yaml`.

### Riêng Version 2 (SU-MedVQA)

- `src/models/su_medvqa.py` — lắp ráp 2 mô thức (`SU_MedVQA`), tái sử dụng
  toàn bộ module lõi ở trên, `DisentangledFusion(branch_dims=[fusion_d_vis])`.
- `configs/config_v2.yaml`.

### Quy tắc bắt buộc

> Module lõi (mục "dùng chung") **PHẢI** nhận tham số qua constructor/config,
> **KHÔNG ĐƯỢC** hardcode số mô thức, tên phiên bản, hay giả định về việc có
> đồ thị/tabular hay không. Nếu một thay đổi cho module lõi chỉ có lợi cho
> một phiên bản và làm phiên bản kia phải sửa theo, đó là dấu hiệu code đang
> rò rỉ đặc thù phiên bản vào phần dùng chung — cần dừng lại và tách riêng.

---

## 5. Quy tắc tuân thủ dữ liệu (BẮT BUỘC — nhắc lại để phiên mới không vi phạm)

- **MIMIC** (MIMIC-IV, MIMIC-CXR-JPG, MIMIC-CXR-VQA) là dữ liệu **credentialed**.
  - **CHỈ** được tải từ **PhysioNet** sau khi đã hoàn tất khóa học và bài thi
    **CITI** ("Data or Specimens Only Research").
  - **TUYỆT ĐỐI KHÔNG** dùng bản sao lậu trên Kaggle, GitHub, hay bất kỳ
    nguồn phi chính thức nào, kể cả khi có vẻ "tiện" hơn.
  - **KHÔNG BAO GIỜ** commit dữ liệu MIMIC (thô hay đã xử lý) lên git — xem
    `.gitignore` ở gốc dự án, thư mục `data/` đã bị chặn hoàn toàn.
- **VQA-RAD** và **SLAKE** là dữ liệu **công khai** (Lau et al. 2018; Liu et al.
  2021) — tải tự do qua Hugging Face/OSF, xem `scripts/download_eval_datasets.py`.
  Đã tải xong tại `data/raw/vqa-rad/`, `data/raw/slake/`.

---

## 6. Vị trí số liệu cho bản thảo

- Toàn bộ số liệu dùng trong CẢ HAI bản thảo phải được ghi qua
  `src/utils/results_logger.ResultsLogger` — không hardcode, không chỉ in
  console.
- Ánh xạ đầy đủ bảng/hình ↔ file JSON: **`outputs/tables/MANIFEST.md`**
  (đã cập nhật phân tách rõ bảng nào thuộc V1 (`outputs/tables/v1/...`), bảng
  nào thuộc V2 (`outputs/tables/v2/...`), bảng nào dùng chung/không versioned
  vì là thống kê dữ liệu thô).
- Xem tiến độ điền số bất cứ lúc nào bằng:
  ```
  python scripts/build_paper_tables.py
  ```
  In ra markdown sẵn dán vào bản thảo, tách riêng mục "Version 1" và
  "Version 2", ô nào chưa có dữ liệu in `[…]`.

---

## 7. Việc tiếp theo ngay (next actions)

### Version 1 (S-VLG)

1. Hoàn tất khóa học + bài thi CITI trên PhysioNet (việc của người dùng,
   không phải AI) để được cấp quyền truy cập MIMIC-IV/MIMIC-CXR-JPG.
2. Sau khi có quyền: viết script tải + tiền xử lý MIMIC thật (Giai đoạn 6),
   dựng cohort thật (`window_hours`, `target_num_patients` trong
   `configs/config_v1.yaml`), xây `PatientKnowledgeGraph` từ mã ICD/CPT
   lịch sử thật (tuân thủ chặt ràng buộc chống rò rỉ đã ghi trong
   `src/graph/patient_graph.py`).
3. Chọn `jaccard_threshold_tau`, `retrieval_M`, `gamma_threshold` qua
   validation thật (hiện đang để `null` trong config).
4. Huấn luyện (Giai đoạn 7), ghi mọi số liệu qua `ResultsLogger(experiment_version="v1")`.
5. Đánh giá toàn diện + điền Bảng 6-11, Hình 7-11 (Giai đoạn 8).

### Version 2 (SU-MedVQA)

1. Viết pipeline tiền xử lý/split cho VQA-RAD + SLAKE dùng cho huấn luyện
   thật (Giai đoạn 6) — dữ liệu đã có sẵn tại `data/raw/`.
2. Viết `src/data/vqa_dataset.py` (Dataset/DataLoader thống nhất) và
   `src/utils/metrics.py` (vqa_acc, exact_match, BLEU-4, F1, AUC-ROC) nếu
   chưa có — đây là 2 module lõi dùng chung còn thiếu (xem mục 4).
3. Huấn luyện `SU_MedVQA` (Giai đoạn 7), ghi số liệu qua
   `ResultsLogger(experiment_version="v2")`.
4. Chạy ablation (RPR-CoAttention bật/tắt, uncertainty gate bật/tắt) →
   `log_ablation(...)` → Bảng 9 (V2).
5. Đánh giá + điền Bảng 6, 7, 9, 10 và Hình 8, 9, 10 (Giai đoạn 8, xem
   `MANIFEST.md` mục Version 2).

---

## 8. Lịch sử thay đổi (changelog)

> **Mọi phiên làm việc PHẢI cập nhật phần này và mục "2. Trạng thái hiện
> tại" trước khi kết thúc phiên** — kể cả khi chỉ thực hiện một thay đổi
> nhỏ. Ghi ngày (định dạng `YYYY-MM-DD`), tóm tắt ngắn gọn việc đã làm, và
> tác động đến trạng thái Giai đoạn nếu có.

- **2026-07-04**: Tạo `PROJECT_STATE.md` (tài liệu này). Refactor tách hai
  phiên bản: `DisentangledFusion` tổng quát hóa nhận `branch_dims: list`
  (thay vì `d_vis`/`d_tab` cố định), hỗ trợ `num_branches=1` (V2) và `=3`
  (V1); tạo `src/models/su_medvqa.py` (V2, tái sử dụng module lõi);
  `ResultsLogger` nhận `experiment_version` để tách `outputs/tables/v1/` và
  `outputs/tables/v2/`; tạo `configs/config_v1.yaml` + `configs/config_v2.yaml`
  (kế thừa `configs/config.yaml` qua `load_version_config()`); cập nhật
  `outputs/tables/MANIFEST.md` và `scripts/build_paper_tables.py` để phân
  tách rõ bảng/hình theo version. Toàn bộ self-test (mltm, graph_sage,
  vision_encoder, rpr_coattention, disentangled_fusion ×2 cấu hình, decoder,
  svlg, su_medvqa, results_logger, test_graph_rag) chạy lại PASS sau refactor.

---

## Hướng dẫn cho phiên làm việc mới

- **Nếu bạn là AI** (Claude trong phiên chat mới, hoặc Claude Code) hay là
  **người** mới tiếp nhận dự án này: đọc file `PROJECT_STATE.md` này **ĐẦU
  TIÊN**, trước khi đọc code hay bắt đầu bất kỳ việc gì. File này là nguồn
  sự thật duy nhất về việc gì đã xong, việc gì đang chờ, và việc gì cần làm
  tiếp theo cho từng phiên bản (V1/V2).
- Trước khi sửa module lõi dùng chung (mục 4), kiểm tra xem thay đổi có làm
  hỏng phiên bản còn lại không — chạy self-test của module đó (và của
  `svlg.py`/`su_medvqa.py`) trước khi coi là xong.
- Trước khi động vào dữ liệu, đọc lại mục 5 (quy tắc tuân thủ dữ liệu).
- **Khi kết thúc phiên làm việc** (dù bởi người hay AI): cập nhật lại mục
  "2. Trạng thái hiện tại" (nếu Giai đoạn nào chuyển trạng thái) và thêm một
  dòng mới vào "8. Lịch sử thay đổi" mô tả việc đã làm. Đừng để file này lạc
  hậu so với thực tế code — đó chính là lý do nó tồn tại.
