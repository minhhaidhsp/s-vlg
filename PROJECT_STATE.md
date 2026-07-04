# PROJECT_STATE.md — Nguồn sự thật duy nhất về trạng thái dự án S-VLG

> **Đọc file này ĐẦU TIÊN** trước khi làm bất cứ việc gì trong dự án — dù bạn
> là người hay là AI (Claude trong phiên chat mới, hoặc Claude Code). Xem mục
> "HƯỚNG DẪN CHO PHIÊN LÀM VIỆC MỚI" ở cuối file trước khi bắt đầu, và cập
> nhật lại mục 2 + mục 8 (changelog) trước khi kết thúc phiên.

Ngày tạo file: **2026-07-04**. Cập nhật lần cuối: **2026-07-04**.

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
| **Version 1 (S-VLG)** | **PENDING** tại ranh giới Giai đoạn 5/6 | Đã xong Giai đoạn 1-5 (khung dự án, framework tiền xử lý, toàn bộ module mô hình, Graph-RAG, ResultsLogger + tách version, cơ chế kết quả tạm/provisional + checkpoint mỗi epoch). **ĐANG CHỜ**: phê duyệt CITI trên PhysioNet để được cấp quyền tải MIMIC-IV/MIMIC-CXR-JPG thật. Khi có dữ liệu thật, tiếp tục Giai đoạn 6 (tiền xử lý MIMIC thật, dựng cohort, xây patient knowledge graph từ dữ liệu thật) rồi Giai đoạn 7-8. |
| **Version 2 (SU-MedVQA)** | **ĐANG LÀM** — smoke test toàn diện đã chạy PASS trên CPU local | Chạy được ngay vì dùng dữ liệu công khai VQA-RAD/SLAKE (đã tải xong tại `data/raw/vqa-rad/`, `data/raw/slake/`). Đã xong Giai đoạn 1-3, 5 (module lõi dùng chung + `su_medvqa.py` + hạ tầng kết quả, gồm cả cơ chế provisional). Đã chạy `scripts/run_smoketest_v2.py` (n=50/dataset, 2 epoch + 4 biến thể ablation, CPU/test_mode) — sinh đủ 8/8 mục Bảng 6,7,9,10,11 + Hình 8,9,10 với `status="provisional"`, xác nhận toàn bộ pipeline train→eval→ablation→risk-coverage→compile chạy thông trên máy local. Số liệu xấu (vqa_acc≈0, tiny model) — ĐÚNG NHƯ DỰ KIẾN, chưa phải kết quả thật. Kế hoạch: chạy lại với dữ liệu đầy đủ + epoch/seed thật trên Colab GPU (Giai đoạn 6-7-8 thật, không có Giai đoạn 4 — V2 không dùng Graph-RAG). |

**Mới**: đã có cơ chế lấy kết quả TẠM (không cần chờ train đủ 20 epoch × 3
seed) — xem mục 6 và `PAPER_DATA_MAP.md`. `src/train/train_loop.py` lưu
checkpoint sau MỖI epoch; `scripts/eval_checkpoint.py --checkpoint <bất kỳ>`
đánh giá và ghi kết quả với `status="provisional"` + `epochs_trained`;
`scripts/compile_paper_data.py --version {v1,v2}` gộp mọi số liệu đã có
thành một file `outputs/PAPER_DATA_{version}.md` để điền bản thảo ngay.

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
| `src/utils/results_logger.py` | Ghi kết quả thực nghiệm | Nhận `experiment_version="v1"/"v2"`, ghi vào `outputs/tables/{version}/...`; mỗi record có `status` ("provisional"/"final") + `epochs_trained`; `mark_final(...)` nâng cấp record khi đủ epoch×seed. |
| `src/train/checkpoint_utils.py`, `src/train/train_loop.py` | Lưu/khôi phục checkpoint, vòng train chung | Lưu checkpoint sau MỖI epoch (`{version}_seed{seed}_epoch{N}.pt`), hỗ trợ resume (khôi phục epoch/seed/optimizer state). Dùng chung cho `svlg.py`/`su_medvqa.py` qua tham số `compute_loss_fn`. |
| `scripts/eval_checkpoint.py` | Đánh giá MỘT checkpoint bất kỳ | Nhận `--checkpoint`, KHÔNG giả định checkpoint cuối; đọc `epochs_trained` từ metadata; ghi qua `ResultsLogger` với `status` suy ra từ `--target-epochs`. |
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
- **`PAPER_DATA_MAP.md`** (gốc dự án): kịch bản chi tiết — với TỪNG bảng/hình,
  ghi rõ trường số liệu cần có, script sinh, file JSON, và **train tối thiểu
  bao nhiêu thì con số đó mới có nghĩa** (vd: Bảng 6 cần >=1 epoch để có số
  tạm; Risk-coverage bắt buộc eval hết val+test, không được lấy tập con).
  Kèm mục "Thứ tự ưu tiên lấy số" — số nào lấy trước từ 1 lần train chính,
  số nào (ablation) phải train riêng nên để sau cùng.
- **`scripts/compile_paper_data.py --version {v1,v2}`**: đọc TẤT CẢ JSON kết
  quả của phiên bản đó, sinh **MỘT file duy nhất**
  `outputs/PAPER_DATA_{version}.md` — đây là file thực sự dùng để điền vào
  bản thảo, có tóm tắt tiến độ ở đầu và mỗi số kèm nhãn `[FINAL]` /
  `[TẠM epoch=N]` / `[THIẾU]`. Cập nhật lại bất cứ khi nào có số liệu mới —
  không sửa tay file này, luôn chạy lại script.

### Quy trình lấy số liệu tạm (provisional) — không cần chờ train xong 20 epoch × 3 seed

1. Train vài epoch (`src/train/train_loop.py` lưu checkpoint sau MỖI epoch).
2. Chạy `scripts/eval_checkpoint.py --checkpoint <checkpoint bất kỳ> --version
   {v1,v2} --kind {metrics,ablation,risk_coverage}` — script tự đọc
   `epochs_trained` từ metadata checkpoint, ghi kết quả qua `ResultsLogger`
   với `status="provisional"` (hoặc `"final"` nếu đã đạt `--target-epochs`).
3. Chạy `python scripts/compile_paper_data.py --version {v1,v2}` → sinh/ cập
   nhật `outputs/PAPER_DATA_{version}.md`.
4. Điền số vào bản thảo từ file đó (ghi rõ số nào còn `[TẠM]` trong bản nháp).
5. Khi train xong đủ epoch × seed: gọi `ResultsLogger.mark_final(...)` để
   nâng record từ provisional lên final, rồi lặp lại bước 3-4.

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

- **2026-07-04 (tiếp)**: Thêm cơ chế lấy kết quả TẠM (provisional) để không
  phải chờ train xong 20 epoch × 3 seed mới có số điền bản thảo. Tạo
  `PAPER_DATA_MAP.md` (kịch bản số liệu chi tiết cho từng bảng/hình, kèm mục
  "train tối thiểu bao nhiêu để có nghĩa" và thứ tự ưu tiên lấy số).
  `ResultsLogger` thêm `status`/`epochs_trained` vào mỗi record (mặc định
  `"provisional"`) và hàm `mark_final(...)` để nâng cấp khi đủ dữ liệu.
  Thêm `src/train/checkpoint_utils.py` + `src/train/train_loop.py` (lưu
  checkpoint sau MỖI epoch, hỗ trợ resume — tự test bằng SU_MedVQA tiny-mode,
  xác nhận resume tiếp đúng epoch/seed/optimizer). Thêm
  `scripts/eval_checkpoint.py` (đánh giá một checkpoint bất kỳ qua
  `--checkpoint`, không giả định checkpoint cuối, log `epochs_trained` thật
  từ metadata — đã test cả 3 `--kind` metrics/ablation/risk_coverage trên
  checkpoint thật, thư mục output tách biệt để không đụng `outputs/tables/`
  thật). Thêm `scripts/compile_paper_data.py --version {v1,v2}` gộp mọi JSON
  kết quả của phiên bản đó thành `outputs/PAPER_DATA_{version}.md` (tóm tắt
  tiến độ + nhãn `[FINAL]`/`[TẠM epoch=N]`/`[THIẾU]` từng ô) — đã chạy thử
  cho cả v1/v2, đúng như dự kiến gần như toàn bộ `[THIẾU]` vì chưa có
  thực nghiệm thật. Sửa `configs/config.yaml`: `model.vit_name` từ `null`
  sang `"vit_base_patch16_224"` (giá trị `null` khiến `.get(key, default)`
  trả `None` thay vì fallback — bug lộ ra khi dùng config thật qua
  `load_version_config` lần đầu, không phải khi test bằng dict tự tạo).

- **2026-07-04 (tiếp 2)**: Chạy smoke test toàn diện cho V2 (SU-MedVQA) trên
  CPU local, xác nhận toàn bộ pipeline train→eval→ablation→risk-coverage→
  compile chạy thông và sinh đúng định dạng số cho MỌI bảng/hình V2. Thêm:
  - `src/data/vqa_dataset.py`: `VQADataset`/`DataLoader` cho VQA-RAD/SLAKE
    (resize 224 + chuẩn hóa ImageNet, tokenize bằng tokenizer của decoder,
    tự chia val từ train khi bộ dữ liệu chưa có sẵn — VQA-RAD; SLAKE dùng
    split có sẵn).
  - `src/eval/metrics.py`: vqa_acc, exact_match, BLEU-1/4 (sacrebleu),
    precision/recall/F1/AUC-ROC cho câu hỏi CLOSED, phân rã theo category,
    và `risk_coverage_curve` (LƯU Ý: phải chạy trên toàn bộ val+test, không
    lấy tập con).
  - Thêm cờ ablation vào 2 module lõi dùng chung: `RPRCoAttention`/
    `RPR2DSelfAttention` nhận `use_rel_pos_bias` (tắt = chú ý chéo tiêu
    chuẩn, không vị trí tương đối) + `return_attn` (trả ma trận attention
    cho Hình 10); `DisentangledFusion` nhận `deterministic` (z=mu, bỏ L_KL,
    U=NaN "không định nghĩa"). `SU_MedVQA` expose cả hai qua constructor.
    Cổng bất định tắt được qua `gamma=float("inf")` khi gọi `generate()`
    (không cần sửa code — đúng thiết kế "cổng chỉ áp ở suy diễn").
  - `scripts/run_smoketest_v2.py`: chạy toàn chuỗi — train full model
    2 epoch, eval → Bảng 6/7 (kèm dòng riêng theo dataset cho "VQA-RAD/SLAKE
    riêng", gộp vào Bảng 6 thay vì tạo Bảng 8 riêng cho V2), Bảng 10 + Hình 8
    (risk-coverage trên toàn bộ val+test), Hình 10 (attention heatmap các
    câu hỏi Organ/Position), Bảng 11 (chi phí — đo thật trên CPU, `dataset`
    field ghi rõ caveat "cần đo lại trên GPU Colab", `gpu_mem_gb` để trống
    → `[THIẾU]` vì không đo được trên CPU); ablation 4 biến thể (full,
    no_rpr, no_gate, no_disentangle) mỗi biến thể train 1 epoch riêng →
    Bảng 9 + Hình 9. Đã chạy thật với `--n 50 --epochs 2`: **8/8 mục sinh
    đủ số** (`status="provisional"`), 0 mục thiếu — xác nhận pipeline hoạt
    động đúng end-to-end (số xấu, vqa_acc≈0, vì tiny model/2 epoch — ĐÚNG
    NHƯ DỰ KIẾN, không phải kết quả thật).
  - Cập nhật `MANIFEST.md`/`PAPER_DATA_MAP.md`: thêm Bảng 11 cho V2 (trước
    đó ghi nhầm là "V1 riêng").

- **2026-07-04 (tiếp 3)**: Sửa lỗi `vqa_acc = 0.0000` tuyệt đối trên MỌI
  nhóm (kể cả `answer_type:CLOSED`) phát hiện từ smoke test V2. **Chẩn đoán**
  (script `scripts/diagnose_metrics.py`, in 20 ví dụ thật): KHÔNG phải lỗi
  hàm chấm điểm — `src/eval/metrics.py` đã decode token→text đúng trước khi
  so khớp. Nguyên nhân thật: `decoder.generate()` dùng greedy decoding
  (`do_sample=False`) trên `tiny-gpt2` (test_mode) bị **suy biến**, luôn
  sinh y hệt `" stairs stairs stairs..."` cho MỌI câu hỏi bất kể input —
  do `tiny-gpt2` có `n_embd=2`, gần như không đủ năng lực biểu diễn để phân
  biệt ngữ cảnh, decode theo argmax luôn hội tụ về cùng 1 token. Đây đúng là
  lỗi ở bước generate như task đã dự đoán trước, không phải lỗi so khớp.
  **Sửa**:
  - `src/eval/metrics.py`: thêm `normalize_vqa_answer` (bỏ dấu câu, mạo từ
    a/an/the, gộp khoảng trắng) dùng cho `vqa_accuracy` (chuẩn VQA, khoan
    dung hơn); `exact_match` giữ nguyên strict (chỉ lowercase+strip).
    `closed_question_prf1_auc` cũng dùng `normalize_vqa_answer` để bền hơn
    với dấu câu thừa.
  - `src/models/decoder.py`: thêm `VQADecoder.predict_closed(...)` — với câu
    hỏi CLOSED (yes/no), so sánh trực tiếp log-probability token kế tiếp cho
    "yes" vs "no" (không generate tự do), tránh hoàn toàn kiểu suy biến trên.
    Đây là cách đánh giá câu hỏi nhị phân chuẩn trong VQA, không phải cách
    né lỗi.
  - `scripts/run_smoketest_v2.py`: `evaluate()` định tuyến câu hỏi CLOSED
    qua `predict_closed`, câu hỏi OPEN vẫn qua `generate()` (cổng bất định
    áp dụng đồng nhất trước khi định tuyến).
  - Chạy lại `run_smoketest_v2.py --n 50 --epochs 2`: **Bảng 7
    `answer_type:CLOSED` vqa_acc = 0.2727** (không còn 0 tuyệt đối, đúng dải
    kỳ vọng ~30-60% dù hơi thấp do model quá nhỏ/1-2 epoch); `OPEN` vẫn thấp
    (bình thường, generate() vẫn suy biến cho câu mở — không ảnh hưởng vì
    câu mở vốn khó, không kỳ vọng ~50% như nhị phân).
  - **Kiểm tra biến thể ablation**: cả 4 biến thể (full/no_rpr/no_gate/
    no_disentangle) vẫn cho đúng CÙNG một số `vqa_acc=0.190476...` và
    predictions per-sample **giống hệt bit-for-bit**. Đã xác minh RIÊNG
    (không qua training) rằng `use_rel_pos_bias`/`disentangle_deterministic`
    THẬT SỰ thay đổi `z_final` (chênh lệch tuyệt đối 2-4, không phải do cờ
    bị bỏ qua) và `U` đúng là `NaN` khi `deterministic=True`. Nguyên nhân số
    trùng nhau: **nghẽn cổ chai ở chính decoder `tiny-gpt2` (`n_embd=2`)** —
    với hidden dim quá nhỏ và chỉ 1 epoch train, khác biệt kiến trúc thượng
    nguồn không đủ mạnh để lật quyết định yes/no cuối cùng của LM. Đây là
    giới hạn cố hữu của việc dùng tiny-gpt2 cho smoke test (đúng mục đích
    thiết kế ban đầu: "chỉ test shape, không kỳ vọng chất lượng"), KHÔNG
    phải lỗi cờ ablation — không cố "sửa" thêm gì ở đây; ablation có ý nghĩa
    thống kê thật sự sẽ cần chạy trên Qwen2.5+LoRA thật (Giai đoạn 6-7-8,
    GPU Colab), như kế hoạch đã ghi.
  - Xóa checkpoint smoke test cũ, giữ lại `scripts/diagnose_metrics.py` làm
    công cụ debug (in câu hỏi/đáp án chuẩn/đáp án sinh ra/đúng-sai từ một
    checkpoint bất kỳ) cho các lần chẩn đoán sau này.

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
