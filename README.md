# S-VLG — Vision-Language-Graph cho Medical VQA

## Muc tieu du an

S-VLG (Structured Vision-Language-Graph) la mo hinh nghien cuu cho bai toan
Medical Visual Question Answering (Med-VQA), ket hop:

- **Vision encoder (ViT)** de trich xuat dac trung anh y te.
- **LLM (Qwen2.5-3B-Instruct)** fine-tune bang LoRA de sinh cau tra loi.
- **Graph module (GraphSAGE)** de mo hinh hoa quan he giua cac thuc the y khoa
  (chan doan, xet nghiem, thuoc...) trong khoang thoi gian benh nhan
  (mac dinh `window_hours = 48`).
- **Uncertainty estimation** de loc/canh bao cac du doan khong chac chan
  (`gamma_threshold`, chon qua validation).

Du lieu benh nhan su dung tu MIMIC (credentialed dataset).

## Cau truc thu muc

```
S-VLG/
├── data/                  # KHONG commit — xem quy tac ben duoi
│   ├── raw/               # Du lieu goc (MIMIC raw exports)
│   ├── interim/           # Du lieu trung gian sau buoc lam sach
│   └── processed/         # Du lieu da xu ly, san sang cho training
├── src/
│   ├── data/               # Script/module tien xu ly, dataset, dataloader
│   ├── models/             # Kien truc mo hinh (ViT encoder, LLM adapter, LoRA)
│   ├── graph/               # Xay dung graph, GraphSAGE, retrieval
│   ├── train/               # Vong lap huan luyen
│   ├── eval/                # Danh gia (BLEU, sacrebleu, cac metric khac)
│   └── utils/               # Tien ich chung (config loader, logging...)
├── configs/                # File cau hinh YAML
├── notebooks/               # Notebook thu nghiem, EDA
├── scripts/                 # Script chay tu dong hoa (CLI entrypoints)
└── outputs/
    ├── checkpoints/          # Model weights (KHONG commit)
    ├── logs/                 # Training logs (KHONG commit)
    ├── tables/               # Ket qua dang bang
    └── figures/              # Bieu do, hinh anh ket qua
```

## Quy tac KHONG commit du lieu (bat buoc tuan thu)

⚠️ **Du lieu MIMIC la credentialed data — TUYET DOI KHONG duoc push len GitHub
o bat ky dang nao** (raw, da xu ly, sample, checkpoint chua embed thong tin
benh nhan, v.v.).

- Toan bo thu muc `data/` bi loai tru boi `.gitignore` (chi giu `.gitkeep`
  de bao toan cau truc thu muc).
- `outputs/checkpoints/` va `outputs/logs/` cung bi loai tru vi co the chua
  thong tin ro ri tu du lieu huan luyen hoac co dung luong qua lon.
- Cac dinh dang file du lieu/anh nang (`*.csv`, `*.parquet`, `*.npz`, `*.jpg`,
  `*.png`, `*.ckpt`, `*.pt`, `*.pth`...) bi chan tuong minh trong `.gitignore`.
- Truoc moi lan `git add`/commit, kiem tra lai `git status` de dam bao khong
  vo tinh stage du lieu nhay cam.

## Luong lam viec

1. **Local (may nay, khong GPU manh):**
   - Tien xu ly du lieu MIMIC: `data/raw` → `data/interim` → `data/processed`.
   - Viet/test code trong `src/` voi du lieu mau nho (khong dua len git).
   - Cai dat cac phu thuoc khong can `bitsandbytes`
     (xem ghi chu trong `requirements.txt`).

2. **Colab / may co GPU (huan luyen):**
   - Upload/kem `data/processed` (qua Google Drive hoac phuong tien duoc
     phep, khong qua git) vao moi truong Colab.
   - Cai dat day du `requirements.txt` (bao gom `bitsandbytes` tren Linux).
   - Chay huan luyen tu `src/train/`, luu checkpoint vao `outputs/checkpoints/`
     (giu lai o Colab/Drive, khong dua ve git repo).
   - Xuat ket qua danh gia (`outputs/tables/`, `outputs/figures/`) de dong bo
     ve local phuc vu bao cao (day la cac file nho, khong chua du lieu goc,
     co the can nhac commit tuy noi dung).

## Cau hinh

Tat ca sieu tham so nam trong [configs/config.yaml](configs/config.yaml).
Cac gia tri de trong (`null`) la nhung sieu tham so can chon qua validation
(vi du `jaccard_threshold_tau`, `retrieval_M`, `gamma_threshold`).

Load config trong code Python bang [src/utils/config.py](src/utils/config.py):

```python
from src.utils.config import load_config

cfg = load_config("configs/config.yaml")
```
