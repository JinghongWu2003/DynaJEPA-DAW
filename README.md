# DynaJEPA-DAW

DynaJEPA-DAW implements a lightweight Joint-Embedding Predictive Architecture (JEPA) with a Difficulty-Aware Weighting (DAW) extension on the STL-10 dataset. The project explores how giving slightly more emphasis to harder examples—measured via per-sample loss—affects representation quality. Two DAW variants are included: an instant difficulty view that uses the current loss, and an EMA-based view that smooths difficulty over time.

## Features
- JEPA-style online/target encoders built on a ResNet-18 backbone.
- Dual-view STL-10 augmentations tailored for self-supervised training.
- Difficulty-Aware Weighting (instant and EMA) with adjustable bounds.
- Linear probe and k-NN evaluation utilities to assess representation quality.
- Ready-to-run bash scripts and a Colab notebook for reproducibility.

## Project Structure
```
DynaJEPA-DAW/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data/
│   ├── models/
│   ├── daw/
│   ├── training/
│   ├── evals/
│   └── utils/
├── scripts/
└── notebooks/
```

## Setup
The code targets **Python 3.12+**. Install dependencies with:
```bash
pip install -r requirements.txt
```
PyTorch and torchvision are intentionally unpinned so you can choose the right wheel (CPU or CUDA). STL-10 data downloads automatically into `./data` on first use.

## Quickstart
Run baseline pretraining:
```bash
bash scripts/run_pretrain_baseline.sh
```
Run DAW (instant difficulty):
```bash
bash scripts/run_pretrain_daw_instant.sh
```
Run DAW (EMA difficulty buffer):
```bash
bash scripts/run_pretrain_daw_ema.sh
```

## Evaluation
Linear probe (freeze encoder, train a linear head):
```bash
bash scripts/run_linear_probe.sh
```

k-NN evaluation:
```bash
bash scripts/run_knn_eval.sh
```

## Colab Notebook
A ready-to-run notebook is provided at `notebooks/colab_dynajepa_daw_stl10.ipynb`. Open it in Google Colab, follow the top cells to clone the repo, install requirements, and launch short pretraining and evaluation runs.

## Notes
- Default checkpoints are saved under `./checkpoints`.
- Logging is console-first; TensorBoard can be enabled via command-line flag.
- Seeding utilities aim for reproducibility, but minor nondeterminism may remain on GPU.
