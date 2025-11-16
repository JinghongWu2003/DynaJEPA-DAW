# DynaJEPA-STL10

DynaJEPA-STL10 provides a compact self-supervised learning setup for the STL-10 dataset. It features a simplified Joint-Embedding Predictive Architecture (JEPA) with Difficulty-Aware Weighting (DAW) and a curriculum schedule, alongside a convolutional autoencoder baseline and a linear probing utility.

## Features
- PyTorch 2.x implementation compatible with Python 3.12+
- Automatic STL-10 download to `./data`
- DAW and curriculum-based weighting for JEPA training
- Baseline convolutional autoencoder
- Linear probe evaluation script
- Turnkey shell scripts and a Colab notebook

## Quickstart

### Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Train JEPA on STL-10 (unlabeled split)
```bash
bash scripts/run_jepa_stl10.sh
```

### Train the convolutional autoencoder
```bash
bash scripts/run_autoencoder_stl10.sh
```

### Linear probe evaluation
After training, evaluate a frozen encoder with a linear classifier:
```bash
python -m src.eval_linear_probe --checkpoint ./checkpoints/jepa_latest.pt --model jepa
# or for the autoencoder encoder
python -m src.eval_linear_probe --checkpoint ./checkpoints/autoencoder_latest.pt --model autoencoder
```

### Colab notebook
Open `notebooks/colab_jepa_stl10.ipynb` in Google Colab, run all cells top-to-bottom, and follow the prompts. It installs dependencies, runs a short JEPA training loop, and performs a quick linear probe.

## Notes
- STL-10 is downloaded automatically into `./data`.
- Checkpoints are stored under `./checkpoints`; embeddings and logs go to `./outputs` by default.
- The code is intentionally compact and easy to modify for experiments.

