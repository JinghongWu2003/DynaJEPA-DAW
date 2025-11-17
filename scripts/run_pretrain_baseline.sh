#!/bin/bash
set -e

echo "Running JEPA baseline pretraining on STL-10"
python -m src.training.train_jepa_daw \
  --mode baseline \
  --batch-size 256 \
  --epochs 5 \
  --lr 3e-4 \
  --ema-momentum 0.99 \
  --image-size 96
