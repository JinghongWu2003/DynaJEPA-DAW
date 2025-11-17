#!/bin/bash
set -e

echo "Running JEPA pretraining with instant DAW on STL-10"
python -m src.training.train_jepa_daw \
  --mode daw_instant \
  --batch-size 256 \
  --epochs 5 \
  --lr 3e-4 \
  --ema-momentum 0.99 \
  --gamma 1.0 \
  --w-min 0.5 \
  --w-max 2.0 \
  --image-size 96
