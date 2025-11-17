#!/bin/bash
set -e

echo "Running JEPA pretraining with EMA DAW on STL-10"
python -m src.training.train_jepa_daw \
  --mode daw_ema \
  --batch-size 256 \
  --epochs 5 \
  --lr 3e-4 \
  --ema-momentum 0.99 \
  --daw-alpha 0.9 \
  --gamma 1.0 \
  --w-min 0.5 \
  --w-max 2.0 \
  --image-size 96
