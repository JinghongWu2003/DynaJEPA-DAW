#!/bin/bash

set -e

echo "Launching convolutional autoencoder training on STL-10..."
# source .venv/bin/activate  # Uncomment if using a virtual environment

python -m src.train_autoencoder \
  --batch-size 256 \
  --epochs 30 \
  --model-dim 256

