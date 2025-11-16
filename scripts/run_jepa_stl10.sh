#!/bin/bash

set -e

echo "Launching JEPA training on STL-10..."
# source .venv/bin/activate  # Uncomment if using a virtual environment

python -m src.train_jepa \
  --batch-size 256 \
  --epochs 50 \
  --model-dim 256 \
  --use-daw \
  --use-curriculum

