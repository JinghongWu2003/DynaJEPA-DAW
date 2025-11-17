#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <checkpoint_path>"
  exit 1
fi

CHECKPOINT=$1
echo "Running k-NN eval with checkpoint ${CHECKPOINT}"
python -m src.evals.knn_eval --checkpoint ${CHECKPOINT} --k 200 --batch-size 256
