#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <checkpoint_path>"
  exit 1
fi

CHECKPOINT=$1
echo "Running linear probe with checkpoint ${CHECKPOINT}"
python -m src.evals.linear_probe --checkpoint ${CHECKPOINT} --epochs 5 --batch-size 256
