#!/bin/bash
# Run the SOTA recipe (no auxiliary losses) as baseline
# Use this to establish the baseline BPB on our hardware

cd "$(dirname "$0")/.."
source .venv/bin/activate

SEED=${1:-42}

RUN_ID="sota_baseline_seed${SEED}" \
SEED=$SEED \
ITERATIONS=500 \
TRAIN_BATCH_TOKENS=65536 \
VAL_LOSS_EVERY=100 \
MAX_WALLCLOCK_SECONDS=300 \
python3 train_gpt_sota.py
