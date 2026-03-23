#!/bin/bash
# Test focal loss only (no other aux losses)
cd "$(dirname "$0")/.."
source .venv/bin/activate

SEED=${1:-42}
GAMMA=${2:-2.0}

RUN_ID="focal_gamma${GAMMA}_seed${SEED}" \
SEED=$SEED \
ITERATIONS=500 \
TRAIN_BATCH_TOKENS=65536 \
VAL_LOSS_EVERY=100 \
MAX_WALLCLOCK_SECONDS=300 \
USE_AUX_LOSSES=1 \
USE_FOCAL_LOSS=1 \
FOCAL_GAMMA=$GAMMA \
LAMBDA_DECORR=0 \
LAMBDA_RANK=0 \
LAMBDA_UNIGRAM=0 \
python3 train_gpt_aux.py
