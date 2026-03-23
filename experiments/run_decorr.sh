#!/bin/bash
# Test inter-layer decorrelation loss only
cd "$(dirname "$0")/.."
source .venv/bin/activate

SEED=${1:-42}
LAMBDA=${2:-0.01}

RUN_ID="decorr_lambda${LAMBDA}_seed${SEED}" \
SEED=$SEED \
ITERATIONS=500 \
TRAIN_BATCH_TOKENS=65536 \
VAL_LOSS_EVERY=100 \
MAX_WALLCLOCK_SECONDS=300 \
USE_AUX_LOSSES=1 \
USE_FOCAL_LOSS=0 \
LAMBDA_DECORR=$LAMBDA \
LAMBDA_RANK=0 \
LAMBDA_UNIGRAM=0 \
python3 train_gpt_aux.py
