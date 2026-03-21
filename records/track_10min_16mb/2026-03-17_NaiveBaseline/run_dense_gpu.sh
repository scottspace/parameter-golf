#!/usr/bin/env bash
# Dense factorized baseline (no MoE) — single W=UV MLP per layer
set -euo pipefail
cd /workspace/parameter-golf

SCRIPT=records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py
ITERATIONS=${2:-20000}
RUN_ID=${1:-dense_factorized_v1}

env \
    RUN_ID=$RUN_ID \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    \
    NUM_LAYERS=10 \
    MODEL_DIM=512 \
    NUM_HEADS=8 \
    NUM_KV_HEADS=4 \
    MLP_MULT=3 \
    \
    USE_FACTOR_MLP=1 \
    MLP_LOW_RANK_R=64 \
    USE_FACTOR_ATTN=1 \
    ATTN_K_RANK=16 \
    ATTN_V_RANK=16 \
    ATTN_PROJ_RANK=32 \
    \
    TRAIN_SEQ_LEN=2048 \
    TRAIN_BATCH_TOKENS=786432 \
    ITERATIONS=$ITERATIONS \
    MAX_WALLCLOCK_SECONDS=600 \
    WARMUP_STEPS=20 \
    WARMDOWN_ITERS=3000 \
    TRAIN_LOG_EVERY=100 \
    \
    WEIGHT_DECAY=0.04 \
    GRAD_CLIP_NORM=0.3 \
    MUON_MOMENTUM=0.99 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 \
    \
    USE_SWA=1 \
    SWA_START_FRAC=0.5 \
    SWA_EVERY=50 \
    \
    QUANT_BITS=6 \
    USE_ZSTD=1 \
    ZSTD_LEVEL=22 \
    \
    python3 "$SCRIPT"
