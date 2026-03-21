#!/usr/bin/env bash
set -euo pipefail
cd /workspace/parameter-golf

# Ensure zstd compression is available
pip install -q pyzstd 2>/dev/null || true

SCRIPT=records/track_10min_16mb/2026-03-22_LoMoE/train_gpt.py
RUN_ID=${1:-factorized_moe}

env \
    RUN_ID=$RUN_ID \
    SEED=${2:-1337} \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    \
    NUM_LAYERS=8 \
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
    USE_MOE=1 \
    MOE_NUM_EXPERTS=8 \
    MOE_TOP_K=2 \
    MOE_AUX_LOSS_COEFF=0.01 \
    \
    TRAIN_SEQ_LEN=2048 \
    TRAIN_BATCH_TOKENS=196608 \
    GRAD_ACCUM_STEPS=2 \
    ITERATIONS=${LOMOE_ITERATIONS:-7500} \
    MAX_WALLCLOCK_SECONDS=${LOMOE_WALLCLOCK:-600} \
    WARMUP_STEPS=20 \
    WARMDOWN_ITERS=3000 \
    TRAIN_LOG_EVERY=100 \
    \
    WEIGHT_DECAY=0.04 \
    GRAD_CLIP_NORM=0.1 \
    MATRIX_LR=0.02 \
    MUON_MOMENTUM=0.99 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 \
    \
    USE_BIGRAM_HASH=1 \
    BIGRAM_HASH_BUCKETS=4096 \
    USE_SMEAR_GATE=1 \
    USE_ORTHO_INIT=1 \
    \
    USE_SWA=1 \
    SWA_START_FRAC=0.75 \
    SWA_EVERY=100 \
    \
    QUANT_BITS=8 \
    USE_ZSTD=1 \
    ZSTD_LEVEL=22 \
    \
    python3 "$SCRIPT"
