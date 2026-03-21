#!/usr/bin/env bash
# =============================================================================
# Competitive run (PyTorch/CUDA): all tricks enabled
# =============================================================================
set -euo pipefail

cd /workspace/parameter-golf

SCRIPT=records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py
RUN_ID=${1:-competitive_gpu_v1}

env \
    RUN_ID=$RUN_ID \
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
    TRAIN_SEQ_LEN=2048 \
    TRAIN_BATCH_TOKENS=786432 \
    MAX_WALLCLOCK_SECONDS=600 \
    WARMUP_STEPS=20 \
    WARMDOWN_ITERS=3000 \
    TRAIN_LOG_EVERY=100 \
    \
    WEIGHT_DECAY=0.04 \
    GRAD_CLIP_NORM=0.3 \
    \
    MUON_MOMENTUM=0.99 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 \
    \
    USE_SWA=1 \
    SWA_START_FRAC=0.5 \
    SWA_EVERY=50 \
    \
    USE_BIGRAM_HASH=1 \
    BIGRAM_HASH_BUCKETS=4096 \
    \
    USE_SMEAR_GATE=1 \
    USE_ORTHO_INIT=1 \
    USE_SWIGLU=1 \
    USE_MOE=1 \
    MOE_NUM_EXPERTS=15 \
    MOE_TOP_K=2 \
    MOE_AUX_LOSS_COEFF=0.01 \
    USE_MUP=1 \
    MUP_BASE_DIM=256 \
    \
    QUANT_BITS=6 \
    USE_ZSTD=1 \
    ZSTD_LEVEL=22 \
    \
    python3 "$SCRIPT"

echo ""
echo "=== Done: $RUN_ID ==="
