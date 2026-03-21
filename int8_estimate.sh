#!/usr/bin/bash
for L in 9 12 15 18 22; do
    echo "=== L=$L ==="
    USE_FACTOR_MLP=1 MLP_LOW_RANK_R=64 \
    USE_FACTOR_ATTN=1 ATTN_K_RANK=16 ATTN_V_RANK=16 ATTN_PROJ_RANK=32 \
    NUM_LAYERS=$L ITERATIONS=10 SKIP_FINAL_EVAL=0 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    RUN_ID="size_L${L}" \
    python3 records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py 2>&1 | grep -E "model_params|factorized"
    echo ""
  done
