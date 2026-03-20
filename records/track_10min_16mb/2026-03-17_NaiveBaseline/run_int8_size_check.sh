#!/usr/bin/env bash
set -euo pipefail

cd /Users/scottpenberthy/work/parameter-golf
source .venv/bin/activate

SCRIPT=records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt_mlx.py
LOGDIR=logs

# Quick size check: train 10 iters just to get weights, then serialize with int8+zlib
run_size_check() {
    local layers="$1"
    local run_id="size_check_L${layers}"

    echo "--- L=$layers ---"

    env \
        USE_FACTOR_MLP=1 \
        MLP_LOW_RANK_R=64 \
        USE_FACTOR_ATTN=1 \
        ATTN_K_RANK=16 \
        ATTN_V_RANK=16 \
        ATTN_PROJ_RANK=32 \
        NUM_LAYERS="$layers" \
        ITERATIONS=10 \
        TRAIN_BATCH_TOKENS=8192 \
        VAL_LOSS_EVERY=0 \
        VAL_BATCH_SIZE=8192 \
        MAX_WALLCLOCK_SECONDS=0 \
        WARMUP_STEPS=0 \
        SKIP_FINAL_EVAL=0 \
        RUN_ID="$run_id" \
        python3 "$SCRIPT" 2>&1 | grep -E "^(model_params|serialized_model_int8|saved_model)" || true

    echo ""
}

echo "=== Int8+zlib size estimates for depth sweep configs ==="
echo ""

for L in 9 12 15 18 22; do
    run_size_check "$L"
done
