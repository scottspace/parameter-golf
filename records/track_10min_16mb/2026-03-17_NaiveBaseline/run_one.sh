#!/usr/bin/env bash
set -euo pipefail

cd /Users/scottpenberthy/work/parameter-golf
source .venv/bin/activate

SCRIPT=records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt_mlx.py
LOGDIR=logs

# Best config: MLP r64, attn kv=16 proj=32, pure UV, sigma quantization
RUN_ID="best_sigma_r64_kv16_proj32"
echo "=== Running best config with sigma quantization ==="
echo "  MLP_LOW_RANK_R=64, ATTN_K_RANK=16, ATTN_V_RANK=16, ATTN_PROJ_RANK=32"
echo "  QUANT_MODE=sigma, 500 iterations"

env \
    USE_FACTOR_MLP=1 \
    MLP_LOW_RANK_R=64 \
    USE_FACTOR_ATTN=1 \
    ATTN_K_RANK=16 \
    ATTN_V_RANK=16 \
    ATTN_PROJ_RANK=32 \
    QUANT_MODE=sigma \
    ITERATIONS=500 \
    TRAIN_BATCH_TOKENS=8192 \
    VAL_LOSS_EVERY=0 \
    VAL_BATCH_SIZE=8192 \
    MAX_WALLCLOCK_SECONDS=0 \
    WARMUP_STEPS=0 \
    SKIP_FINAL_EVAL=0 \
    RUN_ID="$RUN_ID" \
    python3 "$SCRIPT"

echo ""
echo "=== Run complete ==="
LOGFILE="$LOGDIR/${RUN_ID}.txt"
echo "Log: $LOGFILE"
echo ""
echo "--- Key metrics ---"
grep "^model_params:" "$LOGFILE" | head -1
grep "^step:500/500" "$LOGFILE" | head -1
grep "serialized_model_sigma" "$LOGFILE"
grep "final_sigma_zlib_roundtrip " "$LOGFILE"
