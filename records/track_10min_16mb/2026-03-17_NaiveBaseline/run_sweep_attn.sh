#!/usr/bin/env bash
set -euo pipefail

cd /Users/scottpenberthy/work/parameter-golf
source .venv/bin/activate

SCRIPT=records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt_mlx.py
RESULTS=records/track_10min_16mb/2026-03-17_NaiveBaseline/sweep_attn_results.jsonl
LOGDIR=logs

# MLP fixed at pure UV rank=64 (best from Phase 1), attention factorized
COMMON="USE_FACTOR_MLP=1 MLP_LOW_RANK_R=64 MLP_RESIDUAL_MODE=none USE_FACTOR_ATTN=1 ITERATIONS=500 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=0 SKIP_FINAL_EVAL=1"

run_one() {
    local run_id="$1" k_rank="$2" v_rank="$3" proj_rank="$4"

    if [ -f "$RESULTS" ] && grep -q "\"run_id\":\"${run_id}\"" "$RESULTS" 2>/dev/null; then
        echo "SKIP $run_id (already completed)"
        return 0
    fi

    echo "START $run_id  k=$k_rank v=$v_rank proj=$proj_rank"
    local t0=$(date +%s)

    env $COMMON \
        RUN_ID="$run_id" \
        ATTN_K_RANK="$k_rank" \
        ATTN_V_RANK="$v_rank" \
        ATTN_PROJ_RANK="$proj_rank" \
        python3 "$SCRIPT" || {
            echo "FAILED $run_id — will retry on next invocation"
            return 1
        }

    local t1=$(date +%s)
    local logfile="$LOGDIR/${run_id}.txt"

    local train_loss=$(grep "^step:.*train_loss:" "$logfile" | tail -1 | sed 's/.*train_loss:\([^ ]*\).*/\1/')
    local params=$(grep "^model_params:" "$logfile" | head -1 | sed 's/model_params:\([0-9]*\).*/\1/')
    local elapsed=$((t1 - t0))

    echo "{\"run_id\":\"${run_id}\",\"k_rank\":${k_rank},\"v_rank\":${v_rank},\"proj_rank\":${proj_rank},\"train_loss\":${train_loss:-null},\"params\":${params:-null},\"elapsed_s\":${elapsed}}" >> "$RESULTS"
    echo "DONE $run_id  train_loss=${train_loss:-?} params=${params:-?}  (${elapsed}s)"
}

# Baseline: MLP factorized, attention dense
echo "=== Baseline: MLP rank=64 pure UV, attention dense ==="
env USE_FACTOR_MLP=1 MLP_LOW_RANK_R=64 MLP_RESIDUAL_MODE=none USE_FACTOR_ATTN=0 \
    ITERATIONS=500 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 \
    MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=0 SKIP_FINAL_EVAL=1 \
    RUN_ID=sweep_attn_baseline \
    python3 "$SCRIPT" || true
logfile="$LOGDIR/sweep_attn_baseline.txt"
if [ -f "$logfile" ] && ! grep -q "sweep_attn_baseline" "$RESULTS" 2>/dev/null; then
    train_loss=$(grep "^step:.*train_loss:" "$logfile" | tail -1 | sed 's/.*train_loss:\([^ ]*\).*/\1/')
    params=$(grep "^model_params:" "$logfile" | head -1 | sed 's/model_params:\([0-9]*\).*/\1/')
    echo "{\"run_id\":\"sweep_attn_baseline\",\"k_rank\":\"dense\",\"v_rank\":\"dense\",\"proj_rank\":\"dense\",\"train_loss\":${train_loss:-null},\"params\":${params:-null},\"elapsed_s\":0}" >> "$RESULTS"
    echo "DONE sweep_attn_baseline  train_loss=${train_loss:-?} params=${params:-?}"
fi

echo ""
echo "=== Grid: k/v/proj ranks tied, sweep {16, 32, 64} ==="
for R in 16 32 64; do
    run_one "sweep_attn_tied_r${R}" "$R" "$R" "$R" || true
done

echo ""
echo "=== Sweep proj_rank with k=v=32 ==="
for R in 16 32 64; do
    run_one "sweep_attn_proj_r${R}" 32 32 "$R" || true
done

echo ""
echo "=== Sweep k=v rank with proj=32 ==="
for R in 16 32 64; do
    run_one "sweep_attn_kv_r${R}" "$R" "$R" 32 || true
done

echo ""
echo "=== All sweeps complete ==="
cat "$RESULTS" 2>/dev/null
