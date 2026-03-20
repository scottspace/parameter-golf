#!/usr/bin/env bash
set -euo pipefail

cd /Users/scottpenberthy/work/parameter-golf
source .venv/bin/activate

SCRIPT=records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt_mlx.py
RESULTS=records/track_10min_16mb/2026-03-17_NaiveBaseline/sweep_results.jsonl
LOGDIR=logs

# Common env — pure UV (no residual), no validation
COMMON="USE_FACTOR_MLP=1 MLP_RESIDUAL_MODE=none ITERATIONS=500 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=0 SKIP_FINAL_EVAL=1"

run_one() {
    local run_id="$1" rank="$2"

    # Skip if already completed
    if [ -f "$RESULTS" ] && grep -q "\"run_id\":\"${run_id}\"" "$RESULTS" 2>/dev/null; then
        echo "SKIP $run_id (already completed)"
        return 0
    fi

    echo "START $run_id  rank=$rank"
    local t0=$(date +%s)

    env $COMMON \
        RUN_ID="$run_id" \
        MLP_LOW_RANK_R="$rank" \
        python3 "$SCRIPT" || {
            echo "FAILED $run_id — will retry on next invocation"
            return 1
        }

    local t1=$(date +%s)
    local logfile="$LOGDIR/${run_id}.txt"

    local train_loss=$(grep "^step:.*train_loss:" "$logfile" | tail -1 | sed 's/.*train_loss:\([^ ]*\).*/\1/')
    local params=$(grep "^model_params:" "$logfile" | head -1 | sed 's/model_params:\([0-9]*\).*/\1/')
    local step_avg=$(grep "^step:500/500" "$logfile" | head -1 | sed 's/.*step_avg:\([^ ]*\)ms.*/\1/')
    local elapsed=$((t1 - t0))

    echo "{\"run_id\":\"${run_id}\",\"rank\":${rank},\"train_loss\":${train_loss:-null},\"params\":${params:-null},\"step_avg_ms\":${step_avg:-null},\"elapsed_s\":${elapsed}}" >> "$RESULTS"
    echo "DONE $run_id  train_loss=${train_loss:-?} params=${params:-?} step_avg=${step_avg:-?}ms  (${elapsed}s)"
}

echo "=== Pure UV sweep: MLP_LOW_RANK_R in {8, 16, 32, 64} (no residual) ==="
for R in 8 16 32 64; do
    run_one "sweep_none_r${R}" "$R" || true
done

echo ""
echo "=== All sweeps complete ==="
echo "Results in: $RESULTS"
cat "$RESULTS" 2>/dev/null
