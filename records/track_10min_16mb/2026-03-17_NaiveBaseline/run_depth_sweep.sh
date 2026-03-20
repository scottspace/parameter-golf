#!/usr/bin/env bash
set -euo pipefail

cd /Users/scottpenberthy/work/parameter-golf
source .venv/bin/activate

SCRIPT=records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt_mlx.py
RESULTS=records/track_10min_16mb/2026-03-17_NaiveBaseline/depth_sweep_results.jsonl
LOGDIR=logs

run_one() {
    local layers="$1"
    local run_id="depth_L${layers}_r64_kv16_proj32"

    if [ -f "$RESULTS" ] && grep -q "\"run_id\":\"${run_id}\"" "$RESULTS" 2>/dev/null; then
        echo "SKIP $run_id (already completed)"
        return 0
    fi

    echo "START $run_id  layers=$layers"
    local t0=$(date +%s)

    env \
        USE_FACTOR_MLP=1 \
        MLP_LOW_RANK_R=64 \
        USE_FACTOR_ATTN=1 \
        ATTN_K_RANK=16 \
        ATTN_V_RANK=16 \
        ATTN_PROJ_RANK=32 \
        NUM_LAYERS="$layers" \
        ITERATIONS=500 \
        TRAIN_BATCH_TOKENS=8192 \
        VAL_LOSS_EVERY=0 \
        VAL_BATCH_SIZE=8192 \
        MAX_WALLCLOCK_SECONDS=0 \
        WARMUP_STEPS=0 \
        SKIP_FINAL_EVAL=1 \
        RUN_ID="$run_id" \
        python3 "$SCRIPT" || {
            echo "FAILED $run_id"
            return 1
        }

    local t1=$(date +%s)
    local logfile="$LOGDIR/${run_id}.txt"
    local elapsed=$((t1 - t0))

    local train_loss=$(grep "^step:500/500" "$logfile" | sed 's/.*train_loss:\([^ ]*\).*/\1/')
    local params=$(grep "^model_params:" "$logfile" | head -1 | sed 's/model_params:\([0-9]*\).*/\1/')
    local step_avg=$(grep "^step:500/500" "$logfile" | sed 's/.*step_avg:\([^ ]*\)ms.*/\1/')

    echo "{\"run_id\":\"${run_id}\",\"layers\":${layers},\"train_loss\":${train_loss:-null},\"params\":${params:-null},\"step_avg_ms\":${step_avg:-null},\"elapsed_s\":${elapsed}}" >> "$RESULTS"
    echo "DONE $run_id  layers=$layers params=${params:-?} train_loss=${train_loss:-?} step_avg=${step_avg:-?}ms (${elapsed}s)"
    echo ""
}

echo "=== Depth sweep: NUM_LAYERS in {9, 12, 15, 18, 22} ==="
echo "  MLP_LOW_RANK_R=64, ATTN_K_RANK=16, ATTN_V_RANK=16, ATTN_PROJ_RANK=32"
echo "  500 iters, SKIP_FINAL_EVAL=1 (train loss comparison only)"
echo ""

for L in 9 12 15 18 22; do
    run_one "$L" || true
done

echo ""
echo "=== Depth sweep complete ==="
echo "Results in: $RESULTS"
echo ""
if [ -f "$RESULTS" ]; then
    printf "%-40s %10s %8s %10s\n" "Run" "Params" "Loss" "ms/step"
    printf "%s\n" "----------------------------------------------------------------------"
    while IFS= read -r line; do
        run_id=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['run_id'])")
        layers=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['layers'])")
        params=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('params','?'))")
        loss=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('train_loss','?'))")
        ms=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('step_avg_ms','?'))")
        printf "%-40s %10s %8s %10s\n" "$run_id" "$params" "$loss" "$ms"
    done < "$RESULTS"
fi
