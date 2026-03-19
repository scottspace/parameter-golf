#!/usr/bin/env bash
set -euo pipefail

cd /Users/scottpenberthy/work/parameter-golf
source .venv/bin/activate

SCRIPT=records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt_mlx.py
RESULTS=records/track_10min_16mb/2026-03-17_NaiveBaseline/sweep_results.jsonl
LOGDIR=logs

# Common env
COMMON="USE_FACTOR_MLP=1 MLP_RESIDUAL_MODE=low_rank ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192"

run_one() {
    local run_id="$1" rank="$2" resid_rank="$3"

    # Skip if already completed
    if [ -f "$RESULTS" ] && grep -q "\"run_id\":\"${run_id}\"" "$RESULTS" 2>/dev/null; then
        echo "SKIP $run_id (already completed)"
        return 0
    fi

    echo "START $run_id  rank=$rank resid_rank=$resid_rank"
    local t0=$(date +%s)

    env $COMMON \
        RUN_ID="$run_id" \
        MLP_LOW_RANK_R="$rank" \
        MLP_RESIDUAL_RANK="$resid_rank" \
        python3 "$SCRIPT" || {
            echo "FAILED $run_id — will retry on next invocation"
            return 1
        }

    local t1=$(date +%s)
    local logfile="$LOGDIR/${run_id}.txt"

    # Extract results from log
    local train_loss=$(grep "^step:200/200 train_loss:" "$logfile" | head -1 | sed 's/.*train_loss:\([^ ]*\).*/\1/')
    local val_loss=$(grep "^step:200/200 val_loss:" "$logfile" | head -1 | sed 's/.*val_loss:\([^ ]*\).*/\1/')
    local val_bpb=$(grep "^step:200/200.*val_bpb:" "$logfile" | head -1 | sed 's/.*val_bpb:\([^ ]*\).*/\1/')
    local params=$(grep "^model_params:" "$logfile" | head -1 | sed 's/model_params:\([0-9]*\).*/\1/')
    local int8_bytes=$(grep "^serialized_model_int8_zlib:" "$logfile" | head -1 | sed 's/serialized_model_int8_zlib:\([0-9]*\).*/\1/')
    local q_val_loss=$(grep "^final_int8_zlib_roundtrip " "$logfile" | head -1 | sed 's/.*val_loss:\([^ ]*\).*/\1/')
    local q_val_bpb=$(grep "^final_int8_zlib_roundtrip " "$logfile" | head -1 | sed 's/.*val_bpb:\([^ ]*\).*/\1/')
    local elapsed=$((t1 - t0))

    echo "{\"run_id\":\"${run_id}\",\"rank\":${rank},\"resid_rank\":${resid_rank},\"train_loss\":${train_loss:-null},\"val_loss\":${val_loss:-null},\"val_bpb\":${val_bpb:-null},\"params\":${params:-null},\"int8_bytes\":${int8_bytes:-null},\"q_val_loss\":${q_val_loss:-null},\"q_val_bpb\":${q_val_bpb:-null},\"elapsed_s\":${elapsed}}" >> "$RESULTS"
    echo "DONE $run_id  val_bpb=${val_bpb:-?} int8=${int8_bytes:-?} bytes  (${elapsed}s)"
}

echo "=== Sweep 1: MLP_RESIDUAL_RANK over {4, 8, 16, 32} (MLP_LOW_RANK_R=32) ==="
for R in 4 8 16 32; do
    run_one "mlx_sweep_resid_r${R}" 32 "$R" || true
done

echo ""
echo "=== Sweep 2: MLP_LOW_RANK_R over {8, 16, 32, 64} (MLP_RESIDUAL_RANK=8) ==="
for R in 8 16 32 64; do
    run_one "mlx_sweep_rank_r${R}" "$R" 8 || true
done

echo ""
echo "=== All sweeps complete ==="
echo "Results in: $RESULTS"
cat "$RESULTS" 2>/dev/null
