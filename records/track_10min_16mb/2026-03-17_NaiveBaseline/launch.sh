#!/usr/bin/env bash
RUN_ID=${1:-competitive_gpu_v3}
ITERS=${2:-3600}
LOGFILE=/workspace/parameter-golf/logs/${RUN_ID}.out

nohup bash /workspace/parameter-golf/records/track_10min_16mb/2026-03-17_NaiveBaseline/run_competitive_gpu.sh "$RUN_ID" "$ITERS" > "$LOGFILE" 2>&1 &
echo "PID: $! — logging to $LOGFILE"
echo "tail -f $LOGFILE"
