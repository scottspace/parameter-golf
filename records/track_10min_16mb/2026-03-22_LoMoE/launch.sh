#!/usr/bin/env bash
if [ "$1" = "kill" ]; then
    pkill -f "train_gpt.py" && echo "Killed." || echo "Nothing running."
    exit 0
fi

RUN_ID=${1:-lomoe_v1}
LOGFILE=/workspace/parameter-golf/logs/${RUN_ID}.out

nohup bash /workspace/parameter-golf/records/track_10min_16mb/2026-03-22_LoMoE/run.sh "$RUN_ID" > "$LOGFILE" 2>&1 &
echo "PID: $! — logging to $LOGFILE"
echo "tail -f $LOGFILE"
