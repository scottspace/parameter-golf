#!/usr/bin/env bash
DIR=$(cd "$(dirname "$0")" && pwd)
LOGS=/workspace/parameter-golf/logs

case "${1:-run}" in
    kill)  pkill -f "train_gpt.py" && echo "Killed." || echo "Nothing running." ;;
    log)   tail -f $LOGS/lomoe.out ;;
    sweep) python3 "$DIR/sweep_compression.py" "${2:-$LOGS/lomoe_model.pt}" --eval ;;
    *)
        nohup bash "$DIR/run.sh" lomoe > $LOGS/lomoe.out 2>&1 &
        echo "PID: $! — bash launch.sh log to watch"
        ;;
esac
