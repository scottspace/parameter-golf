#!/usr/bin/env bash
DIR=$(cd "$(dirname "$0")" && pwd)
LOGS=/workspace/parameter-golf/logs

case "${1:-run}" in
    kill)  pkill -f "train_gpt.py" && echo "Killed." || echo "Nothing running." ;;
    log)   tail -f $LOGS/lomoe.out ;;
    sweep) python3 "$DIR/sweep_compression.py" "${2:-$LOGS/lomoe_model.pt}" --eval ;;
    long)
        LOMOE_ITERATIONS=7500 LOMOE_WALLCLOCK=0 \
        nohup bash "$DIR/run.sh" lomoe > $LOGS/lomoe.out 2>&1 &
        echo "PID: $! — 7500 steps, ~52 min. bash launch.sh log to watch"
        ;;
    *)
        nohup bash "$DIR/run.sh" lomoe > $LOGS/lomoe.out 2>&1 &
        echo "PID: $! — 10 min cap. bash launch.sh log to watch"
        ;;
esac
