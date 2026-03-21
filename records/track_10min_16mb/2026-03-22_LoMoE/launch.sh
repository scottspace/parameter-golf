#!/usr/bin/env bash
DIR=$(cd "$(dirname "$0")" && pwd)

case "${1:-run}" in
    kill) pkill -f "train_gpt.py" && echo "Killed." || echo "Nothing running." ;;
    log)  tail -f /workspace/parameter-golf/logs/lomoe.out ;;
    *)
        nohup bash "$DIR/run.sh" lomoe > /workspace/parameter-golf/logs/lomoe.out 2>&1 &
        echo "PID: $! — bash launch.sh log to watch"
        ;;
esac
