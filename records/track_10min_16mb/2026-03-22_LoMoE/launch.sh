#!/usr/bin/env bash
DIR=$(cd "$(dirname "$0")" && pwd)
LOGS=/workspace/parameter-golf/logs

case "${1:-run}" in
    kill)  pkill -f "train_gpt.py" && echo "Killed." || echo "Nothing running." ;;
    log)   tail -f $LOGS/lomoe.out ;;
    sweep)
        env NUM_LAYERS=8 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
            USE_FACTOR_MLP=1 MLP_LOW_RANK_R=64 USE_FACTOR_ATTN=1 \
            ATTN_K_RANK=16 ATTN_V_RANK=16 ATTN_PROJ_RANK=32 \
            USE_MOE=1 MOE_NUM_EXPERTS=8 MOE_TOP_K=2 MOE_AUX_LOSS_COEFF=0.01 \
            TRAIN_SEQ_LEN=2048 GRAD_ACCUM_STEPS=1 \
            python3 "$DIR/sweep_compression.py" "${2:-$LOGS/lomoe_model.pt}" --eval
        ;;
    *)
        nohup bash "$DIR/run.sh" lomoe > $LOGS/lomoe.out 2>&1 &
        echo "PID: $! — bash launch.sh log to watch"
        ;;
esac
