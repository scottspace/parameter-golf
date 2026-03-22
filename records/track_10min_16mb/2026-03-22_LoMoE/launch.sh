#!/usr/bin/env bash
DIR=$(cd "$(dirname "$0")" && pwd)
LOGS=/workspace/parameter-golf/logs
mkdir -p "$LOGS"

case "${1:-run}" in
    kill)  pkill -f "train_gpt" && echo "Killed." || echo "Nothing running." ;;
    log)   tail -f $LOGS/lomoe.out ;;
    sweep) python3 "$DIR/sweep_compression.py" "${2:-$LOGS/lomoe_model.pt}" --eval ;;
    save)
        cp $LOGS/lomoe.out "$DIR/train_seed1337.log" 2>/dev/null && echo "Copied training log"
        cp $LOGS/lomoe_config.json "$DIR/" 2>/dev/null && echo "Copied config"
        cp $LOGS/lomoe_model.int*.ptz "$DIR/" 2>/dev/null && echo "Copied artifact"
        python3 -c "
import re, json, os
log = open('$LOGS/lomoe.out').read()
s = json.load(open('$DIR/submission.json'))
m = re.search(r'final_.*_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)', log)
if m: s['val_loss'], s['val_bpb'] = float(m[1]), float(m[2])
m2 = re.search(r'Serialized model int.*: (\d+) bytes', log)
m3 = re.search(r'Code size: (\d+) bytes', log)
if m2: s['bytes_model'] = int(m2[1])
if m3: s['bytes_code'] = int(m3[1])
if m2 and m3: s['bytes_total'] = int(m2[1]) + int(m3[1])
m4 = re.search(r'step:(\d+)/\d+ val_loss.*stopping_early', log, re.S)
if m4: s['step_stop'] = int(m4[1])
s['date'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
json.dump(s, open('$DIR/submission.json','w'), indent=2)
print('Updated submission.json')
print(json.dumps(s, indent=2))
"
        echo "Ready to commit and push."
        ;;
    dense)
        nohup bash "$DIR/run_dense.sh" lomoe_dense > $LOGS/lomoe.out 2>&1 &
        echo "PID: $! — dense factorized, 10 min cap. bash launch.sh log to watch"
        ;;
    long)
        LOMOE_ITERATIONS=7500 LOMOE_WALLCLOCK=0 \
        nohup bash "$DIR/run.sh" lomoe > $LOGS/lomoe.out 2>&1 &
        echo "PID: $! — MoE, 7500 steps. bash launch.sh log to watch"
        ;;
    *)
        nohup bash "$DIR/run.sh" lomoe > $LOGS/lomoe.out 2>&1 &
        echo "PID: $! — MoE, 10 min cap. bash launch.sh log to watch"
        ;;
esac
