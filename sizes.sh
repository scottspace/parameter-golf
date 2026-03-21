#!/usr/bin/bash
for L in 9 12 15 18 22; do
    echo "=== L=$L ==="
    grep -E "model_params|Serialized model int8" logs/size_L${L}.txt 2>/dev/null || echo "(still running or no log)"
done
