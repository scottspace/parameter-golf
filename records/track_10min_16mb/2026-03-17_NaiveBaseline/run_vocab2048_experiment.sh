#!/usr/bin/env bash
# =============================================================================
# Vocab 2048 experiment — streaming transcode, no retokenized shards on disk
# =============================================================================
# 1. Train a 2048-vocab SentencePiece BPE (small sample decoded in memory)
# 2. Baseline: vocab=1024 with best factorized config
# 3. Experiment: vocab=2048 streaming from 1024 shards via TranscodingTokenLoader
# =============================================================================
set -euo pipefail

cd /Users/scottpenberthy/work/parameter-golf
source .venv/bin/activate

SCRIPT=records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt_mlx.py
LOGDIR=logs
RESULTS=records/track_10min_16mb/2026-03-17_NaiveBaseline/vocab_experiment_results.jsonl

SP1024=data/tokenizers/fineweb_1024_bpe.model
SP2048=data/tokenizers/fineweb_2048_bpe.model
SRC_DATA=./data/datasets/fineweb10B_sp1024

COMMON="USE_FACTOR_MLP=1 MLP_LOW_RANK_R=64 MLP_RESIDUAL_MODE=none USE_FACTOR_ATTN=1 ATTN_K_RANK=16 ATTN_V_RANK=16 ATTN_PROJ_RANK=32 ITERATIONS=500 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=0 SKIP_FINAL_EVAL=1"

extract_result() {
    local logfile="$1" run_id="$2" vocab="$3"
    local train_loss=$(grep "^step:.*train_loss:" "$logfile" | tail -1 | sed 's/.*train_loss:\([^ ]*\).*/\1/')
    local params=$(grep "^model_params:" "$logfile" | head -1 | sed 's/model_params:\([0-9]*\).*/\1/')
    echo "{\"run_id\":\"${run_id}\",\"vocab\":${vocab},\"train_loss\":${train_loss:-null},\"params\":${params:-null}}" >> "$RESULTS"
    echo "  vocab=${vocab} train_loss=${train_loss:-?} params=${params:-?}"
}

# =============================================================================
# STEP 1: Train 2048-vocab SentencePiece BPE
# =============================================================================
echo "=== Step 1: Training 2048-vocab SentencePiece BPE ==="
if [ -f "$SP2048" ]; then
    echo "  SKIP (already exists)"
else
    python3 << 'PYEOF'
import sentencepiece as spm
import numpy as np
import glob

# Decode a sample from 1024 shards — enough for tokenizer training
sp = spm.SentencePieceProcessor(model_file="data/tokenizers/fineweb_1024_bpe.model")
shards = sorted(glob.glob("data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))[:3]
sample_file = "/tmp/sp2048_train_sample.txt"

print(f"Decoding {len(shards)} shards for tokenizer training...")
with open(sample_file, "w", encoding="utf-8") as out:
    for i, path in enumerate(shards):
        header = np.fromfile(path, dtype="<i4", count=256)
        tokens = np.fromfile(path, dtype="<u2", count=int(header[2]), offset=256*4)
        for start in range(0, len(tokens), 2000):
            text = sp.decode(tokens[start:start+2000].tolist())
            out.write(text + "\n")
        print(f"  shard {i+1}/{len(shards)} decoded")

print("Training SentencePiece BPE vocab_size=2048...")
spm.SentencePieceTrainer.train(
    input=sample_file,
    model_prefix="data/tokenizers/fineweb_2048_bpe",
    vocab_size=2048,
    model_type="bpe",
    character_coverage=0.9995,
    byte_fallback=True,
    num_threads=8,
    train_extremely_large_corpus=True,
    max_sentence_length=16384,
    shuffle_input_sentence=True,
    input_sentence_size=5000000,
)
print("Done.")
PYEOF
fi

# =============================================================================
# STEP 2: Baseline — vocab=1024
# =============================================================================
echo ""
echo "=== Step 2: Baseline (vocab=1024) ==="
RUN_ID=vocab_1024_baseline
if grep -q "$RUN_ID" "$RESULTS" 2>/dev/null; then
    echo "  SKIP (already in results)"
else
    env $COMMON \
        VOCAB_SIZE=1024 \
        DATA_PATH=$SRC_DATA \
        TOKENIZER_PATH=$SP1024 \
        RUN_ID=$RUN_ID \
        python3 "$SCRIPT"
    extract_result "$LOGDIR/${RUN_ID}.txt" "$RUN_ID" 1024
fi

# =============================================================================
# STEP 3: Experiment — vocab=2048 (streaming transcode)
# =============================================================================
echo ""
echo "=== Step 3: Experiment (vocab=2048, streaming transcode) ==="
RUN_ID=vocab_2048_transcode
if grep -q "$RUN_ID" "$RESULTS" 2>/dev/null; then
    echo "  SKIP (already in results)"
else
    env $COMMON \
        VOCAB_SIZE=2048 \
        DATA_PATH=$SRC_DATA \
        TOKENIZER_PATH=$SP2048 \
        TRANSCODE_SOURCE_TOKENIZER=$SP1024 \
        RUN_ID=$RUN_ID \
        python3 "$SCRIPT"
    extract_result "$LOGDIR/${RUN_ID}.txt" "$RUN_ID" 2048
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=== Results ==="
cat "$RESULTS" 2>/dev/null
echo ""
echo "Done."
