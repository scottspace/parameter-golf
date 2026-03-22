# LoMoE — Findings & Iteration Log

## Summary

LoMoE (Low-Rank Mixture of Experts) is a factorized MoE architecture
where each expert's weights are parameterized as W=UV. The idea: rank
reduction and quantization are orthogonal compression axes. Apply both
to fit more effective capacity into 16MB.

The architecture works. The speed doesn't — on the competition's
10-min / 8xH100 constraint. The compute-all-experts overhead (3x
slower per step than dense) means fewer training steps, which costs
more quality than the extra MoE capacity provides.

## Key Results

| Config | val_bpb | Steps | ms/step | Artifact | Hardware |
|--------|---------|-------|---------|----------|----------|
| SOTA (dense int6) | 1.145 | 7379 | 81 | 15.9MB | 8xH100 |
| Baseline (dense int8) | 1.224 | ~10000 | ~60 | 15.9MB | 8xH100 |
| Dense factorized r=64 | 1.358 | 6657 | 90 | 6.7MB | 8xH100 |
| LoMoE 8-GPU | 1.393 | 2479 | 241 | 15.3MB | 8xH100 |
| LoMoE 1-GPU (long) | 1.511 | 7500 | 331 | 15.3MB | 1xH100 |

## What We Learned

### 1. MoE compute overhead kills at short time budgets

Computing all 8 experts on all tokens (needed for torch.compile
fullgraph) is 3x slower than a single dense MLP. Sparse dispatch
(sort + bmm) saves compute but breaks torch.compile and DDP.

At 241ms/step vs 81ms/step, LoMoE gets 2479 steps while SOTA gets
7379. The MoE capacity advantage doesn't compensate for 3x fewer
gradient updates in 10 minutes.

### 2. Factorization rank must match the 16MB budget

Rank=64 produces a 6.7MB artifact — only 42% of the 16MB cap.
The model is capacity-starved. Dense models use 22M params in
int6 to fill 16MB. Factorized models need rank=192-256 to match.

The insight: W=UV isn't about being small. It's about compressing
better at the same effective capacity. High-rank factorized in
int8 can match dense in int6 at the same artifact size, with
potentially lower quantization penalty.

### 3. Quantization-Aware Training (QAT) works

QAT reduced the quantization penalty from 0.065 to 0.003 bpb
(int8). The model learns to place weights on quantization grid
points. STE (straight-through estimator) is simple to implement:
quantize forward, pass gradients through.

### 4. SWA intensity matters for quantization

75 SWA checkpoints (start_frac=0.5, every=50) produced weights
so smooth that int6 lost 0.184 bpb. Reducing to 18 checkpoints
(start_frac=0.75, every=100) cut the penalty to 0.065 bpb.
More averaging = smoother weights = harder to quantize.

### 5. FP16 tied embeddings are important

The embedding matrix serves two roles (input embed + output head).
Quantizing it damages both. Keeping it in fp16 costs ~1MB but
preserves quality. All top entries do this.

### 6. Mixed quantization: experts int6, attention int8

Expert U/V matrices are small and plentiful — they tolerate lower
precision. Attention weights encode precise geometric relationships
and need int8. This mirrors the SOTA finding (int5 MLP, int6 attn).

### 7. The early loss spike is MoE router initialization

Steps 1-10 see loss spike to 16+ (above random 6.93). The router
starts random, assigning tokens arbitrarily. Experts receive
scrambled inputs and produce garbage. Recovers by step 10-20.
Not worth fixing at 10 wasted steps out of 7500.

### 8. Batch size tradeoff on 1 GPU

Small batch (98K, grad_accum=1): 330ms/step, ~1800 steps in 10 min.
Large batch (786K, grad_accum=8): 2200ms/step, ~270 steps in 10 min.
Small batch wins on 1 GPU — more parameter updates beats cleaner
gradients at this scale.

### 9. torch.compile + DDP + MoE is fragile

- Sparse dispatch (.item() calls) breaks fullgraph=True
- dynamic=True + DDP crashes with "int has no attribute meta"
- Compute-all dispatch works with fullgraph=True + DDP
- No compile + DDP works but is slow (510ms/step)
- Best combo: compute-all + fullgraph=True (241ms/step)

### 10. Dense factorized at high rank is the sweet spot

Rank=192-256 factorized gives near-dense expressiveness (16-20M
params) while compressing to int8+zstd (estimated ~14MB). This
preserves the novel compression approach without MoE's speed
penalty. Still to be validated.

## Architecture

8-layer transformer, 512 dim, 8 heads (4 KV). Each layer:
- Dense Q, factorized K (rank-16), V (rank-16), proj (rank-32)
- MoE: 8 factorized experts (rank-64), top-2 routing
- Dense: single factorized MLP (rank-64/192)
- BigramHash (4096 buckets), SmearGate, OrthoInit

Training: Muon + AdamW, WD=0.04, grad_clip=0.1, QAT (int6 STE),
momentum warmup 0.92→0.99/1500 steps, SWA last 25%/every 100 steps.

## Next Steps

1. Dense factorized at rank=192 on 8xH100 (running now)
2. Extended LoMoE run (60 min) to find crossover vs dense baseline
3. If rank=192 artifact fits 16MB, try rank=256
4. Consider SwiGLU activation (adds params but better quality)
5. Tune LRs to match SOTA recipe (matrix_lr=0.02, embed_lr=0.03)
