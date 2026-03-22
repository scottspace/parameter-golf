# LoMoE — Low-Rank Mixture of Experts

## Score: val_bpb = TBD

8xH100, 600s training. ~15MB artifact (int8+zstd-22).

## Method

LoMoE replaces each transformer layer's MLP with a mixture of 8
low-rank experts (top-2 routing). Expert weights are parameterized
as W=UV (rank-64), reducing per-expert storage. At submission time,
U and V are quantized to int8 per-row and compressed with zstd-22.
This stacks two orthogonal compression axes: rank reduction and
quantization.

Sparse bmm dispatch sorts tokens by expert assignment, pads each
expert's batch, and runs 4 torch.bmm calls over all experts in
parallel. Only routed tokens are computed (2/8 per token).

Attention uses dense Q with factorized K (rank-16), V (rank-16),
and output projection (rank-32).

BigramHash embedding (4096 buckets) provides token-pair context.
SmearGate blends adjacent token embeddings via a learned gate.
Orthogonal weight initialization for 2D matrices.

## Config

| Parameter | Value |
|-----------|-------|
| num_layers | 8 |
| model_dim | 512 |
| mlp_mult | 3 |
| moe_num_experts | 8 |
| moe_top_k | 2 |
| mlp_low_rank_r | 64 |
| attn_k/v_rank | 16 |
| attn_proj_rank | 32 |
| matrix_lr | 0.02 |
| weight_decay | 0.04 |
| grad_clip_norm | 0.1 |
| warmdown_iters | 3000 |
| swa_start_frac | 0.75 |
| swa_every | 100 |
| quant | int8 per-row + zstd-22 |

## Training

Muon optimizer for matrix params, AdamW for embeddings/scalars.
Momentum warmup 0.92->0.99 over 1500 steps. SWA every 100 steps
over the last 25% of training. BigramHash + SmearGate + OrthoInit
enabled. Sparse bmm MoE dispatch at ~325ms/step on 1xH100.
