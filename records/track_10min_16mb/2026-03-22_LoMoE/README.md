# LoMoE — Low-Rank Mixture of Experts

## Score: val_bpb = TBD

8xH100, 600s training. ~16MB artifact.

## Method

LoMoE replaces each transformer layer's MLP with a mixture of 8
low-rank experts (top-2 routing). Expert weights are parameterized
as W=UV (rank-64), reducing per-expert storage. At submission time,
U and V are quantized to int6 per-row and compressed with zstd-22.
This stacks two orthogonal compression axes: rank reduction and
quantization.

Attention uses dense Q with factorized K (rank-16), V (rank-16),
and output projection (rank-32).

Sort-based sparse dispatch groups tokens by expert assignment and
runs each expert on its contiguous slice.

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
| weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| warmdown_iters | 3000 |
| swa_start_frac | 0.5 |
| quant | int6 per-row + zstd-22 |

## Training

Muon optimizer for matrix params, AdamW for embeddings/scalars.
Momentum warmup 0.92->0.99 over 1500 steps. SWA every 50 steps
over the last 50% of training.
