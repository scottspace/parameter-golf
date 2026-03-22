# LoMoE — Low-Rank Mixture of Experts

## Score: val_bpb = TBD

8xH100 SXM, 600s training. Artifact under 16MB.

## Method

Each transformer layer uses 8 factorized experts (top-2 routing)
instead of a single MLP. Expert weights are parameterized as W=UV
(rank-64). At submission time, expert U/V matrices are quantized
to int6 per-row, attention weights to int8, and tied embeddings
are kept in fp16. Compressed with zstd-22.

Quantization-aware training (QAT) fake-quantizes weights during
the forward pass using a straight-through estimator. The model
learns values that fall on quantization grid points, reducing
post-training quantization penalty.

Sparse bmm dispatch sorts tokens by expert assignment and runs
4 batched matmuls over all experts. Only routed tokens are
computed (2 of 8 per token).

Attention uses dense Q with factorized K (rank-16), V (rank-16),
and output projection (rank-32). BigramHash embedding (4096
buckets) and SmearGate provide token-pair context. Orthogonal
weight initialization.

## Config

| Parameter | Value |
|-----------|-------|
| num_layers | 8 |
| model_dim | 512 |
| num_heads / num_kv_heads | 8 / 4 |
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
| qat_bits | 6 |
| quant (experts) | int6 per-row |
| quant (attention) | int8 per-row |
| quant (embeddings) | fp16 |
| compression | zstd-22 |

## Training

Muon optimizer for 2D matrix params. AdamW for embeddings and
scalars. Muon momentum warmup 0.92 to 0.99 over 1500 steps.
SWA every 100 steps over the last 25% of training. Trained at
seq_len=2048, batch=786432.
