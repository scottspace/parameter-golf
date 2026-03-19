# Factorized MLP Residual Mode Sweep

    ## Baseline (completed)
    | Run | Mode | Rank | Resid Rank | Train Loss | Val Loss | Val BPB | Params | Int8+zlib |
    |-----|------|------|-----------|------------|----------|---------|--------|-----------|
    | mlx_smoke_factor | dense | 32 | — | 4.3067 | 4.4288 | 2.6230 | 17,944,648 | 12.41 MB |
    | mlx_smoke_low_rank | low_rank | 32 | 8 | 4.3905 | 4.4502 | 2.6357 | 8,728,648 | 6.15 MB |

    ## Sweep: MLP_RESIDUAL_RANK over {4, 8, 16, 32}

    All runs use:
    - `USE_FACTOR_MLP=1 MLP_LOW_RANK_R=32 MLP_RESIDUAL_MODE=low_rank`
    - `ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192`

    ```bash
    for R in 4 8 16 32; do
      RUN_ID=mlx_sweep_resid_r${R} \
      USE_FACTOR_MLP=1 \
      MLP_LOW_RANK_R=32 \
      MLP_RESIDUAL_MODE=low_rank \
      MLP_RESIDUAL_RANK=$R \
      ITERATIONS=200 \
      TRAIN_BATCH_TOKENS=8192 \
      VAL_LOSS_EVERY=0 \
      VAL_BATCH_SIZE=8192 \
      python3 records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt_mlx.py
    done

    Sweep: MLP_LOW_RANK_R over {8, 16, 32, 64} (with residual_rank=8)

    for R in 8 16 32 64; do
      RUN_ID=mlx_sweep_rank_r${R} \
      USE_FACTOR_MLP=1 \
      MLP_LOW_RANK_R=$R \
      MLP_RESIDUAL_MODE=low_rank \
      MLP_RESIDUAL_RANK=8 \
      ITERATIONS=200 \
      TRAIN_BATCH_TOKENS=8192 \
      VAL_LOSS_EVERY=0 \
      VAL_BATCH_SIZE=8192 \
      python3 records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt_mlx.py
    done

    What to look for

    - Train loss convergence: does low_rank residual match dense at 200 steps?
    - Param count vs quality tradeoff: low_rank r=8 uses ~1.5K residual params vs 8K dense per layer
    - Int8+zlib size: the competition metric — smaller is better at equal BPB
    - Sweet spot: best BPB per compressed byte

    Ask the main agent to write this to `sweep.md` when it's done with the current smoke test.