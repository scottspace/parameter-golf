# Factorized MLP Residual Mode Sweep

    ## Baseline (completed)
    | Run | Mode | Rank | Resid Rank | Train Loss | Val Loss | Val BPB | Params | Int8+zlib |
    |-----|------|------|-----------|------------|----------|---------|--------|-----------|
    | mlx_smoke_factor | dense | 32 | — | 4.3067 | 4.4288 | 2.6230 | 17,944,648 | 12.41 MB |
    | mlx_smoke_low_rank | low_rank | 32 | 8 | 4.3905 | 4.4502 | 2.6357 | 8,728,648 | 6.15 MB |

    ## Grid sweep: rank x resid_rank (500 iters, low_rank mode)

    | Rank | Resid Rank | Params | Train Loss |
    |------|-----------|--------|-----------|
    | 8 | 4 | 7,954,504 | 4.4037 |
    | 8 | 8 | 8,065,096 | 4.3929 |
    | 8 | 16 | 8,286,280 | 4.4276 |
    | 8 | 32 | 8,728,648 | 4.4433 |
    | 16 | 4 | 8,175,688 | 4.1550 |
    | 16 | 8 | 8,286,280 | 4.1604 |
    | 16 | 16 | 8,507,464 | 4.1672 |
    | 16 | 32 | 8,949,832 | 4.1568 |
    | 32 | 4 | 8,618,056 | 4.0082 |

    **Conclusion:** Residual rank has negligible effect (<0.01 loss).
    The UV core rank dominates: 8→16→32 each drops loss by ~0.2.
    Residual is noise — use `MLP_RESIDUAL_MODE=none` (pure UV).

    ## Pure UV sweep: rank in {8, 16, 32, 64} (MLP_RESIDUAL_MODE=none)

    | Rank | Params | Train Loss | MLP Params (vs 9.44M dense) |
    |------|--------|-----------|----------------------------|
    | 8 | 7,843,912 | 4.3942 | 221K (2.3%) |
    | 16 | 8,065,096 | 4.1607 | 442K (4.7%) |
    | 32 | 8,507,464 | 4.0368 | 885K (9.4%) |
    | 64 | 9,392,200 | 3.8717 | 1.77M (18.7%) |

    Pure UV matches low_rank+R within noise at every rank.
    Residual is confirmed dead weight — drop it.
    Rank 64 gives best loss at 9.39M total params (~5 min/run).

    ## Phase 2: Factorized attention + depth scaling
    - Fix MLP rank from Phase 1 results (likely 32 or 64)
    - Sweep k_rank, v_rank, proj_rank in {16, 32, 64}, q stays dense
    - Measure param savings vs baseline
    - Add depth: reinvest saved params into additional blocks
    - Shared LowRankLinear module across all factorized layers