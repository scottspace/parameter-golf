#!/usr/bin/bash
  USE_FACTOR_MLP=1 MLP_LOW_RANK_R=64 \
  USE_FACTOR_ATTN=1 ATTN_K_RANK=16 ATTN_V_RANK=16 ATTN_PROJ_RANK=32 \
  NUM_LAYERS=15 \
  RUN_ID="contest_L15_r64" \
  python3 records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py
