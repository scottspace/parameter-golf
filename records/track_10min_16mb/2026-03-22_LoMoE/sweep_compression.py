#!/usr/bin/env python3
"""Sweep quantization/compression on a saved model. No retraining."""
import io, os, sys, zlib
os.chdir("/workspace/parameter-golf")
import torch
sys.path.insert(0, os.path.dirname(__file__))
from train_gpt import (quantize_state_dict_int8, dequantize_state_dict_int8,
    GPT, Hyperparameters, eval_val, load_validation_tokens, build_sentencepiece_luts,
    CastedLinear, LowRankLinear, restore_low_dim_params_to_fp32, MoEMLP)
import sentencepiece as spm
try:
    import pyzstd; HAS_ZSTD = True
except ImportError:
    os.system("pip install -q pyzstd")
    try:
        import pyzstd; HAS_ZSTD = True
    except ImportError:
        HAS_ZSTD = False

def compress(data, method, level):
    if method == "zstd" and HAS_ZSTD:
        return pyzstd.compress(data, level)
    return zlib.compress(data, min(level, 9))

def load_config(model_path):
    """Load config JSON saved alongside the model (same prefix, _config.json)."""
    import json as _json
    config_path = model_path.replace("_model.pt", "_config.json")
    if not os.path.exists(config_path):
        # Try legacy path: final_model.pt -> look for any *_config.json in logs/
        d = os.path.dirname(model_path) or "."
        candidates = [f for f in os.listdir(d) if f.endswith("_config.json")]
        if candidates:
            config_path = os.path.join(d, sorted(candidates)[-1])
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            config = _json.load(f)
        for k, v in config.items():
            if k.startswith("_") or k == "n_params": continue
            os.environ.setdefault(k.upper(), str(v))
    else:
        print(f"WARNING: no config found, using env vars / defaults")

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_eval = "--eval" in sys.argv
    if not model_path or model_path.startswith("--"):
        print("Usage: python sweep_compression.py <model.pt> [--eval]"); return

    load_config(model_path)
    print(f"Loading {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    combos = []
    for bits in [6, 7, 8]:
        for method, level in [("zlib", 9), ("zstd", 19), ("zstd", 22)]:
            if method == "zstd" and not HAS_ZSTD: continue
            combos.append((bits, method, level))

    results = []
    for bits, method, level in combos:
        qobj, stats = quantize_state_dict_int8(state_dict, bits=bits)
        buf = io.BytesIO(); torch.save(qobj, buf); raw = buf.getvalue()
        blob = compress(raw, method, level)
        fits = "YES" if len(blob) <= 16_000_000 else "NO"
        results.append((bits, method, level, len(blob), len(blob)/1e6, fits, qobj))

    print(f"\n{'bits':>4} {'compress':>10} {'bytes':>12} {'MB':>8} {'<16MB':>6}")
    print("-" * 46)
    for bits, method, level, sz, mb, fits, _ in results:
        print(f"{bits:>4} {method}-{level:>2}   {sz:>12,} {mb:>8.2f} {fits:>6}")

    if not run_eval: return
    print("\nRoundtrip eval on combos that fit under 16MB...")
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        use_factor_mlp=args.use_factor_mlp, mlp_low_rank_r=args.mlp_low_rank_r,
        use_factor_attn=args.use_factor_attn, attn_k_rank=args.attn_k_rank,
        attn_v_rank=args.attn_v_rank, attn_proj_rank=args.attn_proj_rank,
        use_bigram_hash=args.use_bigram_hash, bigram_hash_buckets=args.bigram_hash_buckets,
        use_smear_gate=args.use_smear_gate, use_ortho_init=False, use_swiglu=args.use_swiglu,
        use_mup=args.use_mup, mup_base_dim=args.mup_base_dim,
        use_moe=args.use_moe, moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k, moe_aux_loss_coeff=args.moe_aux_loss_coeff,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, (CastedLinear, LowRankLinear)): m.float()
    restore_low_dim_params_to_fp32(model)

    print(f"\n{'bits':>4} {'compress':>10} {'val_loss':>10} {'val_bpb':>10} {'MB':>8}")
    print("-" * 50)
    for bits, method, level, sz, mb, fits, qobj in results:
        if fits != "YES": continue
        model.load_state_dict(dequantize_state_dict_int8(qobj), strict=True)
        vl, vb = eval_val(args, model, 0, 1, device, 1, val_tokens, *luts)
        print(f"{bits:>4} {method}-{level:>2}   {vl:>10.4f} {vb:>10.4f} {mb:>8.2f}")

if __name__ == "__main__":
    main()
