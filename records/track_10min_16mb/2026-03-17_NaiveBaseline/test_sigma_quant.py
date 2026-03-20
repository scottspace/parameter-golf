#!/usr/bin/env python3
"""
Local test for sigma (7-level) quantization vs int8 quantization.

Tests:
  1. Pack/unpack roundtrip correctness
  2. Quantize/dequantize roundtrip on synthetic weights
  3. Quantize/dequantize roundtrip on a real model (if .npz exists)
  4. Compressed size comparison: int8+zlib vs sigma+zlib
  5. Reconstruction error (MSE, max error, per-tensor breakdown)

Usage:
    python test_sigma_quant.py                         # synthetic only
    python test_sigma_quant.py path/to/model.npz       # also test on real weights
"""
import pickle
import sys
import zlib

import numpy as np

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

# Import both quantization paths from the training script
from train_gpt_mlx import (
    _pack_nibble,
    _unpack_nibble,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    quantize_state_dict_sigma,
    dequantize_state_dict_sigma,
)


def compressed_size(obj: dict) -> int:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return len(zlib.compress(raw, level=9))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─── Test 1: nibble pack/unpack roundtrip ───────────────────────────────────
def test_pack_unpack():
    print_header("Test 1: Nibble pack/unpack roundtrip")
    for n in [1, 2, 3, 7, 15, 16, 100, 1023, 1024]:
        orig = np.random.randint(0, 7, size=n, dtype=np.uint8)
        packed = _pack_nibble(orig)
        unpacked = _unpack_nibble(packed, n)
        assert np.array_equal(orig, unpacked.astype(np.uint8)), f"FAIL n={n}"
        print(f"  n={n:5d}  packed_bytes={len(packed):5d}  ratio={len(packed)/n:.2f}  OK")
    print("  All pack/unpack tests passed.")


# ─── Test 2: synthetic weight roundtrip ─────────────────────────────────────
def make_synthetic_state(seed=42) -> dict[str, mx.array]:
    rng = np.random.RandomState(seed)
    state = {}
    # Simulate typical transformer weight shapes
    state["embed.weight"] = mx.array(rng.randn(1024, 512).astype(np.float32) * 0.02, dtype=mx.bfloat16)
    state["layer0.attn.qkv.weight"] = mx.array(rng.randn(1536, 512).astype(np.float32) * 0.04, dtype=mx.bfloat16)
    state["layer0.attn.proj.weight"] = mx.array(rng.randn(512, 512).astype(np.float32) * 0.04, dtype=mx.bfloat16)
    state["layer0.mlp.up.weight"] = mx.array(rng.randn(1024, 512).astype(np.float32) * 0.03, dtype=mx.bfloat16)
    state["layer0.mlp.down.weight"] = mx.array(rng.randn(512, 1024).astype(np.float32) * 0.03, dtype=mx.bfloat16)
    # Small tensor (should passthrough)
    state["layer0.norm.weight"] = mx.array(np.ones(512, dtype=np.float32), dtype=mx.bfloat16)
    # 1D bias
    state["layer0.mlp.bias"] = mx.array(rng.randn(1024).astype(np.float32) * 0.01, dtype=mx.bfloat16)
    # Add some outliers to one tensor
    big = rng.randn(512, 512).astype(np.float32) * 0.04
    big[10, 20] = 0.5   # ~12.5σ outlier
    big[100, 200] = -0.4  # ~10σ outlier
    state["layer0.attn.proj_with_outliers.weight"] = mx.array(big, dtype=mx.bfloat16)
    return state


def test_synthetic_roundtrip():
    print_header("Test 2: Synthetic weight roundtrip")
    state = make_synthetic_state()
    total_params = sum(int(v.size) for v in state.values())
    print(f"  Synthetic model: {total_params:,} params, {len(state)} tensors")

    # int8 path
    int8_obj, int8_stats = quantize_state_dict_int8(state)
    int8_size = compressed_size(int8_obj)
    int8_recon = dequantize_state_dict_int8(int8_obj)

    # sigma path
    sigma_obj, sigma_stats = quantize_state_dict_sigma(state)
    sigma_size = compressed_size(sigma_obj)
    sigma_recon = dequantize_state_dict_sigma(sigma_obj)

    print(f"\n  {'Method':<12} {'Compressed':>12} {'Payload':>12} {'Ratio':>8}")
    print(f"  {'-'*46}")
    baseline = int8_stats["baseline_tensor_bytes"]
    print(f"  {'baseline':<12} {baseline:>12,}")
    print(
        f"  {'int8+zlib':<12} {int8_size:>12,} {int8_stats['int8_payload_bytes']:>12,} "
        f"{baseline / max(int8_size, 1):>7.2f}x"
    )
    print(
        f"  {'sigma+zlib':<12} {sigma_size:>12,} {sigma_stats['sigma_payload_bytes']:>12,} "
        f"{baseline / max(sigma_size, 1):>7.2f}x"
    )
    print(f"\n  sigma vs int8: {sigma_size / max(int8_size, 1):.2%} of int8 size")

    # Per-tensor error
    print(f"\n  {'Tensor':<45} {'int8 MSE':>12} {'sigma MSE':>12} {'sigma MaxErr':>12}")
    print(f"  {'-'*83}")
    for name, orig_arr in state.items():
        orig = np.array(orig_arr.astype(mx.float32), dtype=np.float32)
        if name in int8_recon:
            i8 = np.array(int8_recon[name].astype(mx.float32), dtype=np.float32)
            i8_mse = mse(orig, i8)
        else:
            i8_mse = float('nan')
        if name in sigma_recon:
            sg = np.array(sigma_recon[name].astype(mx.float32), dtype=np.float32)
            sg_mse = mse(orig, sg)
            sg_max = max_abs_err(orig, sg)
        else:
            sg_mse = float('nan')
            sg_max = float('nan')
        print(f"  {name:<45} {i8_mse:>12.2e} {sg_mse:>12.2e} {sg_max:>12.2e}")


# ─── Test 3: real model roundtrip ──────────────────────────────────────────
def test_real_model(npz_path: str):
    print_header(f"Test 3: Real model roundtrip ({npz_path})")
    weights = dict(mx.load(npz_path))
    total_params = sum(int(v.size) for v in weights.values())
    print(f"  Model: {total_params:,} params, {len(weights)} tensors")

    # int8
    int8_obj, int8_stats = quantize_state_dict_int8(weights)
    int8_size = compressed_size(int8_obj)
    int8_recon = dequantize_state_dict_int8(int8_obj)

    # sigma
    sigma_obj, sigma_stats = quantize_state_dict_sigma(weights)
    sigma_size = compressed_size(sigma_obj)
    sigma_recon = dequantize_state_dict_sigma(sigma_obj)

    baseline = int8_stats["baseline_tensor_bytes"]
    print(f"\n  {'Method':<12} {'Compressed':>12} {'Payload':>12} {'Ratio vs raw':>14}")
    print(f"  {'-'*52}")
    print(f"  {'baseline':<12} {baseline:>12,}")
    print(
        f"  {'int8+zlib':<12} {int8_size:>12,} {int8_stats['int8_payload_bytes']:>12,} "
        f"{baseline / max(int8_size, 1):>13.2f}x"
    )
    print(
        f"  {'sigma+zlib':<12} {sigma_size:>12,} {sigma_stats['sigma_payload_bytes']:>12,} "
        f"{baseline / max(sigma_size, 1):>13.2f}x"
    )

    savings_mb = (int8_size - sigma_size) / (1024 * 1024)
    print(f"\n  sigma vs int8: {sigma_size / max(int8_size, 1):.2%} of int8 size ({savings_mb:+.2f} MB)")
    print(f"  int8 size:  {int8_size / (1024*1024):.2f} MB")
    print(f"  sigma size: {sigma_size / (1024*1024):.2f} MB")

    # Budget projection
    budget_mb = 16.0
    sigma_headroom = budget_mb - sigma_size / (1024 * 1024)
    int8_headroom = budget_mb - int8_size / (1024 * 1024)
    print(f"\n  16 MB budget headroom:")
    print(f"    int8:  {int8_headroom:.2f} MB remaining")
    print(f"    sigma: {sigma_headroom:.2f} MB remaining")
    if sigma_headroom > int8_headroom:
        extra_params = (sigma_headroom - int8_headroom) * 1024 * 1024 / (sigma_size / total_params)
        print(f"    sigma allows ~{extra_params:,.0f} more params at same compression ratio")

    # Aggregate error
    total_mse_int8 = 0.0
    total_mse_sigma = 0.0
    total_vals = 0
    for name in weights:
        orig = np.array(weights[name].astype(mx.float32), dtype=np.float32)
        n = orig.size
        if name in int8_recon:
            i8 = np.array(int8_recon[name].astype(mx.float32), dtype=np.float32)
            total_mse_int8 += float(np.sum((orig - i8) ** 2))
        if name in sigma_recon:
            sg = np.array(sigma_recon[name].astype(mx.float32), dtype=np.float32)
            total_mse_sigma += float(np.sum((orig - sg) ** 2))
        total_vals += n

    rmse_int8 = (total_mse_int8 / total_vals) ** 0.5
    rmse_sigma = (total_mse_sigma / total_vals) ** 0.5
    print(f"\n  Aggregate RMSE:")
    print(f"    int8:  {rmse_int8:.6e}")
    print(f"    sigma: {rmse_sigma:.6e}")
    print(f"    sigma/int8 RMSE ratio: {rmse_sigma / max(rmse_int8, 1e-30):.2f}x")

    # Pickle roundtrip test (simulate what the training script does)
    print(f"\n  Pickle roundtrip test...")
    raw = pickle.dumps(sigma_obj, protocol=pickle.HIGHEST_PROTOCOL)
    blob = zlib.compress(raw, level=9)
    reloaded = pickle.loads(zlib.decompress(blob))
    recon2 = dequantize_state_dict_sigma(reloaded)
    for name in weights:
        if name in sigma_recon and name in recon2:
            a = np.array(sigma_recon[name].astype(mx.float32), dtype=np.float32)
            b = np.array(recon2[name].astype(mx.float32), dtype=np.float32)
            assert np.array_equal(a, b), f"Pickle roundtrip mismatch on {name}"
    print("    Pickle roundtrip: OK (identical after serialize/deserialize)")


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_pack_unpack()
    test_synthetic_roundtrip()

    if len(sys.argv) > 1:
        test_real_model(sys.argv[1])
    else:
        # Try to find a .npz model file in the current directory
        import glob
        npz_files = sorted(glob.glob("*.npz")) + sorted(glob.glob("*/*.npz"))
        if npz_files:
            print(f"\n  Found model file: {npz_files[0]}")
            test_real_model(npz_files[0])
        else:
            print("\n  No .npz model file found. Pass one as argument to test on real weights.")
            print("  Usage: python test_sigma_quant.py path/to/model.npz")

    print(f"\n{'='*60}")
    print("  All tests passed!")
    print(f"{'='*60}")
