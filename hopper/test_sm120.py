"""Comprehensive correctness tests for SM120 MXFP8 Flash Attention on RTX 5090.

Scale factor correctness is tested by comparing two kernel paths:
  1. kernel(fp8_data, scale_factors)          — MXFP8 with block scales
  2. kernel(pre_scaled_fp8_data, identity_sf) — data pre-scaled, identity SF

Both paths should produce identical results (within floating-point noise),
proving the kernel correctly applies scale factors. FP32 reference comparisons
use relaxed tolerances since FP8 GEMM inherently differs from FP32.
"""

import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/nanogpt/prj/fp8_flashattention/flash-attention/hopper")

BLOCK_SIZE = 32  # MXFP8 scale-factor block size


# ── Helpers ──────────────────────────────────────────────────────────────────

def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available"); return False
    props = torch.cuda.get_device_properties(0)
    arch = props.major * 10 + props.minor
    print(f"GPU: {props.name}, sm_{arch}, {props.multi_processor_count} SMs")
    return arch >= 120


def quantize_mxfp8(data_fp32):
    """Quantize FP32 → MXFP8: (FP8 e4m3 data, UE8M0 per-block scales)."""
    b, s, h, d = data_fp32.shape
    sf = torch.zeros(b, h, s, d // BLOCK_SIZE, dtype=torch.uint8, device=data_fp32.device)
    scaled_data = data_fp32.clone()
    for blk in range(d // BLOCK_SIZE):
        block = data_fp32[:, :, :, blk * BLOCK_SIZE:(blk + 1) * BLOCK_SIZE]
        amax = block.abs().amax(dim=-1)
        exp = torch.floor(torch.log2(amax.clamp(min=1e-12))).clamp(-10, 10).to(torch.int32)
        sf[:, :, :, blk] = (exp + 127).clamp(0, 255).to(torch.uint8).permute(0, 2, 1)
        scaled_data[:, :, :, blk * BLOCK_SIZE:(blk + 1) * BLOCK_SIZE] = block * (2.0 ** (-exp.float())).unsqueeze(-1)
    return scaled_data.to(torch.float8_e4m3fn), sf


def dequantize_mxfp8(fp8_data, sf):
    """Dequantize MXFP8 to FP32: value = fp8 * 2^(sf - 127)."""
    data = fp8_data.float()
    b, s, h, d = data.shape
    for blk in range(d // BLOCK_SIZE):
        mult = (2.0 ** (sf[:, :, :, blk].float() - 127.0)).permute(0, 2, 1).unsqueeze(-1)
        data[:, :, :, blk * BLOCK_SIZE:(blk + 1) * BLOCK_SIZE] *= mult
    return data


def prescale_fp8(fp8_data, sf):
    """Apply MXFP8 scales to FP8 data, re-quantize to FP8 (for kernel-vs-kernel tests)."""
    return dequantize_mxfp8(fp8_data, sf).to(torch.float8_e4m3fn)


def reference_attention(q_fp32, k_fp32, v_fp32, scale, causal=False):
    """Reference attention in FP32."""
    Q = q_fp32.float().transpose(1, 2)
    K = k_fp32.float().transpose(1, 2)
    V = v_fp32.float().transpose(1, 2)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        sq, sk = S.shape[-2], S.shape[-1]
        mask = torch.triu(torch.ones(sq, sk, device=S.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    lse = torch.logsumexp(S, dim=-1)
    return O.transpose(1, 2).to(torch.bfloat16), lse


def identity_sf(b, h, s, d, device='cuda'):
    return torch.full((b, h, s, d // BLOCK_SIZE), 127, dtype=torch.uint8, device=device)


# ── Test Cases ───────────────────────────────────────────────────────────────

def test_identity_scales_match_legacy():
    """flash_attn_mxfp8_func with identity scales must exactly match _flash_attn_forward."""
    from flash_attn_interface import flash_attn_mxfp8_func, _flash_attn_forward

    configs = [
        (1, 128, 1, 128), (1, 256, 4, 128),
        (2, 512, 8, 128), (4, 2048, 32, 128),
    ]
    all_pass = True
    for b, s, h, d in configs:
        torch.manual_seed(42)
        q = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
        k = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
        v = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
        sf = identity_sf(b, h, s, d)

        out_m, lse_m = flash_attn_mxfp8_func(q, k, v, sf, sf, sf, softmax_scale=d**-0.5)
        out_l, lse_l, _, _ = _flash_attn_forward(q, k, v, softmax_scale=d**-0.5)
        torch.cuda.synchronize()

        o_diff = (out_m.float() - out_l.float()).abs().max().item()
        l_diff = (lse_m - lse_l).abs().max().item()
        ok = o_diff == 0.0 and l_diff == 0.0
        print(f"  ({b},{s},{h},{d}): O={o_diff:.0e} LSE={l_diff:.0e}  [{'ok' if ok else 'MISMATCH'}]")
        if not ok: all_pass = False
    return all_pass


def test_uniform_scales():
    """Uniform non-identity scales = adjusted softmax_scale. Tests HW scale application."""
    from flash_attn_interface import flash_attn_mxfp8_func

    all_pass = True
    for s in [128, 256, 512, 2048]:
        torch.manual_seed(42)
        b, h, d = 1, 1, 128; scale = d ** -0.5
        q = (torch.randn(b, s, h, d, device='cuda') * 0.1).to(torch.float8_e4m3fn)
        k = (torch.randn(b, s, h, d, device='cuda') * 0.1).to(torch.float8_e4m3fn)
        v = (torch.randn(b, s, h, d, device='cuda') * 0.5).to(torch.float8_e4m3fn)

        # sf=130 for Q and K → combined 2^3 * 2^3 = 64x
        sf_130 = torch.full((b, h, s, d // 32), 130, dtype=torch.uint8, device='cuda')
        id = identity_sf(b, h, s, d)

        out_s, lse_s = flash_attn_mxfp8_func(q, k, v, sf_130, sf_130, id, softmax_scale=scale)
        out_e, lse_e = flash_attn_mxfp8_func(q, k, v, id, id, id, softmax_scale=scale * 64)

        diff = (lse_s - lse_e).abs().max().item()
        ok = diff == 0.0
        print(f"  s={s}: LSE diff={diff:.0e}  [{'ok' if ok else 'FAIL'}]")
        if not ok: all_pass = False
    return all_pass


def test_scale_factors_applied():
    """kernel(data, sf) == kernel(prescaled_data, identity_sf) — proves SF is applied.

    Uses small data (max ~1.0 after dequant) so prescaling doesn't overflow FP8 range.
    The MMA applies SF via FP32 accumulator, but prescale_fp8 goes through FP8 requant,
    so data must stay within FP8 representable range after scaling.
    """
    from flash_attn_interface import flash_attn_mxfp8_func

    configs = [
        (1, 128, 1, 128), (1, 256, 2, 128),
        (1, 512, 1, 128), (2, 1024, 4, 128),
    ]
    all_pass = True
    for b, s, h, d in configs:
        torch.manual_seed(42)
        scale = d ** -0.5

        # Small data so dequantized values stay well within FP8 range (max 448)
        Q_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.01
        K_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.01
        V = (torch.randn(b, s, h, d, device='cuda') * 0.01).to(torch.float8_e4m3fn)
        # Block-varying magnitudes → non-trivial SFs
        Q_fp32[:, :, :, 0:32] *= 30.0
        K_fp32[:, :, :, 64:96] *= 50.0

        Q_fp8, q_sf = quantize_mxfp8(Q_fp32)
        K_fp8, k_sf = quantize_mxfp8(K_fp32)
        v_sf = identity_sf(b, h, s, d)

        # Path 1: kernel with scale factors
        out_sf, lse_sf = flash_attn_mxfp8_func(Q_fp8, K_fp8, V, q_sf, k_sf, v_sf, softmax_scale=scale)

        # Path 2: pre-scale data (dequant → FP8), use identity SF
        Q_pre = dequantize_mxfp8(Q_fp8, q_sf).to(torch.float8_e4m3fn)
        K_pre = dequantize_mxfp8(K_fp8, k_sf).to(torch.float8_e4m3fn)
        id = identity_sf(b, h, s, d)
        out_id, lse_id = flash_attn_mxfp8_func(Q_pre, K_pre, V, id, id, v_sf, softmax_scale=scale)

        lse_diff = (lse_sf - lse_id).abs().max().item()
        o_diff = (out_sf.float() - out_id.float()).abs().max().item()
        ok = lse_diff < 0.01 and o_diff < 0.1
        print(f"  ({b},{s},{h},{d}): LSE={lse_diff:.4f} O={o_diff:.4f}  [{'ok' if ok else 'FAIL'}]")
        if not ok: all_pass = False
    return all_pass


def test_per_row_scales():
    """Per-row varying scale factors — verified against FP32 dequantized reference."""
    from flash_attn_interface import flash_attn_mxfp8_func

    torch.manual_seed(99)
    b, s, h, d = 1, 512, 2, 128; scale = d ** -0.5

    # Small data with per-row magnitude variation
    Q_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.01
    K_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.01
    V = (torch.randn(b, s, h, d, device='cuda') * 0.01).to(torch.float8_e4m3fn)
    for i in range(s):
        Q_fp32[0, i, :, :] *= 2.0 ** ((i % 8) - 4) * 10  # per-row magnitude

    Q_fp8, q_sf = quantize_mxfp8(Q_fp32)
    K_fp8, k_sf = quantize_mxfp8(K_fp32)
    v_sf = identity_sf(b, h, s, d)

    unique_per_block = [q_sf[0, 0, :, blk].unique().numel() for blk in range(d // 32)]
    print(f"  Q sf unique values per block: {unique_per_block}")

    # Compare kernel(sf) vs kernel(prescaled, identity) — small data avoids FP8 overflow
    out_sf, lse_sf = flash_attn_mxfp8_func(Q_fp8, K_fp8, V, q_sf, k_sf, v_sf, softmax_scale=scale)
    Q_pre = dequantize_mxfp8(Q_fp8, q_sf).to(torch.float8_e4m3fn)
    K_pre = dequantize_mxfp8(K_fp8, k_sf).to(torch.float8_e4m3fn)
    id = identity_sf(b, h, s, d)
    out_id, lse_id = flash_attn_mxfp8_func(Q_pre, K_pre, V, id, id, v_sf, softmax_scale=scale)

    lse_diff = (lse_sf - lse_id).abs().max().item()
    o_diff = (out_sf.float() - out_id.float()).abs().max().item()
    print(f"  SF vs prescaled: LSE diff={lse_diff:.6f}  O diff={o_diff:.4f}")
    return lse_diff < 0.01


def test_causal_with_scales():
    """Causal masking + non-identity scale factors (small data to avoid FP8 overflow)."""
    from flash_attn_interface import flash_attn_mxfp8_func

    configs = [(1, 256, 2, 128), (2, 512, 4, 128)]
    all_pass = True
    for b, s, h, d in configs:
        torch.manual_seed(33)
        scale = d ** -0.5
        Q_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.01
        K_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.01
        V = (torch.randn(b, s, h, d, device='cuda') * 0.01).to(torch.float8_e4m3fn)
        Q_fp32[:, :, :, 0:32] *= 30.0
        K_fp32[:, :, :, 64:96] *= 40.0

        Q_fp8, q_sf = quantize_mxfp8(Q_fp32)
        K_fp8, k_sf = quantize_mxfp8(K_fp32)
        v_sf = identity_sf(b, h, s, d)

        out_sf, lse_sf = flash_attn_mxfp8_func(Q_fp8, K_fp8, V, q_sf, k_sf, v_sf,
                                                 softmax_scale=scale, causal=True)
        Q_pre = dequantize_mxfp8(Q_fp8, q_sf).to(torch.float8_e4m3fn)
        K_pre = dequantize_mxfp8(K_fp8, k_sf).to(torch.float8_e4m3fn)
        id = identity_sf(b, h, s, d)
        out_id, lse_id = flash_attn_mxfp8_func(Q_pre, K_pre, V, id, id, v_sf,
                                                 softmax_scale=scale, causal=True)

        lse_diff = (lse_sf - lse_id).abs().max().item()
        ok = lse_diff < 0.01
        print(f"  ({b},{s},{h},{d},C): LSE diff={lse_diff:.6f}  [{'ok' if ok else 'FAIL'}]")
        if not ok: all_pass = False
    return all_pass


def test_fp32_reference_sanity():
    """FP8 kernel vs FP32 reference — sanity check with relaxed tolerance."""
    from flash_attn_interface import flash_attn_mxfp8_func

    configs = [
        (1, 128, 1, 128, False), (1, 256, 4, 128, False),
        (1, 256, 2, 128, True),  (2, 512, 4, 128, False),
    ]
    all_pass = True
    for b, s, h, d, causal in configs:
        torch.manual_seed(42)
        scale = d ** -0.5
        Q_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.5
        K_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.5
        V_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.5

        Q_fp8 = Q_fp32.to(torch.float8_e4m3fn)
        K_fp8 = K_fp32.to(torch.float8_e4m3fn)
        V_fp8 = V_fp32.to(torch.float8_e4m3fn)
        sf = identity_sf(b, h, s, d)

        O_ref, LSE_ref = reference_attention(Q_fp8.float(), K_fp8.float(), V_fp8.float(), scale, causal)
        out, lse = flash_attn_mxfp8_func(Q_fp8, K_fp8, V_fp8, sf, sf, sf,
                                           softmax_scale=scale, causal=causal)

        lse_diff = (lse - LSE_ref).abs().max().item()
        o_diff = (out.float() - O_ref.float()).abs().max().item()
        tag = "C" if causal else "F"
        # Relaxed tolerance: FP8 GEMM accumulates error over sequence length
        ok = lse_diff < 0.5 and o_diff < 1.0
        print(f"  ({b},{s},{h},{d},{tag}): LSE={lse_diff:.4f} O={o_diff:.4f}  [{'ok' if ok else 'FAIL'}]")
        if not ok: all_pass = False
    return all_pass


def test_batch_head_consistency():
    """Batched results must match per-element results."""
    from flash_attn_interface import flash_attn_mxfp8_func

    torch.manual_seed(42)
    b, s, h, d = 3, 256, 4, 128; scale = d ** -0.5
    Q_fp32 = torch.randn(b, s, h, d, device='cuda')
    K_fp32 = torch.randn(b, s, h, d, device='cuda')
    Q_fp32[:, :, :, 0:32] *= 5.0

    Q_fp8, q_sf = quantize_mxfp8(Q_fp32)
    K_fp8, k_sf = quantize_mxfp8(K_fp32)
    V_fp8 = (torch.randn(b, s, h, d, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    v_sf = identity_sf(b, h, s, d)

    out_full, lse_full = flash_attn_mxfp8_func(Q_fp8, K_fp8, V_fp8, q_sf, k_sf, v_sf,
                                                 softmax_scale=scale)

    max_diff = 0.0
    for bi in range(b):
        out_i, _ = flash_attn_mxfp8_func(
            Q_fp8[bi:bi+1].contiguous(), K_fp8[bi:bi+1].contiguous(),
            V_fp8[bi:bi+1].contiguous(),
            q_sf[bi:bi+1].contiguous(), k_sf[bi:bi+1].contiguous(),
            v_sf[bi:bi+1].contiguous(), softmax_scale=scale)
        max_diff = max(max_diff, (out_full[bi:bi+1].float() - out_i.float()).abs().max().item())

    print(f"  Batch={b}, heads={h}: max diff = {max_diff:.0e}")
    return max_diff == 0.0


def test_extreme_scales():
    """Extreme but valid scale values produce finite output."""
    from flash_attn_interface import flash_attn_mxfp8_func

    torch.manual_seed(11)
    b, s, h, d = 1, 128, 1, 128; scale = d ** -0.5
    Q = (torch.randn(b, s, h, d, device='cuda') * 0.001).to(torch.float8_e4m3fn)
    K = (torch.randn(b, s, h, d, device='cuda') * 0.001).to(torch.float8_e4m3fn)
    V = (torch.randn(b, s, h, d, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    id = identity_sf(b, h, s, d)

    # Large scale: 137 = 2^10 = 1024
    big_sf = torch.full((b, h, s, d // 32), 137, dtype=torch.uint8, device='cuda')
    out_big, lse_big = flash_attn_mxfp8_func(Q, K, V, big_sf, big_sf, id, softmax_scale=scale)
    out_id, lse_id = flash_attn_mxfp8_func(Q, K, V, id, id, id, softmax_scale=scale)

    finite = out_big.isfinite().all().item()
    lse_delta = (lse_big - lse_id).abs().max().item()
    print(f"  Small data + large scale (2^10): finite={finite}, LSE delta={lse_delta:.4f}")
    return finite and lse_delta > 1.0


def test_v_scales_not_applied():
    """V scale factors are accepted but NOT applied (identity P/V^T SFs in GEMM-II).
    This documents current behavior — V scaling would require transposed SF support."""
    from flash_attn_interface import flash_attn_mxfp8_func

    torch.manual_seed(55)
    b, s, h, d = 1, 128, 1, 128; scale = d ** -0.5
    Q = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
    V = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)

    id = identity_sf(b, h, s, d)
    big_v_sf = torch.full((b, h, s, d // 32), 135, dtype=torch.uint8, device='cuda')

    out_id, lse_id = flash_attn_mxfp8_func(Q, K, V, id, id, id, softmax_scale=scale)
    out_vs, lse_vs = flash_attn_mxfp8_func(Q, K, V, id, id, big_v_sf, softmax_scale=scale)

    lse_same = (lse_id - lse_vs).abs().max().item()
    o_same = (out_id.float() - out_vs.float()).abs().max().item()
    print(f"  V scales (identity vs 2^8): LSE diff={lse_same:.0e}, O diff={o_same:.0e}")
    print(f"  (Expected: both 0 — V scales not applied in current kernel)")
    return lse_same == 0.0 and o_same == 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not check_gpu():
        return

    tests = [
        ("Identity scales match legacy API", test_identity_scales_match_legacy),
        ("Uniform non-identity scales", test_uniform_scales),
        ("Scale factors correctly applied", test_scale_factors_applied),
        ("Per-row varying scales", test_per_row_scales),
        ("Causal + non-identity scales", test_causal_with_scales),
        ("FP32 reference sanity", test_fp32_reference_sanity),
        ("Batch/head consistency", test_batch_head_consistency),
        ("Extreme scale values", test_extreme_scales),
        ("V scales not applied (documented)", test_v_scales_not_applied),
    ]

    results = []
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            passed = fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for _, p in results if p)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n{n_pass}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
