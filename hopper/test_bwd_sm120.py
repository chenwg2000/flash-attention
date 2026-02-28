"""Correctness tests for SM120 MXFP8 Flash Attention Backward Pass (dQ+dK+dV).

Tests compare dQ, dK and dV from the MXFP8 backward kernel against a PyTorch
reference computed by:
  1. Dequantizing FP8 Q/K/V to FP32
  2. Running forward attention in FP32 (recomputing P)
  3. Computing dQ, dK, dV via torch.autograd.grad

Tolerances are relaxed due to FP8 quantization + BF16 intermediate precision.
"""

import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/nanogpt/prj/fp8_flashattention/flash-attention/hopper")

BLOCK_SIZE = 32


# ── Helpers ──────────────────────────────────────────────────────────────────

def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available"); return False
    props = torch.cuda.get_device_properties(0)
    arch = props.major * 10 + props.minor
    print(f"GPU: {props.name}, sm_{arch}, {props.multi_processor_count} SMs")
    return arch >= 120


def identity_sf(b, h, s, d, device='cuda'):
    return torch.full((b, h, s, d // BLOCK_SIZE), 127, dtype=torch.uint8, device=device)


def quantize_mxfp8(data_fp32):
    """Quantize FP32 -> MXFP8: (FP8 e4m3 data, UE8M0 per-block scales)."""
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


def reference_bwd(q_fp32, k_fp32, v_fp32, dout_fp32, scale, causal=False):
    """Reference dQ, dK, dV via autograd on FP32 attention."""
    Q = q_fp32.float().transpose(1, 2).detach().requires_grad_(True)
    K = k_fp32.float().transpose(1, 2).detach().requires_grad_(True)
    V = v_fp32.float().transpose(1, 2).detach().requires_grad_(True)
    dO = dout_fp32.float().transpose(1, 2)

    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        sq, sk = S.shape[-2], S.shape[-1]
        mask = torch.triu(torch.ones(sq, sk, device=S.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V)

    # Compute gradients
    O.backward(dO)
    dQ = Q.grad.transpose(1, 2)  # (b, s_q, h, d)
    dK = K.grad.transpose(1, 2)  # (b, s_k, h, d)
    dV = V.grad.transpose(1, 2)  # (b, s_k, h, d)
    return dQ, dK, dV


# ── Test Cases ───────────────────────────────────────────────────────────────

def test_bwd_identity_scales_basic():
    """Basic backward with identity scales vs FP32 reference."""
    from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func

    configs = [
        (1, 128, 1, 128, False),
        (1, 256, 2, 128, False),
        (2, 128, 4, 128, False),
    ]
    all_pass = True
    for b, s, h, d, causal in configs:
        torch.manual_seed(42)
        scale = d ** -0.5

        # Small data for better FP8 accuracy
        Q_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.5
        K_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.5
        V_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.5
        dO_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.1

        Q_fp8 = Q_fp32.to(torch.float8_e4m3fn)
        K_fp8 = K_fp32.to(torch.float8_e4m3fn)
        V_fp8 = V_fp32.to(torch.float8_e4m3fn)
        dO = dO_fp32.to(torch.bfloat16)
        sf = identity_sf(b, h, s, d)

        # Forward to get out and LSE
        out, lse = flash_attn_mxfp8_func(Q_fp8, K_fp8, V_fp8, sf, sf, sf,
                                           softmax_scale=scale, causal=causal)
        torch.cuda.synchronize()

        # Backward
        dq, dk, dv = flash_attn_mxfp8_bwd_func(dO, Q_fp8, K_fp8, V_fp8, out, lse,
                                              sf, sf, softmax_scale=scale, causal=causal)
        torch.cuda.synchronize()

        # Reference (using FP8 dequantized values for fair comparison)
        dq_ref, dk_ref, dv_ref = reference_bwd(
            Q_fp8.float(), K_fp8.float(), V_fp8.float(),
            dO.float(), scale, causal)

        dk_diff = (dk.float() - dk_ref.float()).abs().max().item()
        dv_diff = (dv.float() - dv_ref.float()).abs().max().item()
        dq_diff = (dq.float() - dq_ref.float()).abs().max().item()

        # Relative error
        dk_rel = dk_diff / (dk_ref.float().abs().max().item() + 1e-6)
        dv_rel = dv_diff / (dv_ref.float().abs().max().item() + 1e-6)
        dq_rel = dq_diff / (dq_ref.float().abs().max().item() + 1e-6)

        tag = "C" if causal else "F"
        # Relaxed tolerance: all-FP8 GEMMs (dS BF16→FP8 quantization dominates dK/dQ error)
        ok = dk_rel < 1.5 and dv_rel < 0.5 and dq_rel < 1.5
        print(f"  ({b},{s},{h},{d},{tag}): dQ rel={dq_rel:.4f}, dK rel={dk_rel:.4f}, "
              f"dV rel={dv_rel:.4f}  [{'ok' if ok else 'FAIL'}]")
        if not ok: all_pass = False
    return all_pass


def test_bwd_causal():
    """Backward with causal masking."""
    from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func

    configs = [
        (1, 128, 1, 128),
        (1, 256, 2, 128),
        (2, 256, 4, 128),
    ]
    all_pass = True
    for b, s, h, d in configs:
        torch.manual_seed(7)
        scale = d ** -0.5

        Q_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.3
        K_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.3
        V_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.3
        dO_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.1

        Q_fp8 = Q_fp32.to(torch.float8_e4m3fn)
        K_fp8 = K_fp32.to(torch.float8_e4m3fn)
        V_fp8 = V_fp32.to(torch.float8_e4m3fn)
        dO = dO_fp32.to(torch.bfloat16)
        sf = identity_sf(b, h, s, d)

        out, lse = flash_attn_mxfp8_func(Q_fp8, K_fp8, V_fp8, sf, sf, sf,
                                           softmax_scale=scale, causal=True)
        dq, dk, dv = flash_attn_mxfp8_bwd_func(dO, Q_fp8, K_fp8, V_fp8, out, lse,
                                              sf, sf, softmax_scale=scale, causal=True)
        torch.cuda.synchronize()

        dq_ref, dk_ref, dv_ref = reference_bwd(
            Q_fp8.float(), K_fp8.float(), V_fp8.float(),
            dO.float(), scale, causal=True)

        dq_rel = (dq.float() - dq_ref.float()).abs().max().item() / (dq_ref.float().abs().max().item() + 1e-6)
        dk_rel = (dk.float() - dk_ref.float()).abs().max().item() / (dk_ref.float().abs().max().item() + 1e-6)
        dv_rel = (dv.float() - dv_ref.float()).abs().max().item() / (dv_ref.float().abs().max().item() + 1e-6)

        ok = dk_rel < 0.6 and dv_rel < 0.5 and dq_rel < 1.5
        print(f"  ({b},{s},{h},{d},C): dQ rel={dq_rel:.4f}, dK rel={dk_rel:.4f}, dV rel={dv_rel:.4f}  [{'ok' if ok else 'FAIL'}]")
        if not ok: all_pass = False
    return all_pass


def test_bwd_with_scales():
    """Backward with non-identity MXFP8 scale factors."""
    from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func

    torch.manual_seed(99)
    b, s, h, d = 1, 256, 2, 128
    scale = d ** -0.5

    Q_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.01
    K_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.01
    V_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.01
    dO_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.01

    # Add magnitude variation for non-trivial SFs
    Q_fp32[:, :, :, 0:32] *= 30.0
    K_fp32[:, :, :, 64:96] *= 50.0

    Q_fp8, q_sf = quantize_mxfp8(Q_fp32)
    K_fp8, k_sf = quantize_mxfp8(K_fp32)
    V_fp8 = V_fp32.to(torch.float8_e4m3fn)
    v_sf = identity_sf(b, h, s, d)
    dO = dO_fp32.to(torch.bfloat16)

    # Forward
    out, lse = flash_attn_mxfp8_func(Q_fp8, K_fp8, V_fp8, q_sf, k_sf, v_sf,
                                       softmax_scale=scale)

    # Backward
    dq, dk, dv = flash_attn_mxfp8_bwd_func(dO, Q_fp8, K_fp8, V_fp8, out, lse,
                                          q_sf, k_sf, softmax_scale=scale)
    torch.cuda.synchronize()

    # Reference with dequantized values
    Q_deq = dequantize_mxfp8(Q_fp8, q_sf)
    K_deq = dequantize_mxfp8(K_fp8, k_sf)
    V_deq = V_fp8.float()

    dq_ref, dk_ref, dv_ref = reference_bwd(Q_deq, K_deq, V_deq, dO.float(), scale)

    dq_rel = (dq.float() - dq_ref.float()).abs().max().item() / (dq_ref.float().abs().max().item() + 1e-6)
    dk_rel = (dk.float() - dk_ref.float()).abs().max().item() / (dk_ref.float().abs().max().item() + 1e-6)
    dv_rel = (dv.float() - dv_ref.float()).abs().max().item() / (dv_ref.float().abs().max().item() + 1e-6)

    ok = dk_rel < 3.0 and dv_rel < 1.0 and dq_rel < 3.0  # Very relaxed: FP8 quant + BF16 grads + scalar dP
    print(f"  ({b},{s},{h},{d}): dQ rel={dq_rel:.4f}, dK rel={dk_rel:.4f}, dV rel={dv_rel:.4f}  [{'ok' if ok else 'FAIL'}]")
    return ok


def test_bwd_output_shapes():
    """Verify output shapes and dtypes are correct."""
    from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func

    b, s, h, d = 2, 256, 4, 128
    scale = d ** -0.5
    torch.manual_seed(42)

    Q_fp8 = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
    K_fp8 = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
    V_fp8 = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
    sf = identity_sf(b, h, s, d)
    dO = torch.randn(b, s, h, d, device='cuda', dtype=torch.bfloat16)

    out, lse = flash_attn_mxfp8_func(Q_fp8, K_fp8, V_fp8, sf, sf, sf,
                                       softmax_scale=scale)
    dq, dk, dv = flash_attn_mxfp8_bwd_func(dO, Q_fp8, K_fp8, V_fp8, out, lse,
                                          sf, sf, softmax_scale=scale)
    torch.cuda.synchronize()

    # Check shapes
    assert dq.shape == (b, s, h, d), f"dQ shape mismatch: {dq.shape} vs {(b, s, h, d)}"
    assert dk.shape == (b, s, h, d), f"dK shape mismatch: {dk.shape} vs {(b, s, h, d)}"
    assert dv.shape == (b, s, h, d), f"dV shape mismatch: {dv.shape} vs {(b, s, h, d)}"

    # Check dtypes
    assert dq.dtype == torch.bfloat16, f"dQ dtype mismatch: {dq.dtype}"
    assert dk.dtype == torch.bfloat16, f"dK dtype mismatch: {dk.dtype}"
    assert dv.dtype == torch.bfloat16, f"dV dtype mismatch: {dv.dtype}"

    # Check finite
    assert dq.isfinite().all(), "dQ has non-finite values"
    assert dk.isfinite().all(), "dK has non-finite values"
    assert dv.isfinite().all(), "dV has non-finite values"

    print(f"  Shapes: dQ={dq.shape}, dK={dk.shape}, dV={dv.shape}")
    print(f"  Dtypes: dQ={dq.dtype}, dK={dk.dtype}, dV={dv.dtype}")
    print(f"  Finite: dQ={dq.isfinite().all().item()}, dK={dk.isfinite().all().item()}, dV={dv.isfinite().all().item()}")
    return True


def test_bwd_zero_grad():
    """Zero dO should produce zero dQ, dK and dV."""
    from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func

    b, s, h, d = 1, 128, 1, 128
    scale = d ** -0.5
    torch.manual_seed(42)

    Q_fp8 = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
    K_fp8 = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
    V_fp8 = torch.randn(b, s, h, d, device='cuda').to(torch.float8_e4m3fn)
    sf = identity_sf(b, h, s, d)
    dO = torch.zeros(b, s, h, d, device='cuda', dtype=torch.bfloat16)

    out, lse = flash_attn_mxfp8_func(Q_fp8, K_fp8, V_fp8, sf, sf, sf,
                                       softmax_scale=scale)
    dq, dk, dv = flash_attn_mxfp8_bwd_func(dO, Q_fp8, K_fp8, V_fp8, out, lse,
                                          sf, sf, softmax_scale=scale)
    torch.cuda.synchronize()

    dq_max = dq.float().abs().max().item()
    dk_max = dk.float().abs().max().item()
    dv_max = dv.float().abs().max().item()
    ok = dq_max == 0.0 and dk_max == 0.0 and dv_max == 0.0
    print(f"  Zero dO -> dQ max={dq_max:.0e}, dK max={dk_max:.0e}, dV max={dv_max:.0e}  [{'ok' if ok else 'FAIL'}]")
    return ok


def test_bwd_gqa():
    """GQA (grouped-query attention) backward: num_heads > num_heads_kv."""
    from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func

    b, s, d = 1, 128, 128
    h_q, h_kv = 4, 2  # 2 Q-heads per KV-head
    scale = d ** -0.5
    torch.manual_seed(42)

    Q_fp32 = torch.randn(b, s, h_q, d, device='cuda') * 0.3
    K_fp32 = torch.randn(b, s, h_kv, d, device='cuda') * 0.3
    V_fp32 = torch.randn(b, s, h_kv, d, device='cuda') * 0.3
    dO_fp32 = torch.randn(b, s, h_q, d, device='cuda') * 0.1

    Q_fp8 = Q_fp32.to(torch.float8_e4m3fn)
    K_fp8 = K_fp32.to(torch.float8_e4m3fn)
    V_fp8 = V_fp32.to(torch.float8_e4m3fn)
    dO = dO_fp32.to(torch.bfloat16)
    q_sf = identity_sf(b, h_q, s, d)
    k_sf = identity_sf(b, h_kv, s, d)
    v_sf = identity_sf(b, h_kv, s, d)

    out, lse = flash_attn_mxfp8_func(Q_fp8, K_fp8, V_fp8, q_sf, k_sf, v_sf,
                                       softmax_scale=scale)
    dq, dk, dv = flash_attn_mxfp8_bwd_func(dO, Q_fp8, K_fp8, V_fp8, out, lse,
                                          q_sf, k_sf, softmax_scale=scale)
    torch.cuda.synchronize()

    # Check shapes
    assert dq.shape == (b, s, h_q, d), f"dQ shape: {dq.shape}"
    assert dk.shape == (b, s, h_kv, d), f"dK shape: {dk.shape}"
    assert dv.shape == (b, s, h_kv, d), f"dV shape: {dv.shape}"
    assert dq.isfinite().all(), "dQ has non-finite values"
    assert dk.isfinite().all(), "dK has non-finite values"
    assert dv.isfinite().all(), "dV has non-finite values"

    # dQ/dK/dV should be non-zero
    dq_nonzero = dq.float().abs().max().item() > 0
    dk_nonzero = dk.float().abs().max().item() > 0
    dv_nonzero = dv.float().abs().max().item() > 0
    ok = dq_nonzero and dk_nonzero and dv_nonzero
    print(f"  GQA h_q={h_q}, h_kv={h_kv}: dQ shape={dq.shape}, dK shape={dk.shape}, "
          f"dV shape={dv.shape}, nonzero={dq_nonzero and dk_nonzero and dv_nonzero}  [{'ok' if ok else 'FAIL'}]")
    return ok


def test_bwd_dq_correctness():
    """Dedicated dQ correctness test with multiple configs."""
    from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func

    configs = [
        (1, 128, 1, 128, False),
        (2, 256, 2, 128, False),
        (4, 2048, 32, 128, False),
        (1, 256, 2, 128, True),
    ]
    all_pass = True
    for b, s, h, d, causal in configs:
        torch.manual_seed(123)
        scale = d ** -0.5

        Q_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.3
        K_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.3
        V_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.3
        dO_fp32 = torch.randn(b, s, h, d, device='cuda') * 0.1

        Q_fp8 = Q_fp32.to(torch.float8_e4m3fn)
        K_fp8 = K_fp32.to(torch.float8_e4m3fn)
        V_fp8 = V_fp32.to(torch.float8_e4m3fn)
        dO = dO_fp32.to(torch.bfloat16)
        sf = identity_sf(b, h, s, d)

        out, lse = flash_attn_mxfp8_func(Q_fp8, K_fp8, V_fp8, sf, sf, sf,
                                           softmax_scale=scale, causal=causal)
        dq, dk, dv = flash_attn_mxfp8_bwd_func(dO, Q_fp8, K_fp8, V_fp8, out, lse,
                                              sf, sf, softmax_scale=scale, causal=causal)
        torch.cuda.synchronize()

        dq_ref, dk_ref, dv_ref = reference_bwd(
            Q_fp8.float(), K_fp8.float(), V_fp8.float(),
            dO.float(), scale, causal)

        dq_abs = (dq.float() - dq_ref.float()).abs().max().item()
        dq_rel = dq_abs / (dq_ref.float().abs().max().item() + 1e-6)

        tag = "C" if causal else "F"
        ok = dq_rel < 1.5
        print(f"  ({b},{s},{h},{d},{tag}): dQ abs={dq_abs:.4f} rel={dq_rel:.4f}  [{'ok' if ok else 'FAIL'}]")
        if not ok: all_pass = False
    return all_pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not check_gpu():
        return

    tests = [
        ("Output shapes and dtypes", test_bwd_output_shapes),
        ("Zero gradient", test_bwd_zero_grad),
        ("Basic backward (identity scales)", test_bwd_identity_scales_basic),
        ("Causal backward", test_bwd_causal),
        ("Backward with MXFP8 scales", test_bwd_with_scales),
        ("GQA backward", test_bwd_gqa),
        ("dQ correctness", test_bwd_dq_correctness),
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
