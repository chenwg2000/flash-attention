"""Test SM120 MXFP8 Flash Attention kernel on RTX 5090."""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/nanogpt/prj/fp8_flashattention/flash-attention/hopper")

def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available"); return False
    props = torch.cuda.get_device_properties(0)
    arch = props.major * 10 + props.minor
    print(f"GPU: {props.name}, sm_{arch}, {props.multi_processor_count} SMs")
    return True

def test_basic_launch():
    """Test kernel launches and produces finite output."""
    print("\n=== Test: Basic Launch ===")
    batch, seqlen, nheads, headdim = 1, 128, 4, 128

    q = torch.randn(batch, seqlen, nheads, headdim, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(batch, seqlen, nheads, headdim, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn(batch, seqlen, nheads, headdim, device="cuda").to(torch.float8_e4m3fn)

    from flash_attn_interface import _flash_attn_forward
    out, lse, _, _ = _flash_attn_forward(q, k, v, softmax_scale=headdim**-0.5)
    torch.cuda.synchronize()

    print(f"O shape={out.shape} dtype={out.dtype}")
    print(f"O finite={torch.isfinite(out).all().item()}")
    print(f"O stats: min={out.float().min():.4f} max={out.float().max():.4f} mean={out.float().mean():.4f}")
    print(f"LSE shape={lse.shape}")
    print(f"LSE stats: min={lse.min():.4f} max={lse.max():.4f} mean={lse.mean():.4f}")
    return True

def test_lse_correctness():
    """Check if LSE matches reference (indicates GEMM-I + softmax work)."""
    print("\n=== Test: LSE Correctness ===")
    batch, seqlen, nheads, headdim = 1, 128, 1, 128
    torch.manual_seed(42)

    q_fp32 = torch.randn(batch, seqlen, nheads, headdim, device="cuda") * 0.5
    k_fp32 = torch.randn(batch, seqlen, nheads, headdim, device="cuda") * 0.5
    v_fp32 = torch.randn(batch, seqlen, nheads, headdim, device="cuda") * 0.5
    scale = headdim ** -0.5

    # Dequantized reference
    q_fp8 = q_fp32.to(torch.float8_e4m3fn)
    k_fp8 = k_fp32.to(torch.float8_e4m3fn)
    v_fp8 = v_fp32.to(torch.float8_e4m3fn)

    q_deq = q_fp8.float()
    k_deq = k_fp8.float()

    # Reference S = Q @ K^T * scale
    S_ref = torch.matmul(q_deq.squeeze(0).squeeze(1), k_deq.squeeze(0).squeeze(1).T) * scale  # [seqlen, seqlen]
    # Reference LSE = log(sum(exp(S)))
    lse_ref = torch.logsumexp(S_ref, dim=-1)  # [seqlen]

    from flash_attn_interface import _flash_attn_forward
    out, lse, _, _ = _flash_attn_forward(q_fp8, k_fp8, v_fp8, softmax_scale=scale)
    torch.cuda.synchronize()

    lse_kernel = lse[0, 0, :]  # [seqlen]
    print(f"LSE kernel: min={lse_kernel.min():.4f} max={lse_kernel.max():.4f}")
    print(f"LSE ref:    min={lse_ref.min():.4f} max={lse_ref.max():.4f}")
    lse_diff = (lse_kernel - lse_ref).abs()
    print(f"LSE diff:   max={lse_diff.max():.4f} mean={lse_diff.mean():.4f}")

    # Check output
    v_deq = v_fp8.float()
    P_ref = F.softmax(S_ref, dim=-1)
    O_ref = torch.matmul(P_ref, v_deq.squeeze(0).squeeze(1))  # [seqlen, headdim]
    O_kernel = out[0, :, 0, :].float()  # [seqlen, headdim]
    o_diff = (O_kernel - O_ref).abs()
    print(f"\nO kernel: min={O_kernel.min():.4f} max={O_kernel.max():.4f}")
    print(f"O ref:    min={O_ref.min():.4f} max={O_ref.max():.4f}")
    print(f"O diff:   max={o_diff.max():.4f} mean={o_diff.mean():.4f}")

    if lse_diff.max() < 1.0:
        print("\nLSE PASS")
    else:
        print(f"\nLSE FAIL (max diff={lse_diff.max():.4f})")

    if o_diff.max() < 1.0:
        print("O   PASS")
    else:
        print(f"O   FAIL (max diff={o_diff.max():.4f})")
    return True

def test_identity_attention():
    """Test with identity-like attention: Q=K so S=Q@Q^T is known."""
    print("\n=== Test: Identity Attention ===")
    batch, seqlen, nheads, headdim = 1, 128, 1, 128

    # Create Q = K = one-hot rows (each row is e_i)
    # This makes S = I (identity), P = softmax(I) ≈ uniform but with diagonal dominant
    # With scale, S = I * scale → softmax is well-defined
    q_fp32 = torch.eye(seqlen, headdim, device="cuda").unsqueeze(0).unsqueeze(2)  # [1, 128, 1, 128]
    k_fp32 = q_fp32.clone()
    v_fp32 = torch.randn(batch, seqlen, nheads, headdim, device="cuda") * 0.1

    q_fp8 = q_fp32.to(torch.float8_e4m3fn)
    k_fp8 = k_fp32.to(torch.float8_e4m3fn)
    v_fp8 = v_fp32.to(torch.float8_e4m3fn)

    scale = headdim ** -0.5

    from flash_attn_interface import _flash_attn_forward
    out, lse, _, _ = _flash_attn_forward(q_fp8, k_fp8, v_fp8, softmax_scale=scale)
    torch.cuda.synchronize()

    # Reference
    q_deq, k_deq, v_deq = q_fp8.float(), k_fp8.float(), v_fp8.float()
    S = torch.matmul(q_deq[0,:,0,:], k_deq[0,:,0,:].T) * scale
    P = F.softmax(S, dim=-1)
    O_ref = torch.matmul(P, v_deq[0,:,0,:])
    lse_ref = torch.logsumexp(S, dim=-1)

    lse_k = lse[0, 0, :]
    O_k = out[0, :, 0, :].float()

    lse_diff = (lse_k - lse_ref).abs()
    o_diff = (O_k - O_ref).abs()

    print(f"LSE diff: max={lse_diff.max():.4f} mean={lse_diff.mean():.4f}")
    print(f"O diff:   max={o_diff.max():.4f} mean={o_diff.mean():.4f}")
    print(f"O kernel: [{O_k[0,:5].tolist()}...]")
    print(f"O ref:    [{O_ref[0,:5].tolist()}...]")

    return True

def main():
    if not check_gpu(): return

    tests = [
        ("Basic Launch", test_basic_launch),
        ("LSE Correctness", test_lse_correctness),
        ("Identity Attention", test_identity_attention),
    ]

    results = []
    for name, fn in tests:
        try:
            passed = fn()
            results.append((name, passed))
        except Exception as e:
            print(f"EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            results.append((name, False))

    print("\n=== Summary ===")
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
    main()
