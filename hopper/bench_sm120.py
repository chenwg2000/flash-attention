"""Performance benchmark for SM120 MXFP8 Flash Attention forward kernel."""

import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/nanogpt/prj/fp8_flashattention/flash-attention/hopper")

def benchmark_kernel(fn, warmup=10, iters=100):
    """Benchmark a CUDA kernel, return median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2]  # median


def compute_flops(batch, seqlen_q, seqlen_k, nheads, headdim, causal=False):
    """Compute FLOPs for attention forward pass."""
    # GEMM-I: Q @ K^T -> [batch, nheads, seqlen_q, seqlen_k]
    flops_qk = 2 * batch * nheads * seqlen_q * seqlen_k * headdim
    # GEMM-II: P @ V -> [batch, nheads, seqlen_q, headdim]
    flops_pv = 2 * batch * nheads * seqlen_q * headdim * seqlen_k
    total = flops_qk + flops_pv
    if causal:
        total //= 2  # approximately half the work
    return total


def bench_sm120_fa3(batch, seqlen, nheads, headdim, causal=False):
    """Benchmark SM120 MXFP8 FA3 kernel."""
    from flash_attn_interface import _flash_attn_forward

    q = torch.randn(batch, seqlen, nheads, headdim, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(batch, seqlen, nheads, headdim, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn(batch, seqlen, nheads, headdim, device="cuda").to(torch.float8_e4m3fn)
    scale = headdim ** -0.5

    def fn():
        _flash_attn_forward(q, k, v, softmax_scale=scale, causal=causal)

    ms = benchmark_kernel(fn)
    flops = compute_flops(batch, seqlen, seqlen, nheads, headdim, causal)
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops


def bench_torch_sdpa_bf16(batch, seqlen, nheads, headdim, causal=False):
    """Benchmark PyTorch SDPA with BF16."""
    q = torch.randn(batch, nheads, seqlen, headdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, nheads, seqlen, headdim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, nheads, seqlen, headdim, device="cuda", dtype=torch.bfloat16)

    def fn():
        F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    ms = benchmark_kernel(fn)
    flops = compute_flops(batch, seqlen, seqlen, nheads, headdim, causal)
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops


def bench_torch_sdpa_fp16(batch, seqlen, nheads, headdim, causal=False):
    """Benchmark PyTorch SDPA with FP16."""
    q = torch.randn(batch, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)

    def fn():
        F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    ms = benchmark_kernel(fn)
    flops = compute_flops(batch, seqlen, seqlen, nheads, headdim, causal)
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops


def main():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
    print()

    configs = [
        # (batch, seqlen, nheads, headdim, causal)
        (1,   128,  32, 128, False),
        (1,   256,  32, 128, False),
        (1,   512,  32, 128, False),
        (1,  1024,  32, 128, False),
        (1,  2048,  32, 128, False),
        (1,  4096,  32, 128, False),
        (4,   512,  32, 128, False),
        (4,  2048,  32, 128, False),
        (1,  2048,  32, 128, True),
        (4,  2048,  32, 128, True),
    ]

    print(f"{'Config':<40s} {'SM120 FP8':>12s} {'SDPA BF16':>12s} {'SDPA FP16':>12s}")
    print(f"{'(b,s,h,d,causal)':<40s} {'ms / TFLOPS':>12s} {'ms / TFLOPS':>12s} {'ms / TFLOPS':>12s}")
    print("-" * 80)

    for batch, seqlen, nheads, headdim, causal in configs:
        label = f"({batch},{seqlen},{nheads},{headdim},{'C' if causal else 'F'})"

        try:
            ms_fa3, tf_fa3 = bench_sm120_fa3(batch, seqlen, nheads, headdim, causal)
            fa3_str = f"{ms_fa3:.3f} / {tf_fa3:.1f}"
        except Exception as e:
            fa3_str = f"ERR: {e}"[:20]

        try:
            ms_sdpa_bf16, tf_sdpa_bf16 = bench_torch_sdpa_bf16(batch, seqlen, nheads, headdim, causal)
            sdpa_bf16_str = f"{ms_sdpa_bf16:.3f} / {tf_sdpa_bf16:.1f}"
        except Exception as e:
            sdpa_bf16_str = f"ERR"

        try:
            ms_sdpa_fp16, tf_sdpa_fp16 = bench_torch_sdpa_fp16(batch, seqlen, nheads, headdim, causal)
            sdpa_fp16_str = f"{ms_sdpa_fp16:.3f} / {tf_sdpa_fp16:.1f}"
        except Exception as e:
            sdpa_fp16_str = f"ERR"

        print(f"{label:<40s} {fa3_str:>12s} {sdpa_bf16_str:>12s} {sdpa_fp16_str:>12s}")

    print()
    print("Note: SM120 FP8 kernel uses block-scaled MMA for both GEMM-I and GEMM-II.")
    print("      Further optimization possible: TMA loads, pipelined K/V, LDSM.T for V transpose.")


if __name__ == "__main__":
    main()
