"""Full fwd+bwd benchmark suite for SM120 MXFP8 Flash Attention vs SDPA BF16."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/nanogpt/prj/fp8_flashattention/flash-attention/hopper")
from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func, _flash_attn_forward

BLOCK_SIZE = 32

def benchmark_kernel(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return times[len(times) // 2]

def fwd_flops(b, s, h, d, causal=False):
    total = 2 * 2 * b * h * s * s * d  # 2 GEMMs
    return total // 2 if causal else total

def bwd_flops(b, s, h, d, causal=False):
    total = 5 * 2 * b * h * s * s * d  # 5 GEMMs
    return total // 2 if causal else total

def identity_sf(b, h, s, d):
    return torch.full((b, h, s, d // BLOCK_SIZE), 127, dtype=torch.uint8, device="cuda")

# ── SM120 FP8 benchmarks ─────────────────────────────────────────────────────

def bench_sm120_fwd(b, s, h, d, causal):
    q = torch.randn(b, s, h, d, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(b, s, h, d, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn(b, s, h, d, device="cuda").to(torch.float8_e4m3fn)
    scale = d ** -0.5
    def fn():
        _flash_attn_forward(q, k, v, softmax_scale=scale, causal=causal)
    return benchmark_kernel(fn)

def bench_sm120_bwd(b, s, h, d, causal):
    q = torch.randn(b, s, h, d, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(b, s, h, d, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn(b, s, h, d, device="cuda").to(torch.float8_e4m3fn)
    sf = identity_sf(b, h, s, d)
    scale = d ** -0.5
    out, lse = flash_attn_mxfp8_func(q, k, v, sf, sf, sf, softmax_scale=scale, causal=causal)
    dout = torch.randn_like(out)
    def fn():
        flash_attn_mxfp8_bwd_func(dout, q, k, v, out, lse, sf, sf,
                                   softmax_scale=scale, causal=causal)
    return benchmark_kernel(fn)

# ── SDPA BF16 benchmarks ─────────────────────────────────────────────────────

def bench_sdpa_fwd(b, s, h, d, causal):
    q = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
    def fn():
        F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return benchmark_kernel(fn)

def bench_sdpa_bwd(b, s, h, d, causal):
    q = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    dout = torch.randn_like(out)
    def fn():
        out.backward(dout, retain_graph=True)
        q.grad = k.grad = v.grad = None
    return benchmark_kernel(fn)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
    print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}\n")

    configs = [
        # (batch, seqlen, nheads, headdim, causal)
        (1,   512,  32, 128, False),
        (1,  1024,  32, 128, False),
        (1,  2048,  32, 128, False),
        (1,  4096,  32, 128, False),
        (4,   512,  32, 128, False),
        (4,  1024,  32, 128, False),
        (4,  2048,  32, 128, False),
        # causal
        (1,  1024,  32, 128, True),
        (1,  2048,  32, 128, True),
        (1,  4096,  32, 128, True),
        (2,  2048,  32, 128, True),
        (4,  2048,  32, 128, True),
    ]

    hdr = (f"{'Config':<25s} "
           f"{'FP8 fwd':>8s} {'FP8 bwd':>8s} {'f+b ms':>8s} {'TFLOPS':>8s} | "
           f"{'BF16 fwd':>8s} {'BF16 bwd':>8s} {'f+b ms':>8s} {'TFLOPS':>8s} | "
           f"{'Ratio':>6s}")
    print(hdr)
    print("-" * len(hdr))

    for b, s, h, d, causal in configs:
        C = "C" if causal else "F"
        label = f"({b},{s},{h},{d},{C})"

        # SM120 FP8
        try:
            ms_fwd = bench_sm120_fwd(b, s, h, d, causal)
            ms_bwd = bench_sm120_bwd(b, s, h, d, causal)
            ms_fb = ms_fwd + ms_bwd
            flops_fb = fwd_flops(b, s, h, d, causal) + bwd_flops(b, s, h, d, causal)
            tf_fb = flops_fb / (ms_fb * 1e-3) / 1e12
            fp8_str = f"{ms_fwd:8.3f} {ms_bwd:8.3f} {ms_fb:8.3f} {tf_fb:8.1f}"
        except Exception as e:
            fp8_str = f"{'ERR':>8s} {'':>8s} {'':>8s} {'':>8s}"
            ms_fb = None

        # SDPA BF16
        try:
            ms_fwd_s = bench_sdpa_fwd(b, s, h, d, causal)
            ms_bwd_s = bench_sdpa_bwd(b, s, h, d, causal)
            ms_fb_s = ms_fwd_s + ms_bwd_s
            flops_fb_s = fwd_flops(b, s, h, d, causal) + bwd_flops(b, s, h, d, causal)
            tf_fb_s = flops_fb_s / (ms_fb_s * 1e-3) / 1e12
            bf16_str = f"{ms_fwd_s:8.3f} {ms_bwd_s:8.3f} {ms_fb_s:8.3f} {tf_fb_s:8.1f}"
        except Exception as e:
            bf16_str = f"{'ERR':>8s} {'':>8s} {'':>8s} {'':>8s}"
            ms_fb_s = None

        # Ratio
        if ms_fb and ms_fb_s:
            ratio = f"{ms_fb_s / ms_fb:5.2f}x"
        else:
            ratio = "  N/A"

        print(f"{label:<25s} {fp8_str} | {bf16_str} | {ratio:>6s}")

    print()
    print("TFLOPS = (fwd_flops + bwd_flops) / (fwd_ms + bwd_ms). Ratio = SDPA_time / SM120_time (>1 = SM120 faster).")
    print("FP8 fwd: 2 GEMMs. FP8 bwd: 5 GEMMs. Causal halves effective FLOPs.")

if __name__ == "__main__":
    main()
