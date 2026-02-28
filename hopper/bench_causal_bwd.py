"""Benchmark causal vs non-causal SM120 backward to measure early-exit speedup."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/nanogpt/prj/fp8_flashattention/flash-attention/hopper")
from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func

BLOCK_SIZE = 32

def benchmark_kernel(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    return times[len(times) // 2]

def compute_bwd_flops(b, s, h, d, causal=False):
    flops = 5 * 2 * b * h * s * s * d
    if causal:
        flops //= 2
    return flops

def identity_sf(b, h, s, d, device='cuda'):
    return torch.full((b, h, s, d // BLOCK_SIZE), 127, dtype=torch.uint8, device=device)

def bench_bwd(b, s, h, d, causal):
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
    ms = benchmark_kernel(fn)
    flops = compute_bwd_flops(b, s, h, d, causal)
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops

def bench_sdpa_bwd(b, s, h, d, causal):
    q = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    dout = torch.randn_like(out)
    def fn():
        out.backward(dout, retain_graph=True)
        q.grad = k.grad = v.grad = None
    ms = benchmark_kernel(fn)
    flops = compute_bwd_flops(b, s, h, d, causal)
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops

def main():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)\n")

    configs = [
        (1, 512, 32, 128),
        (1, 1024, 32, 128),
        (1, 2048, 32, 128),
        (1, 4096, 32, 128),
        (4, 512, 32, 128),
        (4, 1024, 32, 128),
        (4, 2048, 32, 128),
    ]

    print(f"{'Config':<22s} {'NC bwd ms':>10s} {'NC TFLOPS':>10s} {'C bwd ms':>10s} {'C TFLOPS':>10s} {'Speedup':>8s} | {'SDPA NC':>10s} {'SDPA C':>10s}")
    print("-" * 105)

    for b, s, h, d in configs:
        ms_nc, tf_nc = bench_bwd(b, s, h, d, False)
        ms_c, tf_c = bench_bwd(b, s, h, d, True)
        speedup = ms_nc / ms_c if ms_c > 0 else 0

        try:
            ms_sdpa_nc, _ = bench_sdpa_bwd(b, s, h, d, False)
            ms_sdpa_c, _ = bench_sdpa_bwd(b, s, h, d, True)
            sdpa_nc_str = f"{ms_sdpa_nc:.3f}"
            sdpa_c_str = f"{ms_sdpa_c:.3f}"
        except:
            sdpa_nc_str = "ERR"
            sdpa_c_str = "ERR"

        label = f"({b},{s},{h},{d})"
        print(f"{label:<22s} {ms_nc:10.3f} {tf_nc:10.1f} {ms_c:10.3f} {tf_c:10.1f} {speedup:7.2f}x | {sdpa_nc_str:>10s} {sdpa_c_str:>10s}")

if __name__ == "__main__":
    main()
