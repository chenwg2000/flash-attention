# SM120 Backward Kernel Optimization Analysis (Final)

## Current State (kBlockM=128 + GMEM dO + software-pipelined LDSM/MMA)

| Config | SM120 FP8 bwd | SDPA BF16 bwd | Gap |
|---|---|---|---|
| (1,2048,32,128,F) | 2.179 ms | 0.952 ms | 2.3x slower |
| (4,2048,32,128,F) | 8.849 ms | 3.589 ms | 2.5x slower |
| (1,4096,32,128,C) | 3.876 ms | 1.823 ms | 2.1x slower |
| (4,2048,32,128,C) | 4.181 ms | 2.098 ms | 2.0x slower |

Forward achieves 347 TFLOPS (1.7x faster than BF16). Backward is the entire gap.

## Complete Optimization History

| Optimization | Result | Key Learning |
|---|---|---|
| Warp specialization (4+4 split) | **-4 to -8%** (worse) | TC saturates better with 8 warps. 255 regs = no branch headroom. |
| Zero-fill elimination | **-1 to -2%** (worse) | Vectorized uint4 + `continue` beats per-byte conditional writes. |
| TMA bulk reduce for dQ | **~neutral** | Blackwell L2 atomicAdd already fast (~2 cycles amortized). |
| **kBlockM=128 + GMEM dO** | **+8 to +22%** ✅ | Big win: halved M-blocks, no padding/zero-fill, reduced spills. |
| Software-pipelined LDSM/MMA | **~neutral** | 8-warp scheduler provides sufficient inter-warp overlap. |

### What works on SM120:
- **Structural changes** that reduce total M-block count or eliminate overhead phases
- **Eliminating SMEM buffers** by reading from GMEM/L2 cache
- **Causal early-exit** (skipping fully-masked M-blocks)

### What doesn't work on SM120:
- **Warp specialization**: TC needs all 8 warps for pipeline utilization
- **Zero-fill elimination**: Vectorized uint4 + skip is already optimal
- **TMA bulk reduce**: Blackwell L2 handles atomicAdds efficiently
- **Intra-warp LDSM/MMA pipelining**: Inter-warp overlap is sufficient with 8 warps
- **Any change adding register pressure**: 255 regs at limit, spills kill performance

## The Fundamental Bottleneck: FP32→FP8 Conversion Chain

The remaining ~2x gap to BF16 is architectural. Between every pair of GEMMs, the backward must:

```
GEMM output (FP32 registers)
  → FP32→FP8 conversion (software, ~15-25 cycles per element)
  → Scatter to swizzled SMEM (per-byte writes with swizzle formula)
  → __syncthreads() (256 threads wait)
  → LDSM from swizzled SMEM to registers
  → Next GEMM
```

This chain repeats 4-5 times per M-block:
1. P scatter (Phase E): FP32 acc → FP8 transposed → smem_q → GEMM-3 reads P^T
2. dO→FP8 (Phase H): GMEM BF16 → FP8 → smem_pds → GEMM-2 reads dO_fp8
3. dO→dO^T (Phase F): GMEM BF16 → FP8 transposed → smem_pds → GEMM-3 reads dO^T
4. dS dual scatter (Phase LM): FP32 → FP8 → smem_q + smem_pds → GEMM-4/5 read
5. Q→Q^T (Phase D): FP8 SMEM → FP8 transposed SMEM → GEMM-4 reads Q^T

The forward has only **1 round-trip** (P scatter), with K/V pre-loaded via TMA pipeline.

### Why cuDNN BF16 avoids this:
1. **No FP8 conversion** — BF16 operands go directly to MMA
2. **TMEM on sm_100** — keeps intermediates in tensor memory (SM120 lacks TMEM)
3. **Descriptor-based WGMMA** — reads SMEM without explicit LDSM
4. **Warp-specialized producer/consumer** — overlaps memory with compute

### Theoretical minimum on SM120:
At forward-level TC utilization (~80%), pure GEMM time for backward would be ~0.64 ms
for (1,2048,32,128,F). Adding irreducible FP8 conversion pipeline latency (~0.3-0.5 ms),
the theoretical best is **~1.0 ms** vs current 2.18 ms. Still ~2× room, but requires
near-perfect memory/compute overlap that likely needs TMEM or descriptor-based MMA.

## Remaining Optimization Options (diminishing returns)

| Option | Expected | Feasibility |
|---|---|---|
| Vectorize FP8 scatter (uint32 writes) | 3-5% | Medium — swizzle alignment within 16B chunks |
| Reduce sync count (6→4) | 2-5% | Medium — reorder phases to merge compatible syncs |
| Fuse Phase D+H overlap | 2-3% | Low effort — already writes to disjoint SMEM |
| Reduce register spills further | 2-5% | Low — try `__noinline__` functions |
| TMA Q pipelining | <1% | High effort, Q load is only 0.4% of M-block time |

Combined: potentially ~10-15% more, bringing gap from ~2.1x to ~1.8x vs BF16.

## Hardware Constraints (RTX 5090 / SM 12.0)

- **100 KB SMEM per SM** (Hopper/B100: 228 KB)
- **255 registers max** with 1440B pre-existing spills
- **No TMEM** (tensor memory, only on sm_100 data center Blackwell)
- **Per-warp WGMMA** via LDSM (not descriptor-based like SM90 GMMA)
- **Block-scaled FP8 MMA** requires Blk_MN=128, kSFVecSize=32

## Benchmark Results (current, kBlockM=128 + GMEM dO + pipelined LDSM/MMA)

| Config (b,s,h,d) | FP8 fwd | FP8 bwd | f+b ms | TFLOPS | BF16 fwd | BF16 bwd | f+b ms | TFLOPS | Ratio |
|---|---|---|---|---|---|---|---|---|---|
| (1,512,32,128,F) | 0.039 | 0.177 | 0.216 | 69.5 | 0.032 | 0.089 | 0.121 | 124.3 | 0.56x |
| (1,1024,32,128,F) | 0.079 | 0.632 | 0.710 | 84.7 | 0.110 | 0.287 | 0.397 | 151.5 | 0.56x |
| (1,2048,32,128,F) | 0.250 | 2.179 | 2.429 | 99.0 | 0.417 | 0.952 | 1.369 | 175.7 | 0.56x |
| (1,4096,32,128,F) | 0.813 | 8.549 | 9.362 | 102.8 | 1.443 | 3.410 | 4.853 | 198.2 | 0.52x |
| (4,512,32,128,F) | 0.094 | 0.775 | 0.870 | 69.1 | 0.114 | 0.314 | 0.428 | 140.5 | 0.49x |
| (4,1024,32,128,F) | 0.243 | 2.464 | 2.707 | 88.9 | 0.378 | 1.036 | 1.414 | 170.0 | 0.52x |
| (4,2048,32,128,F) | 0.796 | 8.849 | 9.645 | 99.7 | 1.364 | 3.589 | 4.954 | 194.2 | 0.51x |
| (1,1024,32,128,C) | 0.056 | 0.445 | 0.501 | 60.0 | 0.083 | 0.178 | 0.261 | 115.1 | 0.52x |
| (1,2048,32,128,C) | 0.144 | 1.247 | 1.391 | 86.4 | 0.235 | 0.538 | 0.773 | 155.6 | 0.56x |
| (1,4096,32,128,C) | 0.449 | 3.876 | 4.325 | 111.2 | 0.774 | 1.823 | 2.597 | 185.2 | 0.60x |
| (2,2048,32,128,C) | 0.249 | 2.265 | 2.514 | 95.7 | 0.409 | 1.056 | 1.465 | 164.2 | 0.58x |
| (4,2048,32,128,C) | 0.476 | 4.181 | 4.657 | 103.3 | 0.745 | 2.098 | 2.843 | 169.2 | 0.61x |

Ratio = SDPA_time / SM120_time (>1 = SM120 faster).
Best combined TFLOPS: 111.2 at (1,4096,32,128,C).
Forward alone: ~347 TFLOPS (1.7x faster than BF16).

## Files Modified

| File | Changes |
|---|---|
| `mainloop_bwd_sm120_tma_mma.hpp` | kBlockM=128, GMEM dO, no zero-fills, pipelined LDSM/MMA |
| `tile_size_bwd_sm120.h` | kBlockM 64→128 |
| `bwd_optimization_analysis.md` | This file |
