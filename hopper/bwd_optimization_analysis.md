# SM120 Backward Kernel Optimization Analysis (Updated)

## Current State (after kBlockM=128 + GMEM dO)

| Config | SM120 FP8 bwd | SDPA BF16 bwd | Gap |
|---|---|---|---|
| (1,2048,32,128,F) | 2.212 ms (97.6 TFLOPS f+b) | 0.949 ms (176 TFLOPS f+b) | 2.3x slower |
| (4,2048,32,128,F) | 8.848 ms (99.8 TFLOPS f+b) | 3.590 ms (194 TFLOPS f+b) | 2.5x slower |
| (1,4096,32,128,C) | 3.854 ms (111.8 TFLOPS f+b) | 1.825 ms (185 TFLOPS f+b) | 2.1x slower |
| (4,2048,32,128,C) | 4.179 ms (103.3 TFLOPS f+b) | 2.098 ms (169 TFLOPS f+b) | 2.0x slower |

Forward achieves 347 TFLOPS (1.7x faster than BF16). Backward is the entire gap.

## What We've Tried and Learned

| Optimization | Result | Key Learning |
|---|---|---|
| Warp specialization (4+4 split) | -4 to -8% (worse) | TC saturates better with 8 warps (even wasted). 255 regs = no branch headroom. |
| Zero-fill elimination | -1 to -2% (worse) | Vectorized uint4 zero-fill + `continue` beats per-byte conditional writes. |
| TMA bulk reduce for dQ | ~neutral | Blackwell L2 atomicAdd is fast (~2 cycles amortized). Not the bottleneck. |
| **kBlockM=128 + GMEM dO** | **+8 to +22%** ✅ | Big win from halving M-blocks + eliminating all padding/zero-fill waste. |

Hardware constraints discovered:
- RTX 5090 (SM 12.0): **only 100 KB SMEM per SM** (not 228 KB like Hopper/B100)
- Pre-existing register spills: 1440B stores + 1440B loads (improved from 1624/1744)
- Block-scaled MMA Blk_MN=128 forces 128-row MMA tile regardless of kBlockM

## Where Time Goes Now (per M-block, estimated)

Config: (1,2048,32,128,F), 16 M-blocks per CTA, ~46 µs per M-block

The forward achieves ~5.3 µs per N-block (2 GEMMs + softmax + TMA-pipelined K/V loads).
The backward takes ~46 µs per M-block (5 GEMMs + 5 convert/scatter phases + 6 syncs).

### Why the backward is 8.7x slower per-tile than the forward (with 2.5x more GEMMs)

The 3.5x overhead ratio (8.7x / 2.5x) breaks down into:

**1. The FP32→FP8→SMEM→LDSM round-trip chain (~60% of overhead)**

The core bottleneck. Between every pair of GEMMs, the backward must:
```
GEMM output: FP32 in registers
    → Convert FP32 to FP8 (per-element cast)
    → Scatter FP8 to swizzled SMEM (per-byte writes with swizzle formula)
    → __syncthreads() (wait for all warps)
    → LDSM from swizzled SMEM to registers (next GEMM input)
    → Execute next GEMM
```

This chain happens 4-5 times per M-block:
- P scatter (Phase E): acc_s FP32 → FP8 transposed → smem_q → GEMM-3
- dO→FP8 (Phase H): GMEM BF16 → FP8 → smem_pds → GEMM-2
- dO→dO^T (Phase F): GMEM BF16 → FP8 transposed → smem_pds → GEMM-3
- dS dual scatter (Phase LM): acc_s FP32 → FP8 to smem_q + smem_pds → GEMM-4/5
- Q→Q^T (Phase D): FP8 SMEM → FP8 transposed SMEM → GEMM-4

Each round-trip: 16384 elements / 256 threads = 64 ops per thread + sync overhead.
The forward has only 1 such round-trip (P scatter) because K/V are pre-loaded via TMA.

**2. No compute-memory overlap (~20% of overhead)**

The forward uses TMA async pipelining: K[n+1]/V[n+1] load overlaps with GEMM[n].
The backward loads Q synchronously (all threads blocked during GMEM→SMEM), and every
memory phase (D, H, E, F, LM) forces the TC to idle during the sync.

6 syncs × ~50 ns each = 300 ns direct cost, but the REAL cost is the serialization:
the TC can't start the next GEMM until the previous memory phase + sync completes.

**3. Register spills (~10% of overhead)**

1440B stores + 1440B loads per thread = 360 FP32 spills. At ~10-20 cycles each through L1,
this adds ~0.5-1 µs per M-block. Less than before (1624B) but still significant.

**4. GMEM dO reads (~10% of overhead)**

Phase H and Phase F each read 128×128 BF16 = 32KB from GMEM. Phase H is an L2 miss
(first access); Phase F hits L2 cache. Transposed access in Phase F causes suboptimal
cache line utilization.

## The Fundamental Problem: FP8 Conversion Tax

The core issue is **architectural**: SM120's block-scaled WGMMA requires FP8 operands
loaded from SMEM via LDSM. Every intermediate result (P, dP, dS, dO_fp8, dO^T, Q^T)
must go through the expensive conversion pipeline:

```
Register (FP32) → Convert → Scatter to swizzled SMEM → sync → LDSM → Register → MMA
```

cuDNN's BF16 approach on Blackwell avoids this because:
1. BF16 intermediates don't need per-block scale factors (no identity SF overhead)
2. cuDNN on SM100 uses TMEM (tensor memory) to keep intermediates on-chip
3. Descriptor-based WGMMA can read SMEM without explicit LDSM
4. cuDNN likely uses warp-specialized producer/consumer model

On SM120 (consumer Blackwell), we don't have TMEM, and our per-warp MMA requires
explicit LDSM. The FP8 conversion tax is unavoidable with the current MMA approach.

## Next Steps to Close the Gap (ranked by expected impact)

### Option A: Fuse Phase D+H into overlapped execution (medium impact, ~5-10%)

Currently Phase D (Q→Q^T) and Phase H (dO→FP8) happen sequentially before sync #2.
But Phase D reads smem_q while Phase H reads GMEM (dO) — they don't conflict on SMEM.
They could run truly in parallel if we overlap them within the same code block,
reducing the wall-clock time for this phase by ~30-40%.

Additionally, Phase H could start BEFORE sync #1 completes, because Phase H reads
from GMEM (not SMEM). Only Phase D needs to wait for sync #1 (Q in smem_q).

### Option B: TMA async Q loads + M-block pipelining (medium-high impact, ~10-15%)

Replace cooperative Q load with single-thread TMA. Double-buffer smem_q across M-blocks:
load Q[m+1] via TMA during M-block m's compute phases. This hides Phase A latency
entirely and frees 255 threads from load duty.

SMEM cost: +16KB for double-buffered Q. Total: 82+16=98KB. Fits in 100KB.

This is the same pattern the forward kernel uses for K/V pipelining.

### Option C: Reduce sync count by reordering phases (medium impact, ~5-10%)

Current: 6 syncs per M-block. Each sync serializes compute and memory.
Target: 4-5 syncs by fusing compatible phases.

Potential merges:
- Merge sync #3 with Phase E scatter (Phase E writes smem_q, Phase F writes smem_pds — disjoint)
- Move Phase D earlier (overlap with GEMM-1 for compute warps, memory for others)

### Option D: Reduce register spills via code restructuring (low-medium impact, ~3-5%)

Split the M-block body into smaller `__noinline__` functions to give the compiler
smaller register allocation scopes. Or use `#pragma unroll 1` on the M-block loop
to reduce code duplication.

### Option E: Vectorize the FP8 scatter operations (medium impact, ~5-10%)

Currently each scatter writes one FP8 byte at a time with per-byte swizzle computation.
If we restructure the scatter to write 4 or 8 bytes at a time (uint32_t or uint64_t),
the SMEM write throughput would improve by 4-8×.

This requires grouping consecutive register elements that map to consecutive SMEM
addresses after swizzle. With the 128-byte swizzle pattern (Swizzle<3,4,3>),
consecutive elements within a 16-byte chunk share the same swizzle offset.
This means we CAN vectorize within 16-byte groups.

### Option F: Switch GEMMs 2-5 to BF16 MMA (exploratory, uncertain impact)

GEMMs 2-5 all use identity scale factors — the block scaling adds zero value.
BF16 mma.sync (SM80-style) has half the throughput but:
- BF16 → SMEM writes are 2 bytes (vs 1 byte FP8), but no FP32→FP8 conversion needed
- BF16 intermediates (P, dS) could potentially stay in BF16, skipping the FP32→FP8→BF16 chain
- Simpler scatter (BF16 to BF16 SMEM, no FP8 conversion)

Trade-off: 2× lower MMA throughput vs ~2× less conversion overhead. Net uncertain.

## Estimated Impact and Priority

| Option | Impact | Effort | Risk |
|---|---|---|---|
| B: TMA Q pipeline | 10-15% | High | Low (proven pattern from fwd) |
| E: Vectorized scatter | 5-10% | Medium | Medium (swizzle alignment) |
| A: Fuse D+H overlap | 5-10% | Low | Low |
| C: Reduce syncs | 5-10% | Medium | Medium (correctness) |
| D: Reduce spills | 3-5% | Low | Low |
| F: BF16 GEMMs 2-5 | Uncertain | Very High | High (full rewrite) |

**Recommended next step: Option B (TMA Q pipelining)** — it's the single change most
likely to give double-digit improvement. It eliminates the Phase A load stall (~0.5µs/M-block)
and, more importantly, overlaps the next M-block's Q load with the current M-block's
compute phases.

Combined B+A+E could potentially give 20-30% improvement, bringing the gap from
~2.3x to ~1.8x vs BF16.

## Theoretical Analysis: Can We Ever Match BF16?

**Short answer: probably not on SM120, but we can get much closer.**

The fundamental FP8 conversion tax (5 scatter/convert round-trips per M-block) adds
~15-20 µs of overhead per 46 µs M-block. Even if we perfectly overlap all memory
operations with compute, the FP8→SMEM→LDSM pipeline latency cannot be hidden because
each GEMM's output must be converted before the next GEMM can start (data dependency).

The theoretical minimum backward time on SM120 (all memory hidden, zero sync overhead):
- Pure GEMM time at forward efficiency: ~0.64 ms for (1,2048,32,128,F)
- FP8 conversion pipeline latency (serial, cannot be hidden): ~0.3-0.5 ms
- **Theoretical best: ~1.0-1.1 ms** vs BF16's 0.95 ms → roughly parity

Current: 2.21 ms. Theoretical best: ~1.0 ms. There's still 2× room for improvement,
but it requires near-perfect memory/compute overlap, which likely needs architectural
support (TMEM, descriptor-based MMA) that SM120 doesn't have.

## Files Modified

| File | Changes |
|---|---|
| `mainloop_bwd_sm120_tma_mma.hpp` | kBlockM=128, GMEM dO reads, no zero-fills/padding |
| `tile_size_bwd_sm120.h` | kBlockM 64→128 |
| `bwd_optimization_analysis.md` | This file |
