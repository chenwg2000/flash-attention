# SM120 Backward Kernel Optimization Analysis

## Current Performance Gap

| Config (4,2048,32,128,F) | SM120 FP8 | SDPA BF16 | Gap |
|---|---|---|---|
| Forward | 0.796 ms (347 TFLOPS) | 1.367 ms (200 TFLOPS) | **1.7x faster** |
| Backward | 8.949 ms (77 TFLOPS) | 3.590 ms (193 TFLOPS) | **2.5x slower** |

The forward is great. The backward is the problem. With 5 GEMMs at FP8 (2x throughput over BF16),
the backward *should* be ~2.5x faster than BF16, not 2.5x slower. That's a **6x efficiency gap**
between where we are and where we should be.

## Per-M-Block Time Budget

Grid: 2048 CTAs, 170 SMs, ~12 waves × 32 M-blocks = **~23 µs per M-block**

| Phase | Time (µs) | % of total | Description |
|---|---|---|---|
| **5 GEMMs (MMA+LDSM)** | ~5-6 | 25% | GEMM-1,2 (4 k-iter), GEMM-3,4 (2 k-iter), GEMM-5 (4 k-iter) |
| **atomicAdd dQ** | **~4-6** | **20-25%** | 8192 individual atomicAdds to GMEM, massive L2 contention |
| **Phase D+H+F** | ~3-4 | 15% | Q transpose + dO→FP8 + dO→dO^T FP8 |
| **Phase E+LM scatters** | ~2-3 | 10% | P→FP8→SMEM + dS→FP8→SMEM×2, with 48KB zero-fill |
| **Register spills** | ~2-3 | 10% | 1624B stores + 1744B loads per thread (pre-existing!) |
| **Phase A loads** | ~1-2 | 5% | Cooperative GMEM→SMEM for Q, SFQ, dO |
| **Softmax + mask** | ~1 | 5% | Phase C: exp2f, comparisons |
| **6 syncs** | ~1 | 5% | __syncthreads overhead |

**The MMA tensor core is idle ~75% of the time.** The bottleneck is memory operations between GEMMs.

## What cuDNN Does Differently

cuDNN's Blackwell bwd achieves 193 TFLOPS (BF16) because:
1. **No FP8 conversion overhead** — BF16 operands go directly to GMMA, no scatter/transpose/convert
2. **TMEM on sm_100** — tensor memory keeps intermediates on-chip (SM120 lacks TMEM)
3. **TMA bulk operations** — async DMA for all loads and stores, including atomic reductions
4. **Warp specialization** — producer/consumer model hides memory latency

## What the SM90 FA3 Backward Gets Right

The SM90 (Hopper) backward kernel uses:
```
SM90 backward: 1 warpgroup (128 threads) PRODUCES — async TMA loads
               2 warpgroups (256 threads) CONSUME — GMMA compute
               dQ: SM90_BULK_REDUCE_ADD (hardware atomic reduce DMA)
```

The critical difference: **SM90 uses `cp.reduce.async.bulk.add.f32`** for dQ accumulation.
This is a single hardware DMA instruction that atomically reduces an entire SMEM buffer (16-32KB)
into GMEM. It replaces thousands of individual `atomicAdd` calls.

The instruction is in `hopper/copy_sm90_bulk_reduce.hpp`:
```cpp
// cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32
SM90_BULK_REDUCE_ADD::copy(smem_ptr, gmem_ptr, store_bytes, cache_hint);
```
This PTX instruction (SM90+) works on SM120 since `__CUDA_ARCH__ >= 900` is satisfied.

## Failed Optimization: Warp Specialization (attempted and reverted)

Attempted 4+4 split (compute warps 0-3 do GEMMs, memory warps 4-7 do D+H overlap).
All variants were net-negative:
- **Full warp spec**: 4-8% slower. D+H with 128 threads bottlenecks, +32B spills.
- **GEMM skip only**: TC pipeline is more efficient with 8 warps (even wasted MMAs) than 4.
- **Zero-fill elimination only**: 1-2% slower. Vectorized uint4 zero-fill + `continue` for
  warps 4-7 is faster than per-byte conditional writes for all warps.
- **Root cause**: 255 registers, pre-existing 1624B spills, TC saturates better with 8 warps.

---

# Optimization Plan (Priority Order)

## Step 1: Replace atomicAdd with TMA Bulk Reduce for dQ — TESTED, ~neutral

**RESULT: Net neutral.** RTX 5090 Blackwell L2 handles atomicAdd very efficiently
(~2 cycles amortized per atomicAdd). The extra overhead from scatter + sync + per-row DMA
roughly cancels the savings. Causal configs improve ~1%, non-causal regresses ~1-3%.

### Original Problem Estimate (overestimated)
8192 individual `atomicAdd` calls from 128 threads. Originally estimated ~4-6 µs per M-block,
but actual Blackwell L2 performance makes this closer to ~100-200 ns (2 cycles × 64 ops / warp).

### Solution
Use `SM90_BULK_REDUCE_ADD::copy()` — the `cp.reduce.async.bulk.add.f32` PTX instruction.
Available on SM120 (PTX ISA 8.0, SM90+).

### Flow
```
GEMM-5 → dQ in FP32 registers (acc_dq_rc, 64×128 elements across 256 threads)
→ Scatter dQ to SMEM:
    - Rows 0-31 → smem_pds (reinterpreted as float[32×128] = 16KB)
    - Rows 32-63 → smem_do (reinterpreted as float[32×128] = 16KB)
    Both buffers are free after GEMM-5 finishes reading them.
→ __syncthreads() (ensure scatter visible)
→ Thread 0:
    cp.reduce.async.bulk (smem_pds → GMEM dq_accum[m_start:m_start+32], 16KB)
    cp.reduce.async.bulk (smem_do  → GMEM dq_accum[m_start+32:m_start+64], 16KB)
    tma_store_arrive()
→ GEMM-4 (reads smem_q + smem_pt — no conflict with smem_pds/smem_do)
→ tma_store_wait<0>() before sync #6
```

### Timing Estimate
- Scatter: 128 threads × 32 FP32 stores each ≈ 50 ns
- Bulk DMA: 2 × 16KB atomic reduce ≈ 200-500 ns
- Total: ~600 ns vs current ~4-6 µs
- **5-10x speedup on dQ accumulation**

### SMEM Requirements
No additional SMEM — reuses smem_pds (16KB) + smem_do (16KB) after they're consumed by GEMM-5.
Data written as row-major FP32 (no swizzle needed for bulk reduce).

### GMEM Layout Requirement
`dq_accum` must be row-contiguous within each (batch, head, m_block). The Python interface allocates:
```python
dq_accum = torch.empty({batch_size, num_heads, seqlen_q_rounded * head_size_rounded}, dtype=float32)
```
With row_stride = head_dim = 128, rows are contiguous. 32 × 128 × 4 = 16KB per chunk. ✓

### Key Constraint
`cp.reduce.async.bulk` requires:
- SMEM pointer: 128-byte aligned (our buffers are 1024-byte aligned) ✓
- GMEM pointer: 16-byte aligned (torch tensor allocation guarantees this) ✓
- store_bytes: multiple of 16 (16384 = 16KB) ✓

### Code Changes
- File: `mainloop_bwd_sm120_tma_mma.hpp`
- Add: `#include "copy_sm90_bulk_reduce.hpp"`
- Modify: GEMM-5 section (replace atomicAdd loop with scatter + bulk reduce)
- Modify: sync #6 (add tma_store_wait<0>() for thread 0)

### Sync Structure (unchanged count)
```
Current:                          New:
  GEMM-5 → atomicAdd loop          GEMM-5 → scatter to SMEM
  GEMM-4                            __syncthreads()  (replaces implicit atomicAdd ordering)
  sync #6                           bulk_reduce_add × 2
                                    GEMM-4
                                    tma_store_wait<0>()
                                    sync #6
```
Still 6 syncs (the new __syncthreads after scatter replaces the implicit ordering
from atomicAdd, and sync #6 absorbs the tma_store_wait).

---

## Step 2: Increase kBlockM from 64 to 128 — IMPLEMENTED, 8-22% speedup

**RESULT: 8-22% backward speedup.** Combined with eliminating smem_do (read dO from GMEM/L2).

### Solution: Eliminate smem_do + kBlockM=128
RTX 5090 has only 100KB SMEM. kBlockM=128 with smem_do needs ~114KB (too much).
Key insight: dO is only read twice (Phase H: dO→FP8, Phase F: dO→dO^T FP8).
Reading dO from GMEM (L2 cached) instead of SMEM eliminates smem_do entirely.
SMEM: 98KB → 82KB. kBlockM=128 fits easily.

### Changes
- `tile_size_bwd_sm120.h`: kBlockM 64→128
- `mainloop_bwd_sm120_tma_mma.hpp`:
  - Removed smem_do from TensorStorage (saved 16KB SMEM)
  - New GMEM-reading functions: `convert_gmem_bf16_to_fp8`, `transpose_gmem_bf16_to_fp8`
  - Phase A: removed dO SMEM load + smem_q zero-fill (kBlockM=128=MMA tile, no padding)
  - Phase C: simplified mask (only seqlen_q check, no kBlockM padding check)
  - Phase E: removed zero-fill + row guard (all 128 rows valid)
  - Phase K: removed padding guard
  - Phase LM: removed zero-fill + row guard
  - GEMM-3/4: full k-loop (kIters34=4, was 2 with halving)
  - GEMM-5: simplified atomicAdd row guard
  - SmemLayoutPdS: changed to FP8 atom (was BF16, over-allocated)

### Results (bwd-only speedup vs kBlockM=64 baseline)
| Config | Old bwd (ms) | New bwd (ms) | Speedup |
|---|---|---|---|
| (1,512,32,128,F) | 0.214 | 0.177 | 17% |
| (1,1024,32,128,F) | 0.750 | 0.630 | 16% |
| (1,2048,32,128,F) | 2.671 | 2.212 | 17% |
| (1,4096,32,128,F) | 9.248 | 8.539 | 8% |
| (4,2048,32,128,F) | 8.949 | 8.848 | 1% |
| (1,2048,32,128,C) | 1.573 | 1.241 | 21% |
| (1,4096,32,128,C) | 4.971 | 3.854 | 22% |
| (4,2048,32,128,C) | 5.211 | 4.179 | 20% |

- Spills decreased: 1624/1744 → 1440/1440 (simpler code helps regalloc)
- Best combined TFLOPS: 111.8 at (1,4096,32,128,C), up from 88.5
This is a hard hardware limit on the consumer RTX 5090.

### Problem
kBlockM=64 but MMA tile needs 128 rows (Blk_MN=128 constraint). This causes:
- 50% wasted MMA in GEMMs 1, 2, 5 (warps 4-7 compute on zeros)
- 48KB of zero-fills per M-block (Phase A: 16KB, Phase E: 16KB, Phase LM: 32KB)
- Half the M-block work → twice the M-block overhead

### Solution
Set kBlockM=128. All 8 warps compute on real data. No zero-fills needed.

### Impact
- M-blocks: 2048/128 = 16 instead of 2048/64 = 32 (half iterations)
- Zero-fills eliminated: saves ~48KB SMEM writes per M-block
- GEMM-3/4 k-loop: goes from 2 to 4 iterations (same total work, fewer M-blocks)
- All warps useful in all GEMMs (no wasted MMA)

### SMEM Impact
- smem_do grows: 64×128 BF16 (16KB) → 128×128 BF16 (32KB)
- Total: 98KB → 114KB (fits in SM120's 228KB max)
- May need `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 114*1024)`

### Caution
- atomicAdd for dQ doubles (128 rows instead of 64) — **do Step 1 first!**
- Phase D+H work doubles (128×128 transpose + convert instead of 64×128)
- Phase F work doubles (128×128 BF16→FP8 transpose)
- GMEM loads double for dO (32KB instead of 16KB)

### Code Changes
- `mainloop_bwd_sm120_tma_mma.hpp`: Change kBlockM references, remove zero-fills,
  remove `if (row >= kBlockM) continue` guards, adjust Phase D+H+F for 128 rows
- `flash_bwd_kernel_sm120.h`: Adjust tile shape
- Bwd launch template: Update tile size selection
- `flash_attn_interface.py`: Adjust dq_accum allocation if needed

---

## Step 3: TMA Async Loads for Q/dO + M-Block Pipelining (~5-10% additional)

### Problem
Phase A uses cooperative 256-thread loads. All threads blocked during GMEM→SMEM transfer.

### Solution
Replace with single-thread TMA `SM90_TMA_LOAD` (works on SM120). Enable double-buffering
across M-blocks: load Q[m+1]/dO[m+1] while computing on m.

### Flow
```
Prologue: TMA Q[0] + dO[0]
M-block 0:
  Wait for Q[0]/dO[0]
  Start TMA Q[1]/dO[1] (after Phase D consumes Q[0])
  Compute all GEMMs for M-block 0
  Wait for TMA Q[1]/dO[1] (should already be done)
M-block 1:
  Start TMA Q[2]/dO[2]
  Compute all GEMMs for M-block 1
  ...
```

### SMEM Impact
+32KB for double-buffered Q+dO. Total: ~146KB (with kBlockM=128). Under 228KB limit.

### Code Changes
- Add TMA descriptors for Q and dO (like forward kernel does for K/V)
- Add PipelineTmaAsyncNoCluster<2> for Q/dO pipeline
- Restructure Phase A to use TMA instead of cooperative loads

---

## Step 4: Reduce Register Spills (~5-10% additional)

### Problem
1624B spill stores + 1744B spill loads per thread. This is pre-existing (not caused by any
optimization attempt). ~406 FP32 values spilled per thread.

### Potential Solutions
After Steps 1-3:
- Step 1 removes atomicAdd loop variables
- Step 2 removes zero-fill loops, padding checks, and simplifies scatter logic
- Step 3 removes cooperative load loops
- Combined: may reduce code complexity enough for compiler to find better regalloc

Additionally:
- Consider splitting the M-block body into separate `__noinline__` functions to help regalloc
- Consider using `__launch_bounds__(256, 1)` more aggressively
- Profile with Nsight to identify which specific variables spill

---

## Expected Combined Impact

| Step | Mechanism | Result |
|---|---|---|
| Step 1: TMA Bulk Reduce dQ | Eliminate atomicAdd contention | ~neutral (Blackwell L2 fast) |
| Step 2: kBlockM=128 + GMEM dO | Eliminate padding, halve M-blocks | **8-22% faster** ✅ |
| Step 3: TMA pipeline Q/dO | Async loads, double-buffer | Not yet attempted |
| Step 4: Spill reduction | Simpler code → better regalloc | Partially achieved (1624→1440) |

Current bwd (4,2048,32,128,F): 8.85 ms (was 8.95 ms)
Current bwd (1,4096,32,128,C): 3.85 ms = 111.8 TFLOPS (was 4.97 ms = 88.5)
Theoretical ceiling: ~2.8 ms (~250 TFLOPS)

## Implementation Order

1. **Step 1 first** — most focused change, highest impact/effort ratio, no SMEM layout changes
2. **Step 2 after Step 1** — bigger refactor, but Step 1 mitigates the doubled atomicAdd cost
3. **Step 3 after Step 2** — requires SMEM layout redesign, best done as part of Step 2
4. **Step 4** — may happen naturally from Steps 1-3 simplifying the code

## Files to Modify

| File | Step | Changes |
|---|---|---|
| `mainloop_bwd_sm120_tma_mma.hpp` | 1,2,3,4 | Main kernel logic |
| `flash_bwd_kernel_sm120.h` | 2 | Tile shape, SMEM allocation |
| `copy_sm90_bulk_reduce.hpp` | 1 | Include (already exists) |
| `flash_bwd_launch_template_sm120.h` | 2,3 | Launch config, TMA descriptors |
| `flash_attn_interface.py` | 2 | dq_accum sizing if kBlockM changes |
| `test_bwd_sm120.py` | all | Verify correctness after each step |
| `bench_fwd_bwd_sm120.py` | all | Benchmark after each step |
