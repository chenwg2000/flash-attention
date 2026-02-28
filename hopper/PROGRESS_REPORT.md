# SM120 MXFP8 Flash Attention — Progress Report

**Date:** February 28, 2026
**Target:** NVIDIA RTX 5090 (sm_120, Blackwell consumer, 170 SMs)
**Environment:** CUDA 13.0, PyTorch 2.10.0+cu130, CUTLASS (submodule)

---

## 1. Executive Summary

We have built a **custom FP8 flash attention forward kernel** for the RTX 5090 (sm_120) from scratch inside the Flash Attention 3 codebase. The kernel uses SM120's native **block-scaled MXFP8 MMA** hardware (16x8x32 atoms with UE8M0 per-32-element scaling) and achieves **345 TFLOPS non-causal / 290 TFLOPS causal** on the key benchmark configuration (batch=4, seqlen=2048, 32 heads, hdim=128) — **1.72x / 1.58x faster than PyTorch SDPA BF16**.

This was necessary because:
- FA3's existing SM90 C++ kernels use `sm_90a` GMMA, which does not run on SM120.
- FA3's CuTe DSL path fails on SM120 because CUTLASS DSL v4.4.0's libNVVM cannot lower WGMMA for the `sm_120a` target.
- No existing open-source flash attention implementation supports SM120's block-scaled FP8 MMA.

---

## 2. Architecture Background

### SM120 vs SM90 vs SM100

| Feature | SM90 (Hopper) | SM100 (Blackwell DC) | SM120 (Blackwell Consumer) |
|---------|--------------|---------------------|---------------------------|
| GMMA (warp-group MMA) | Yes | Yes | No |
| WGMMA | Yes | Yes | No (warp-level only) |
| TMA (Tensor Memory Accelerator) | Yes (cluster-scoped) | Yes | Yes (CTA-scoped) |
| TMEM (Tensor Memory) | No | Yes | No |
| Block-scaled MXFP8 MMA | No | Yes | Yes |
| Warp specialization | Yes (producer/consumer WGs) | Yes | No (all threads compute) |

SM120 occupies a unique position: it has TMA and block-scaled FP8 MMA (like SM100), but lacks GMMA, TMEM, and warp specialization (unlike SM90/SM100). This means existing FA3 kernels for either architecture cannot run on SM120 without modification.

### Block-Scaled MXFP8 MMA

SM120's MMA atom is `SM120_16x8x32_TN_VS`:
- **Tile:** 16 rows x 8 columns x 32 K-elements per atom
- **Operands:** FP8 e4m3 data + UE8M0 scale factors (one scale per 32 elements)
- **Accumulator:** FP32
- **Interface:** `make_zip_tensor(data_fragment, scale_fragment)` — the MMA unpacks the zip internally
- **Constraint:** FP8 on SM120 *only* works with block scaling (no unscaled FP8 MMA exists)

---

## 3. Kernel Architecture

### Tile Configuration

| Parameter | Value |
|-----------|-------|
| kBlockM | 128 |
| kBlockN | 128 |
| kHeadDim | 128 |
| kStages | 2 (double-buffered K/V) |
| Warps | 8 (all along M, 1 along N) |
| Threads | 256 |
| Registers | 255 (0 spills) |
| SMEM usage | ~86 KB of 101 KB limit |

### Data Flow Per N-Block Iteration

```
TMA Prologue: prefetch first 2 K/V blocks into SMEM stages 0,1
              (thread 0 issues TMA, pipeline barriers track completion)

N-Block Loop (n = n_block_max-1 down to 0):
  +---------------------------------------------------------+
  | 1. SFK cooperative load (GMEM -> SMEM, 512 bytes)       |
  | 2. consumer_wait (pipeline barrier, waits for K+V TMA)  |
  | 3. __syncthreads                                        |
  +---------------------------------------------------------+
  | 4. GEMM-I: S = Q @ K^T  (block-scaled MMA, K=128)      |
  |    - Q in registers (loaded once before loop via LDSM)  |
  |    - K from swizzled SMEM stage[n%2] via TMA+LDSM      |
  |    - Scale factors: SFQ (registers), SFK (SMEM)         |
  +---------------------------------------------------------+
  | 4b. Causal mask (if Is_causal && diagonal block):       |
  |     set acc_s to -inf where col > row                   |
  +---------------------------------------------------------+
  | 5. Online Softmax (FP32 max/sum/exp in registers)       |
  +---------------------------------------------------------+
  | 6. P scatter: FP32->FP8, store to swizzled SMEM         |
  | 7. V transpose: smem_v -> swizzled smem_vt (__byte_perm)|
  | 8. Identity SF fill for SFP + SFVt                      |
  | 9. __syncthreads                                        |
  +---------------------------------------------------------+
  | 10. GEMM-II: O += P @ V^T  (block-scaled MMA, K=128)   |
  |     - P from swizzled SMEM via LDSM (reuses Q buffer)  |
  |     - V^T from swizzled SMEM via LDSM (reuses K buffer)|
  |     - Scale factors: SFP, SFVt (identity = 1.0)        |
  +---------------------------------------------------------+
  | 11. __syncthreads                                       |
  | 12. consumer_release (pipeline: release current stage)  |
  | 13. producer_acquire + TMA issue for next block         |
  |     (async DMA overlaps with next iteration's compute)  |
  +---------------------------------------------------------+

Epilogue: normalize O by softmax sum, write O (BF16) + LSE (FP32)
```

### SMEM Layout — Swizzled + Aggressive Overlapping

All MMA data operands use **SW128 swizzled layouts** (`GMMA::Layout_K_SW128_Atom<uint8_t>` = `Swizzle<3,4,3>`) for bank-conflict-free LDSM (ldmatrix) access. Scale factors remain non-swizzled (separate block-scaled format).

```
+---------------------------+  offset 0
| smem_q / smem_p           |  16 KB  (swizzled; Q loaded once, reused as P)
+---------------------------+  offset 16K
| smem_k / smem_vt          |  32 KB  (K: swizzled, TMA; V^T: swizzled, transposed)
+---------------------------+  offset 48K
| smem_v                    |  32 KB  (non-swizzled; raw reads for byte transpose)
+---------------------------+  offset 80K
| smem_sfq                  |  ~0.5 KB
+---------------------------+
| smem_sfk / smem_sfp       |  ~1 KB  (SFK for GEMM-I, identity for GEMM-II)
| smem_sfv / smem_sfvt      |  ~1 KB  (identity fill only)
+---------------------------+
| pipeline barriers         |  ~256 bytes (PipelineTmaAsyncNoCluster<2>)
+---------------------------+  ~86 KB total
```

### Causal Masking

For causal attention:
- **N-block early termination:** `n_block_max = min(ceil_div(seqlen_k, kBlockN), ceil_div(m_start + kBlockM, kBlockN))` skips ~50% of N-blocks on average
- **Diagonal block mask:** Per-element check `col > row` sets masked positions to `-inf` before softmax. Only 1 block per M-row needs masking; all others are fully below the diagonal.

---

## 4. Implementation Timeline

### Phase 1: Kernel Skeleton (GEMM-I only)
- Created 5 new files: `tile_size_sm120.h`, `mainloop_fwd_sm120_tma_mma.hpp`, `flash_fwd_kernel_sm120.h`, `flash_fwd_launch_template_sm120.h`, instantiation `.cu` files
- Modified 6 existing files: `setup.py`, `flash.h`, `static_switch.h`, `utils.h`, `flash_api_stable.cpp`, `flash_attn_interface.py`
- Implemented GEMM-I (Q@K^T) with SM120 block-scaled MMA
- Resolved 6 compilation issues (SF SMEM layout, copy atom size, gencode flags, ODR-use, etc.)

### Phase 2: Full Forward Pass (GEMM-I + Softmax + GEMM-II)
- Implemented online softmax with rescaling
- Implemented GEMM-II: P (FP32->FP8) scatter to SMEM, V transpose, block-scaled MMA
- Achieved correctness: LSE exact match, O max diff < 0.013 vs reference
- **Baseline performance: ~73 TFLOPS**

### Phase 3: TMA + Pipeline Optimization
- Replaced cooperative `load_tile()` with TMA loads (`SM90_TMA_LOAD` via `shared::cta` barriers)
- Added `PipelineTmaAsyncNoCluster<2>` double-buffered pipeline
- Vectorized V transpose (4x4 block via `__byte_perm`: 4x fewer SMEM ops)
- Removed barrier between P scatter and V transpose (separate SMEM regions)
- Removed wasteful SFV cooperative load (immediately overwritten by identity fill)
- Vectorized identity SF fill (uint4, 16 bytes/iteration)
- **Performance: ~96 TFLOPS (+32%)**

### Phase 4: Swizzled SMEM + LDSM
- Replaced all `UniversalCopy<uint8_t>` (1 byte/thread/instruction) with LDSM (ldmatrix, 8-16 bytes/thread/instruction) for all four MMA operand fetches (Q, K, P, V^T)
- All data SMEM layouts now use `SM90::GMMA::Layout_K_SW128_Atom<uint8_t>` (`Swizzle<3,4,3>`)
- GEMM-I: `SM75_U32x4_LDSM_N` for Q (A-operand), `SM75_U32x2_LDSM_N` for K (B-operand)
- GEMM-II: `SM75_U32x4_LDSM_N` for P (A-operand), `SM75_U32x2_LDSM_N` for V^T (B-operand)
- Q loaded via `load_tile_swizzled()` with manual SW128 XOR formula
- K TMA carries swizzle in TMA descriptor (hardware applies SW128 automatically)
- V stays non-swizzled in SMEM (source for byte-level transpose); V^T written swizzled
- P scatter and V transpose apply swizzle formula `addr ^ (((addr >> 7) & 7) << 4)` to destination addresses
- Separate TMA_K (swizzled) and TMA_V (non-swizzled) descriptor types
- `as_position_independent_swizzle_tensor` for all swizzled partitions
- **Performance: GEMM-I only: ~177 TFLOPS (+84%); both GEMMs: ~345 TFLOPS (+96%)**

### Phase 5: Causal Masking
- Compute reduced `n_block_max` for causal to skip fully-masked N-blocks above the diagonal
- Apply per-element `-inf` mask on the diagonal block after GEMM-I, before softmax
- Only 1 block per M-row needs masking; remaining blocks are fully valid
- **Causal performance: 172 -> 290 TFLOPS (+68%)**

### Phase 6: MXFP8 API + Scale Factor Fix
- **Wired `flash_attn_mxfp8_func`** to the kernel via a dedicated `fwd_mxfp8` C++ op
  - Lean entry point: Q/K/V (FP8) + q_scale/k_scale/v_scale (UE8M0) + softmax_scale + causal
  - Registered as `flash_attn_3.fwd_mxfp8` via `STABLE_TORCH_LIBRARY`
  - Populates `Flash_fwd_params` scale pointers and strides
- **Verified with non-identity scales** using kernel-vs-kernel cross-check:
  - `kernel(data, sf)` == `kernel(prescaled_data, identity_sf)` within LSE < 0.001
  - Requires small data to avoid FP8 overflow in prescaling (kernel applies SF in FP32 accumulator)
  - Uniform scales: exact match at all sequence lengths (sf=130 == identity+64x softmax_scale)
  - Per-row varying scales: LSE diff < 0.001 across configs up to (2,1024,4,128)
- **Comprehensive test suite** (9 tests, all passing) in `test_sm120.py`
- **V scale factors**: accepted by API but not applied in GEMM-II (identity P/V^T SFs).
  V scaling would require transposed SF support in the V^T SMEM layout.

---

## 5. Performance Results

### Correctness

| # | Test | Method | Result |
|---|------|--------|--------|
| 1 | Identity scales match legacy | `mxfp8(id)` == `_flash_attn_forward` | exact 0.0, 4 configs |
| 2 | Uniform non-identity scales | `sf=130` == `identity + 64x scale` | exact 0.0, s=128..2048 |
| 3 | Scale factors applied | `kernel(sf)` == `kernel(prescaled, id)` | LSE < 0.001, 4 configs |
| 4 | Per-row varying scales | prescale cross-check | LSE = 0.0005 |
| 5 | Causal + non-identity scales | prescale cross-check | LSE < 0.001, 2 configs |
| 6 | FP32 reference sanity | kernel vs FP32 dequantized ref | LSE < 0.05, O < 0.05 |
| 7 | Batch/head consistency | batched == individual | exact 0.0 |
| 8 | Extreme scale values (2^10) | finite output, LSE changes | PASS |
| 9 | V scales not applied | documented: identity P/V^T in GEMM-II | exact 0.0 |

Test methodology: Scale factor correctness is verified by comparing `kernel(data, sf)` against `kernel(prescaled_data, identity_sf)`. Both paths should produce identical results. This requires small input data (~0.01 magnitude) to avoid FP8 overflow during prescaling, since the kernel applies SF in the FP32 MMA accumulator while prescaling goes through FP8 requantization (max 448).

### Performance (median of 100 runs)

| Config (b,s,h,d,causal) | SM120 FP8 | SDPA BF16 | SDPA FP16 | SM120 / SDPA |
|--------------------------|-----------|-----------|-----------|--------------|
| (1, 512, 32, 128, F) | 123 TFLOPS | 135 TFLOPS | 134 TFLOPS | 91% |
| (1, 1024, 32, 128, F) | 218 TFLOPS | 156 TFLOPS | 156 TFLOPS | **140%** |
| (1, 2048, 32, 128, F) | 270 TFLOPS | 165 TFLOPS | 164 TFLOPS | **164%** |
| (1, 4096, 32, 128, F) | **338 TFLOPS** | 190 TFLOPS | 190 TFLOPS | **178%** |
| (4, 512, 32, 128, F) | 185 TFLOPS | 150 TFLOPS | 150 TFLOPS | **123%** |
| **(4, 2048, 32, 128, F)** | **345 TFLOPS** | 201 TFLOPS | 200 TFLOPS | **172%** |
| (1, 2048, 32, 128, C) | **236 TFLOPS** | 145 TFLOPS | 144 TFLOPS | **163%** |
| (4, 2048, 32, 128, C) | **290 TFLOPS** | 183 TFLOPS | 183 TFLOPS | **158%** |

### Optimization Progression

| Optimization | TFLOPS (4,2048,32,128,F) | Delta |
|--------------|--------------------------|-------|
| Baseline (cooperative loads, byte transpose) | 73 | -- |
| + TMA loads + pipeline | 83.5 | +14% |
| + Barrier reduction (4->3 per N-block) | 84.1 | +1% |
| + Vectorized V transpose (__byte_perm 4x4) | 95.5 | +14% |
| + Remove SFV load + vectorize SF fill | 95.6 | +0.1% |
| + Swizzled SMEM + LDSM (GEMM-I only) | 177 | +85% |
| + Swizzled SMEM + LDSM (both GEMMs) | **345** | +95% |

| Optimization | TFLOPS (4,2048,32,128,C) | Delta |
|--------------|--------------------------|-------|
| Before causal masking (all N-blocks computed) | 172 | -- |
| + Causal early termination + diagonal mask | **290** | +68% |

---

## 6. Technical Challenges Overcome

### SM120 Block-Scaled MMA Discovery
FP8 e4m3 on SM120 **only** works with block scaling — there is no unscaled FP8 MMA atom. This required implementing the `make_zip_tensor(data, scale_factor)` pattern for both GEMM-I (Q@K^T) and GEMM-II (P@V^T), with custom scale factor partitioning functions (`sm120_thrfrg_SFA`, `sm120_thrfrg_SFB`).

### TMA on SM120
SM120 uses CTA-scoped TMA (`shared::cta`) rather than cluster-scoped (`shared::cluster`). The same `SM90_TMA_LOAD` copy operation works because CUTLASS's PTX emission switches based on `CUTE_ARCH_TMA_SM120_ENABLED`. The `PipelineTmaAsyncNoCluster` abstraction handles cluster-size-1 barrier semantics.

### Pipeline Synchronization with No Warp Specialization
SM90 FA3 uses separate producer (WG0) and consumer (WG1) warp groups. SM120 has all 256 threads in a single `ProducerConsumer` role. The empty barrier expects only 2 arrivals (one per warp group of 128 threads), so a `__syncthreads` before `consumer_release` is required to ensure all 8 warps finish reading SMEM before the stage is released for TMA reuse.

### SMEM Budget
With 128x128 tiles, double-buffered K/V, and scale factor buffers, SMEM usage reaches ~86 KB against a 101 KB limit. Aggressive union overlapping (Q/P, K/V^T, SFK/SFP, SFV/SFVt) was essential to fit within the budget while maintaining correct data flow between GEMM-I and GEMM-II phases.

### Swizzled SMEM + LDSM Integration
Replacing `UniversalCopy<uint8_t>` with LDSM (`SM75_U32x{2,4}_LDSM_N`) required:
- **SW128 swizzled layouts** for all data operands via `GMMA::Layout_K_SW128_Atom<uint8_t>` (`Swizzle<3,4,3>`)
- **Understanding CuTe's composed layout**: The `smem_ptr_flag_bits<8>` specialization of `upcast` preserves the swizzle unchanged; only the base layout is upcasted. The swizzle formula on byte addresses is `addr ^ (((addr >> 7) & 7) << 4)`.
- **Separate TMA descriptors**: K uses swizzled SMEM layout (TMA hardware applies SW128 automatically), V uses non-swizzled (source for byte-level transpose)
- **Manual swizzle for cooperative loads**: Q loaded via `load_tile_swizzled()` which applies the XOR formula during GMEM->SMEM copy
- **Swizzled writes for GEMM-II operands**: P scatter and V transpose apply the swizzle formula to each destination address. Safe because the XOR operates at 16-byte granularity, and our `uint16_t`/`uint32_t` writes never cross 16-byte block boundaries.
- **`as_position_independent_swizzle_tensor`**: Required to convert composed-layout tensors for CuTe `partition_S`. Requires 1024-byte SMEM alignment (`alignas(1024)`).
- **`partition_fragment_A/B` with flat layout**: Register fragments depend only on tile shape, not swizzle; using a flat (non-swizzled) layout avoids composed-layout issues in the MMA partition.
- **B-operand uses U32x2**: With 8x1x1 warp layout, the B-operand tile is only 8x32 = 256 bytes = 8 bytes/thread, requiring `SM75_U32x2_LDSM_N` instead of `SM75_U32x4_LDSM_N`.

### ODR-Use of Static Constexpr in Device Code
`static constexpr int kStages` triggers undefined symbol errors in CUDA device code on certain compiler configurations. Resolved by copying to a local variable: `int const num_stages = kStages_;`.

---

## 7. Python API

```python
from flash_attn_interface import flash_attn_mxfp8_func

# FP8 data + MXFP8 per-32-element UE8M0 scale factors
out, lse = flash_attn_mxfp8_func(
    q,          # (batch, seqlen_q, nheads, headdim), float8_e4m3fn
    k,          # (batch, seqlen_k, nheads_kv, headdim), float8_e4m3fn
    v,          # (batch, seqlen_k, nheads_kv, headdim), float8_e4m3fn
    q_scale,    # (batch, nheads, seqlen_q, headdim//32), uint8 UE8M0
    k_scale,    # (batch, nheads_kv, seqlen_k, headdim//32), uint8 UE8M0
    v_scale,    # (batch, nheads_kv, seqlen_k, headdim//32), uint8 UE8M0
    softmax_scale=1.0 / math.sqrt(headdim),
    causal=True,
)
# out: (batch, seqlen_q, nheads, headdim) bfloat16
# lse: (batch, nheads, seqlen_q) float32

# For identity scales (plain FP8, no block scaling):
sf = torch.full((batch, nheads, seqlen, headdim // 32), 127, dtype=torch.uint8, device='cuda')
```

For training integration, wrap in `torch.autograd.Function` with a BF16 backward fallback (the kernel is forward-only).

---

## 8. Remaining Optimization Opportunities

| Optimization | Expected Impact | Difficulty |
|-------------|-----------------|------------|
| Reduce `__syncthreads` (3 per N-block -> 2) | +5-10% | Medium |
| Vectorized O epilogue (128-bit stores) | +3-5% | Low |
| Skip identity SF fill (custom MMA) | +3-5% | Medium |
| Register pressure reduction (255 -> <240) | Better occupancy/latency hiding | Medium |
| Warp specialization (if SM120 supports partial) | +10-20% | High |

---

## 9. Files Modified/Created

### New Files
| File | Purpose |
|------|---------|
| `hopper/tile_size_sm120.h` | Tile size heuristics (128x128x128, 8 warps, 2 stages) |
| `hopper/mainloop_fwd_sm120_tma_mma.hpp` | Core mainloop: TMA, pipeline, swizzled SMEM, LDSM, block-scaled MMA, softmax, causal mask |
| `hopper/flash_fwd_kernel_sm120.h` | Kernel driver: pipeline init, epilogue (O + LSE write) |
| `hopper/flash_fwd_launch_template_sm120.h` | Launch template: args construction, kernel dispatch |
| `hopper/instantiations/flash_fwd_hdim{64,128,256}_e4m3_sm120.cu` | Template instantiations |
| `hopper/test_sm120.py` | Correctness test suite (3 tests) |
| `hopper/bench_sm120.py` | Performance benchmark (10 configs, 3 backends) |

### Modified Files
| File | Change |
|------|--------|
| `hopper/setup.py` | SM120 compile rule (`sm_120a` gencode), source list |
| `hopper/flash.h` | MXFP8 scale factor fields in `Flash_fwd_params` |
| `hopper/static_switch.h` | `Arch==120` dispatch case |
| `hopper/utils.h` | `enable_sm120_or_later` architecture guard |
| `hopper/flash_api_stable.cpp` | SM120 dispatch + `mha_fwd_mxfp8` C++ op + `fwd_mxfp8` registration |
| `hopper/flash_attn_interface.py` | `flash_attn_mxfp8_func` wired to `flash_attn_3_gpu.fwd_mxfp8()` |

---

## 10. Build Command Reference

```bash
cd /home/nanogpt/prj/fp8_flashattention/flash-attention/hopper
source ../.venv/bin/activate

# Minimal build (SM120 only, forward only, hdim128 only)
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE \
  FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
  FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
  FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
  FLASH_ATTENTION_DISABLE_PACKGQA=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE \
  FLASH_ATTENTION_DISABLE_VARLEN=TRUE FLASH_ATTENTION_DISABLE_CLUSTER=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
  FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
  python setup.py build_ext --inplace

# Test
python test_sm120.py

# Benchmark
python bench_sm120.py
```
