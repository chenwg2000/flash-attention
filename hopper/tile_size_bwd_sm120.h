/******************************************************************************
 * Tile size heuristics for SM120 backward pass (Blackwell consumer, RTX 5090)
 *
 * Backward uses kBlockM=64 (reduced from fwd's 128) to fit persistent dK/dV
 * accumulators in registers (128 FP32 regs = 64 for dK + 64 for dV).
 * kBlockN=128 must match forward (SF Blk_MN=128 minimum for block-scaled MMA).
 *
 * Grid iterates over N-blocks (one CTA per N-block), accumulating dK/dV
 * across all M-blocks before writing out.
 ******************************************************************************/

#pragma once

// Returns {kBlockM, kBlockN, kNWarps, kStages}
constexpr std::tuple<int, int, int, int> tile_size_bwd_sm120(int headdim) {
    if (headdim <= 128) {
        // kBlockM=64: reduced from forward's 128 to fit dK/dV accumulators in registers
        // kBlockN=128: matches forward, N-dimension for K/V tiles
        // All GEMMs use BF16 MMA (no block-scaled GEMM-1) to avoid SF Blk_MN=128 constraint
        // 8 warps, 1 stage
        return {64, 128, 8, 1};
    } else {  // headdim 256
        return {64, 128, 8, 1};
    }
}
