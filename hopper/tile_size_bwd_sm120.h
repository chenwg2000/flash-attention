/******************************************************************************
 * Tile size heuristics for SM120 backward pass (Blackwell consumer, RTX 5090)
 *
 * kBlockM=128 matches the MMA tile (Blk_MN=128 minimum for block-scaled MMA),
 * eliminating all zero-padding and halving M-block count vs kBlockM=64.
 * dO is read directly from GMEM (L2 cached) to save 32KB SMEM.
 * kBlockN=128 matches forward (SF Blk_MN=128 minimum).
 *
 * Grid iterates over N-blocks (one CTA per N-block), accumulating dK/dV
 * across all M-blocks before writing out.
 *
 * SMEM: ~82 KB (fits in SM120's 100 KB max)
 ******************************************************************************/

#pragma once

// Returns {kBlockM, kBlockN, kNWarps, kStages}
constexpr std::tuple<int, int, int, int> tile_size_bwd_sm120(int headdim) {
    if (headdim <= 128) {
        // kBlockM=128: matches MMA tile, no padding waste, halves M-block count
        // kBlockN=128: matches forward, N-dimension for K/V tiles
        // 8 warps, 1 stage
        return {128, 128, 8, 1};
    } else {  // headdim 256
        return {128, 128, 8, 1};
    }
}
