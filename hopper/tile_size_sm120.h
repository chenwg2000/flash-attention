/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

// Tile size heuristics for SM120 (Blackwell consumer, RTX 5090)
// SM120 uses warp-level mma.sync (32 threads, 16x8x32 tiles) with block-scaled MXFP8
// No warp specialization: all threads do both loads and MMA compute
// 256 threads = 8 warps, kStages=2 (double-buffered K/V)
//
// IMPORTANT: kBlockM and kBlockN must be >= 128 due to the CUTLASS SM120
// block-scaled SF SMEM layout requiring Blk_MN=128 as the minimum tile size.

// Returns {kBlockM, kBlockN, kNWarps, kStages}
constexpr std::tuple<int, int, int, int> tile_size_fwd_sm120(int headdim) {
    if (headdim <= 64) {
        // Q: 128*64=8KB, K: 128*64*2=16KB, V: 128*64*2=16KB = 40KB + SF
        return {128, 128, 8, 2};
    } else if (headdim <= 128) {
        // Q: 128*128=16KB, K: 128*128*2=32KB, V: 128*128*2=32KB = 80KB + SF
        return {128, 128, 8, 2};
    } else {  // headdim 256
        // Q: 128*256=32KB, K: 128*256*1=32KB, V: 128*256*1=32KB = 96KB + SF
        // Use single-buffered to fit in SMEM
        return {128, 128, 8, 1};
    }
}
