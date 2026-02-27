/******************************************************************************
 * SM120 MXFP8 Block-Scaled Flash Attention Forward Kernel Driver
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "utils.h"
#include "mainloop_fwd_sm120_tma_mma.hpp"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_>
class FlashAttnFwdSm120 {

public:
    using CollectiveMainloop = CollectiveMainloop_;
    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TiledMma = typename CollectiveMainloop::TiledMma;
    using Element = typename CollectiveMainloop::Element;
    using ElementAccum = typename CollectiveMainloop::ElementAccum;

    static constexpr int kBlockM = CollectiveMainloop::kBlockM;
    static constexpr int kBlockN = CollectiveMainloop::kBlockN;
    static constexpr int kHeadDim = CollectiveMainloop::kHeadDim;

    static constexpr uint32_t NumMmaThreads = CollectiveMainloop::NumMmaThreads;
    static constexpr uint32_t MaxThreadsPerBlock = NumMmaThreads;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

    using TensorStorage = typename CollectiveMainloop::TensorStorage;
    struct SharedStorage : cute::aligned_struct<128> {
        TensorStorage mainloop;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    struct Arguments {
        typename CollectiveMainloop::Arguments mainloop{};
        cutlass::bfloat16_t* ptr_O;
        int64_t o_batch_stride, o_row_stride, o_head_stride;
        float* ptr_LSE;
        int64_t lse_batch_stride, lse_head_stride;
        int batch_size, num_heads, num_heads_kv;
        int seqlen_q, seqlen_k;
        int head_dim;
    };

    struct Params {
        typename CollectiveMainloop::Params mainloop{};
        cutlass::bfloat16_t* ptr_O;
        int64_t o_batch_stride, o_row_stride, o_head_stride;
        float* ptr_LSE;
        int64_t lse_batch_stride, lse_head_stride;
        int batch_size, num_heads, num_heads_kv;
        int seqlen_q, seqlen_k;
        int head_dim;
        int num_m_blocks;
        int qhead_per_khead;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            args.ptr_O,
            args.o_batch_stride, args.o_row_stride, args.o_head_stride,
            args.ptr_LSE,
            args.lse_batch_stride, args.lse_head_stride,
            args.batch_size, args.num_heads, args.num_heads_kv,
            args.seqlen_q, args.seqlen_k,
            args.head_dim,
            cute::ceil_div(args.seqlen_q, kBlockM),
            args.num_heads / args.num_heads_kv
        };
    }

    static dim3 get_grid_shape(Params const& params) {
        return dim3(params.num_m_blocks, params.num_heads, params.batch_size);
    }

    static dim3 get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTLASS_DEVICE void
    operator()(Params const& params, char* smem_buf) {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int const m_block = blockIdx.x;
        int const bidh = blockIdx.y;
        int const bidb = blockIdx.z;
        int const bidh_kv = bidh / params.qhead_per_khead;
        int const m_start = m_block * kBlockM;
        if (m_start >= params.seqlen_q) return;

        CollectiveMainloop mainloop;
        TiledMma tiled_mma;

        // Output accumulator: O[kBlockM, kHeadDim] in FP32
        Tensor tOrO = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));
        clear(tOrO);

        // Softmax LSE per row
        Tensor acc_s_ref = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
        static constexpr int kAccRows = decltype(size<0>(flash::convert_layout_acc_rowcol(acc_s_ref.layout())))::value;
        float softmax_lse[kAccRows];
        #pragma unroll
        for (int i = 0; i < kAccRows; ++i) softmax_lse[i] = -INFINITY;

        // ====== Mainloop ======
        mainloop.mha_fwd(
            params.mainloop, shared_storage.mainloop, tOrO, softmax_lse,
            m_block, bidb, bidh, bidh_kv, threadIdx.x);

        // ====== Epilogue: write O and LSE ======
        //
        // The MMA accumulator has a fragmented register layout.
        // For the SM120 16x8 atom with 4x2 warp layout:
        //   - Each atom: 16 rows x 8 cols, DRegisters = float[4]
        //   - 4 warps along M x 2 warps along N = covers 64x16 per MMA step
        //   - PermTile 128x16: MMA_M=2 steps along M, MMA_N depends on output dim
        //
        // Register layout per atom (C-operand, float[4]):
        //   lane_idx = threadIdx.x % 32
        //   Within 16x8 atom output:
        //     reg[0]: row = (lane/4),     col = (lane%4)*2
        //     reg[1]: row = (lane/4),     col = (lane%4)*2 + 1
        //     reg[2]: row = (lane/4) + 8, col = (lane%4)*2
        //     reg[3]: row = (lane/4) + 8, col = (lane%4)*2 + 1
        //
        // With 4x2 atom layout and PermTile (128,16):
        //   warp_m = warp_idx / 2  (0..3)
        //   warp_n = warp_idx % 2  (0..1)
        //   MMA_M = 2 (128 / (4*16) = 2)
        //   MMA_N = kHeadDim/16 (128/16 = 8 for headdim=128)
        //
        // With 8x1 atom layout:
        //   warp_m = warp_idx (0..7), warp_n = 0 (always)
        //   atom: 16 rows × 8 cols
        //   8 warps × 16 = 128 M-rows (MMA_M=1)
        //   MMA_N = kHeadDim/8 = 16
        //
        // Acc shape: ((2,2), MMA_M=1, MMA_N=16) → nrow=2, ncol=32
        //   row_i: row_i%2 selects atom_row (0→lane/4, 1→lane/4+8)
        //          row_i/2 selects mma_m (always 0)
        //   col_j: col_j%2 selects atom_col (0→(lane%4)*2, 1→(lane%4)*2+1)
        //          col_j/2 selects mma_n (0..15, offset *8)

        int const warp_idx = threadIdx.x / 32;
        int const lane_idx = threadIdx.x % 32;
        int const warp_m = warp_idx;    // 0..7 (all warps along M)
        int const warp_n = 0;           // always 0
        int const lane_row = lane_idx / 4;  // 0..7
        int const lane_col = lane_idx % 4;  // 0..3

        // Reshape to (nrow, ncol) view
        Tensor tOrO_rowcol = make_tensor(tOrO.data(),
            flash::convert_layout_acc_rowcol(tOrO.layout()));
        int const nrow = size<0>(tOrO_rowcol);
        int const ncol = size<1>(tOrO_rowcol);

        cutlass::bfloat16_t* o_ptr = params.ptr_O
            + bidb * params.o_batch_stride
            + bidh * params.o_head_stride;

        float* lse_ptr = params.ptr_LSE
            + bidb * params.lse_batch_stride
            + bidh * params.lse_head_stride;

        #pragma unroll
        for (int mi = 0; mi < nrow; ++mi) {
            // Map row index to global M position
            // With 8x1 layout: warp_m covers 16 rows, no MMA_M repetition
            int const atom_row_offset = (mi % 2) * 8;  // 0 or 8
            int const mma_m = mi / 2;                   // always 0 for 8x1
            int const global_row = m_start + warp_m * 16 + mma_m * 16 + lane_row + atom_row_offset;
            if (global_row >= params.seqlen_q) continue;

            // Write LSE for this row (only one thread per row writes)
            // lane_col==0 ensures only one thread per quad writes
            if (lane_col == 0) {
                lse_ptr[global_row] = softmax_lse[mi];
            }

            #pragma unroll
            for (int ni = 0; ni < ncol; ++ni) {
                // Map col index to global D position
                // With 8x1: warp_n=0, mma_n=0..15, atom N=8
                int const atom_col_offset = ni % 2;     // 0 or 1
                int const mma_n = ni / 2;                // 0..15
                int const global_col = mma_n * 8 + lane_col * 2 + atom_col_offset;

                if (global_col < params.head_dim) {
                    float val = tOrO_rowcol(mi, ni);
                    cutlass::bfloat16_t val_bf16 = static_cast<cutlass::bfloat16_t>(val);
                    o_ptr[global_row * params.o_row_stride + global_col] = val_bf16;
                }
            }
        }
    }
};

}  // namespace flash
