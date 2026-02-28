/******************************************************************************
 * SM120 MXFP8 Block-Scaled Flash Attention Backward Kernel Driver (Phase 1: dK+dV)
 *
 * Grid: (n_blocks, num_heads_kv, batch_size)
 * Each CTA computes one N-block of dK and dV by iterating over all M-blocks.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "utils.h"
#include "mainloop_bwd_sm120_tma_mma.hpp"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_>
class FlashAttnBwdSm120 {

public:
    using CollectiveMainloop = CollectiveMainloop_;
    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using Element = typename CollectiveMainloop::Element;
    using ElementAccum = typename CollectiveMainloop::ElementAccum;
    using ElementBf16 = typename CollectiveMainloop::ElementBf16;

    static constexpr int kBlockM = CollectiveMainloop::kBlockM;
    static constexpr int kBlockN = CollectiveMainloop::kBlockN;
    static constexpr int kHeadDim = CollectiveMainloop::kHeadDim;

    static constexpr uint32_t NumMmaThreads = CollectiveMainloop::NumMmaThreads;
    static constexpr uint32_t MaxThreadsPerBlock = NumMmaThreads;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

    using TiledMma_G3 = typename CollectiveMainloop::TiledMma_G3;
    using TiledMma_G4 = typename CollectiveMainloop::TiledMma_G4;

    using TensorStorage = typename CollectiveMainloop::TensorStorage;

    struct SharedStorage : cute::aligned_struct<128> {
        TensorStorage mainloop;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    struct Arguments {
        typename CollectiveMainloop::Arguments mainloop{};
        ElementBf16* ptr_dK;
        int64_t dk_batch_stride, dk_row_stride, dk_head_stride;
        ElementBf16* ptr_dV;
        int64_t dv_batch_stride, dv_row_stride, dv_head_stride;
        int batch_size, num_heads, num_heads_kv;
        int seqlen_q, seqlen_k;
        int head_dim;
    };

    struct Params {
        typename CollectiveMainloop::Params mainloop{};
        ElementBf16* ptr_dK;
        int64_t dk_batch_stride, dk_row_stride, dk_head_stride;
        ElementBf16* ptr_dV;
        int64_t dv_batch_stride, dv_row_stride, dv_head_stride;
        int batch_size, num_heads, num_heads_kv;
        int seqlen_q, seqlen_k;
        int head_dim;
        int num_n_blocks;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            args.ptr_dK,
            args.dk_batch_stride, args.dk_row_stride, args.dk_head_stride,
            args.ptr_dV,
            args.dv_batch_stride, args.dv_row_stride, args.dv_head_stride,
            args.batch_size, args.num_heads, args.num_heads_kv,
            args.seqlen_q, args.seqlen_k,
            args.head_dim,
            cute::ceil_div(args.seqlen_k, kBlockN)
        };
    }

    static dim3 get_grid_shape(Params const& params) {
        return dim3(params.num_n_blocks, params.num_heads_kv, params.batch_size);
    }

    static dim3 get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTLASS_DEVICE void
    operator()(Params const& params, char* smem_buf) {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int const n_block = blockIdx.x;
        int const bidh_kv = blockIdx.y;
        int const bidb = blockIdx.z;
        int const n_start = n_block * kBlockN;
        if (n_start >= params.seqlen_k) return;

        // ====== Allocate dK and dV accumulators (FP32, persistent across M-blocks) ======
        TiledMma_G4 tiled_mma_dk;
        TiledMma_G3 tiled_mma_dv;

        // dK accumulator: [kBlockN x kHeadDim] in FP32
        Tensor tOrDK = partition_fragment_C(tiled_mma_dk, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
        clear(tOrDK);

        // dV accumulator: [kBlockN x kHeadDim] in FP32
        Tensor tOrDV = partition_fragment_C(tiled_mma_dv, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
        clear(tOrDV);

        // ====== Run backward mainloop ======
        CollectiveMainloop mainloop;
        mainloop.mha_bwd(
            params.mainloop, shared_storage.mainloop,
            tOrDK, tOrDV,
            n_block, bidb, bidh_kv, threadIdx.x);

        // ====== Epilogue: write dK and dV as BF16 to GMEM ======
        int const warp_idx = threadIdx.x / 32;
        int const lane_idx = threadIdx.x % 32;
        int const lane_row = lane_idx / 4;
        int const lane_col = lane_idx % 4;

        // Write dK
        {
            Tensor tOrDK_rc = make_tensor(tOrDK.data(),
                flash::convert_layout_acc_rowcol(tOrDK.layout()));
            int const nrow = size<0>(tOrDK_rc);
            int const ncol = size<1>(tOrDK_rc);

            // For 8x1x1 warp layout: warp_m = warp_idx, warp_n = 0
            ElementBf16* dk_ptr = params.ptr_dK
                + bidb * params.dk_batch_stride
                + bidh_kv * params.dk_head_stride;

            #pragma unroll
            for (int mi = 0; mi < nrow; ++mi) {
                int const atom_row_offset = (mi % 2) * 8;
                int const mma_m = mi / 2;
                int const global_row = n_start + warp_idx * 16 + mma_m * 16 + lane_row + atom_row_offset;
                if (global_row >= params.seqlen_k) continue;

                #pragma unroll
                for (int ni = 0; ni < ncol; ++ni) {
                    int const atom_col_offset = ni % 2;
                    int const mma_n = ni / 2;
                    int const global_col = mma_n * 8 + lane_col * 2 + atom_col_offset;

                    if (global_col < params.head_dim) {
                        float val = tOrDK_rc(mi, ni);
                        ElementBf16 val_bf16 = static_cast<ElementBf16>(val);
                        dk_ptr[global_row * params.dk_row_stride + global_col] = val_bf16;
                    }
                }
            }
        }

        // Write dV
        {
            Tensor tOrDV_rc = make_tensor(tOrDV.data(),
                flash::convert_layout_acc_rowcol(tOrDV.layout()));
            int const nrow = size<0>(tOrDV_rc);
            int const ncol = size<1>(tOrDV_rc);

            ElementBf16* dv_ptr = params.ptr_dV
                + bidb * params.dv_batch_stride
                + bidh_kv * params.dv_head_stride;

            #pragma unroll
            for (int mi = 0; mi < nrow; ++mi) {
                int const atom_row_offset = (mi % 2) * 8;
                int const mma_m = mi / 2;
                int const global_row = n_start + warp_idx * 16 + mma_m * 16 + lane_row + atom_row_offset;
                if (global_row >= params.seqlen_k) continue;

                #pragma unroll
                for (int ni = 0; ni < ncol; ++ni) {
                    int const atom_col_offset = ni % 2;
                    int const mma_n = ni / 2;
                    int const global_col = mma_n * 8 + lane_col * 2 + atom_col_offset;

                    if (global_col < params.head_dim) {
                        float val = tOrDV_rc(mi, ni);
                        ElementBf16 val_bf16 = static_cast<ElementBf16>(val);
                        dv_ptr[global_row * params.dv_row_stride + global_col] = val_bf16;
                    }
                }
            }
        }
    }
};

}  // namespace flash
