/******************************************************************************
 * SM120 MXFP8 Block-Scaled Flash Attention Backward Launch Template
 * Computes dK + dV + dQ (dQ via atomicAdd to FP32 accumulator, then BF16 postprocess)
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/kernel_launch.h>

#include "static_switch.h"
#include "flash.h"
#include "tile_size_bwd_sm120.h"
#include "flash_bwd_kernel_sm120.h"
#include "flash_bwd_preprocess_sm120.h"
#include "mainloop_bwd_sm120_tma_mma.hpp"

using namespace cute;

template <int kHeadDim, typename Element, bool Is_causal>
void run_flash_bwd_sm120(Flash_bwd_params &params, cudaStream_t stream) {
    static_assert(cute::is_same_v<Element, cutlass::float_e4m3_t>,
                  "SM120 backward path only supports FP8 e4m3");

    // Tile sizes from heuristics
    static constexpr auto tile_config = tile_size_bwd_sm120(kHeadDim);
    static constexpr int kBlockM = std::get<0>(tile_config);
    static constexpr int kBlockN = std::get<1>(tile_config);
    static constexpr int kStages = std::get<3>(tile_config);

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    using CollectiveMainloop = flash::CollectiveMainloopBwdSm120<
        kStages, TileShape_MNK, Element, float, Is_causal>;

    using AttnKernel = flash::enable_sm120_or_later<
        flash::FlashAttnBwdSm120<CollectiveMainloop>>;

    int seqlen_q = params.seqlen_q;
    int seqlen_k = params.seqlen_k;
    int batch_size = params.b;
    int num_heads = params.h;
    int num_heads_kv = params.h_k;

    // ====== Step 1: Preprocessing (dPsum + LSE_log2) ======
    flash::flash_bwd_preprocess_sm120(
        params.do_ptr, params.o_ptr, params.softmax_lse_ptr,
        params.dsoftmax_sum,      // dPsum output
        params.softmax_lse_log2_ptr,  // LSE_log2 output
        batch_size, seqlen_q, num_heads, kHeadDim,
        params.do_batch_stride, params.do_row_stride, params.do_head_stride,
        params.o_batch_stride, params.o_row_stride, params.o_head_stride,
        num_heads * seqlen_q,  // lse_batch_stride
        seqlen_q,              // lse_head_stride
        stream
    );

    // ====== Step 2: Backward kernel (dK + dV + dQ) ======
    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        params.q_batch_stride, params.q_row_stride, params.q_head_stride,
        static_cast<Element const*>(params.k_ptr),
        params.k_batch_stride, params.k_row_stride, params.k_head_stride,
        static_cast<Element const*>(params.v_ptr),
        params.v_batch_stride, params.v_row_stride, params.v_head_stride,
        // MXFP8 scale factors
        static_cast<uint8_t const*>(params.q_scale_ptr),
        params.q_scale_batch_stride, params.q_scale_head_stride, params.q_scale_row_stride,
        static_cast<uint8_t const*>(params.k_scale_ptr),
        params.k_scale_batch_stride, params.k_scale_head_stride, params.k_scale_row_stride,
        // dO (BF16)
        static_cast<cutlass::bfloat16_t const*>(params.do_ptr),
        params.do_batch_stride, params.do_row_stride, params.do_head_stride,
        // Precomputed LSE_log2 and dPsum
        static_cast<float const*>(params.softmax_lse_log2_ptr),
        static_cast<float const*>(params.dsoftmax_sum),
        num_heads * seqlen_q,  // lse_batch_stride
        seqlen_q,              // lse_head_stride
        params.scale_softmax,
        seqlen_q, seqlen_k,
        num_heads, num_heads_kv, batch_size,
        // dQ accumulator (FP32, atomicAdded by mainloop, then postprocessed to BF16)
        static_cast<float*>(params.dq_accum_ptr),
        params.dq_batch_stride, params.dq_row_stride, params.dq_head_stride
    };

    typename AttnKernel::Arguments kernel_args {
        mainloop_args,
        static_cast<cutlass::bfloat16_t*>(params.dk_ptr),
        params.dk_batch_stride, params.dk_row_stride, params.dk_head_stride,
        static_cast<cutlass::bfloat16_t*>(params.dv_ptr),
        params.dv_batch_stride, params.dv_row_stride, params.dv_head_stride,
        batch_size, num_heads, num_heads_kv,
        seqlen_q, seqlen_k, kHeadDim
    };

    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments(kernel_args);

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;

    auto kernel = cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
    CHECK_CUDA_KERNEL_LAUNCH();

    // ====== Step 3: Postprocess — convert dq_accum (FP32) → dq (BF16) ======
    flash::flash_bwd_postprocess_dq_sm120(
        params.dq_accum_ptr, params.dq_ptr,
        batch_size, seqlen_q, num_heads, kHeadDim,
        stream
    );
}

template <typename T, int kHeadDim>
void run_mha_bwd_sm120_(Flash_bwd_params &params, cudaStream_t stream) {
    static_assert(cute::is_same_v<T, cutlass::float_e4m3_t>);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_flash_bwd_sm120<kHeadDim, T, Is_causal>(params, stream);
    });
}
