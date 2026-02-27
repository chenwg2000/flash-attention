/******************************************************************************
 * SM120 MXFP8 Block-Scaled Flash Attention Forward Launch Template
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/kernel_launch.h>

#include "static_switch.h"
#include "flash.h"
#include "tile_size_sm120.h"
#include "flash_fwd_kernel_sm120.h"
#include "mainloop_fwd_sm120_tma_mma.hpp"

using namespace cute;

template <int kHeadDim, typename Element, bool Is_causal, bool Is_local, bool Has_softcap>
void run_flash_fwd_sm120(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(cute::is_same_v<Element, cutlass::float_e4m3_t>,
                  "SM120 path only supports FP8 e4m3");

    // Tile sizes from heuristics
    static constexpr auto tile_config = tile_size_fwd_sm120(kHeadDim);
    static constexpr int kBlockM = std::get<0>(tile_config);
    static constexpr int kBlockN = std::get<1>(tile_config);
    static constexpr int kNWarps = std::get<2>(tile_config);
    static constexpr int kStages = std::get<3>(tile_config);

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    using CollectiveMainloop = flash::CollectiveMainloopFwdSm120<
        kStages, TileShape_MNK, Element, float,
        Is_causal, Is_local, Has_softcap>;

    using AttnKernel = flash::enable_sm120_or_later<
        flash::FlashAttnFwdSm120<CollectiveMainloop>>;

    int seqlen_q = params.seqlen_q;
    int seqlen_k = params.seqlen_k;
    int batch_size = params.b;
    int num_heads = params.h;
    int num_heads_kv = params.h_k;

    // Build mainloop arguments
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
        static_cast<uint8_t const*>(params.v_scale_ptr),
        params.v_scale_batch_stride, params.v_scale_head_stride, params.v_scale_row_stride,
        params.scale_softmax,
        params.softcap,
        params.window_size_left, params.window_size_right,
        seqlen_q, seqlen_k
    };

    // Build kernel arguments
    typename AttnKernel::Arguments kernel_args {
        mainloop_args,
        static_cast<cutlass::bfloat16_t*>(params.o_ptr),
        params.o_batch_stride, params.o_row_stride, params.o_head_stride,
        static_cast<float*>(params.softmax_lse_ptr),
        params.h * seqlen_q,  // lse_batch_stride
        seqlen_q,             // lse_head_stride
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
}

template <typename T, int kHeadDim>
void run_mha_fwd_sm120_(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(cute::is_same_v<T, cutlass::float_e4m3_t>);
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
            run_flash_fwd_sm120<kHeadDim, T, Is_causal, Is_local, Has_softcap>(params, stream);
        });
    });
}
