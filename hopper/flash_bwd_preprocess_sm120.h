/******************************************************************************
 * SM120 MXFP8 Flash Attention Backward Preprocessing Kernel
 *
 * Computes two quantities needed by the backward mainloop:
 *   dPsum[b][h][m] = sum_d( dO[b][m][h][d] * O[b][m][h][d] )   (FP32)
 *   LSE_log2[b][h][m] = LSE[b][h][m] * log2(e)                  (FP32)
 *
 * Grid: (ceil(seqlen_q / 256), num_heads, batch_size)
 * Block: 256 threads, each thread processes 1 row
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

namespace flash {

__global__ void flash_bwd_preprocess_sm120_kernel(
    __nv_bfloat16 const* __restrict__ dO,   // (b, s_q, h, d)
    __nv_bfloat16 const* __restrict__ O,    // (b, s_q, h, d)
    float const* __restrict__ LSE,          // (b, h, s_q)
    float* __restrict__ dPsum,              // (b, h, s_q)
    float* __restrict__ LSE_log2,           // (b, h, s_q)
    int seqlen_q,
    int num_heads,
    int headdim,
    int64_t do_batch_stride,
    int64_t do_row_stride,
    int64_t do_head_stride,
    int64_t o_batch_stride,
    int64_t o_row_stride,
    int64_t o_head_stride,
    int64_t lse_batch_stride,    // = h * s_q
    int64_t lse_head_stride      // = s_q
) {
    int const m = blockIdx.x * blockDim.x + threadIdx.x;
    int const bidh = blockIdx.y;
    int const bidb = blockIdx.z;

    if (m >= seqlen_q) return;

    // Compute dPsum = dot(dO[m,:], O[m,:])
    __nv_bfloat16 const* dO_row = dO + bidb * do_batch_stride
        + m * do_row_stride + bidh * do_head_stride;
    __nv_bfloat16 const* O_row = O + bidb * o_batch_stride
        + m * o_row_stride + bidh * o_head_stride;

    float sum = 0.0f;
    // Vectorized: process 8 BF16 elements at a time via uint4 (16 bytes = 8 bf16)
    int d = 0;
    for (; d + 7 < headdim; d += 8) {
        uint4 dO_vec = *reinterpret_cast<uint4 const*>(dO_row + d);
        uint4 O_vec  = *reinterpret_cast<uint4 const*>(O_row + d);
        __nv_bfloat16 const* dO_v = reinterpret_cast<__nv_bfloat16 const*>(&dO_vec);
        __nv_bfloat16 const* O_v  = reinterpret_cast<__nv_bfloat16 const*>(&O_vec);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum += __bfloat162float(dO_v[i]) * __bfloat162float(O_v[i]);
        }
    }
    for (; d < headdim; ++d) {
        sum += __bfloat162float(dO_row[d]) * __bfloat162float(O_row[d]);
    }

    int const lse_idx = bidb * lse_batch_stride + bidh * lse_head_stride + m;
    dPsum[lse_idx] = sum;
    LSE_log2[lse_idx] = LSE[lse_idx] * float(M_LOG2E);
}

inline void flash_bwd_preprocess_sm120(
    void const* dO,
    void const* O,
    void const* LSE,
    void* dPsum,
    void* LSE_log2,
    int batch_size,
    int seqlen_q,
    int num_heads,
    int headdim,
    int64_t do_batch_stride,
    int64_t do_row_stride,
    int64_t do_head_stride,
    int64_t o_batch_stride,
    int64_t o_row_stride,
    int64_t o_head_stride,
    int64_t lse_batch_stride,
    int64_t lse_head_stride,
    cudaStream_t stream
) {
    constexpr int kThreads = 256;
    dim3 grid(
        (seqlen_q + kThreads - 1) / kThreads,
        num_heads,
        batch_size
    );
    flash_bwd_preprocess_sm120_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(dO),
        static_cast<__nv_bfloat16 const*>(O),
        static_cast<float const*>(LSE),
        static_cast<float*>(dPsum),
        static_cast<float*>(LSE_log2),
        seqlen_q, num_heads, headdim,
        do_batch_stride, do_row_stride, do_head_stride,
        o_batch_stride, o_row_stride, o_head_stride,
        lse_batch_stride, lse_head_stride
    );
}

// ====== Postprocess: convert dq_accum FP32 → dq BF16 ======

__global__ void flash_bwd_convert_dq_kernel_sm120(
    float const* __restrict__ dq_accum,
    __nv_bfloat16* __restrict__ dq,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dq[idx] = __float2bfloat16(dq_accum[idx]);
    }
}

inline void flash_bwd_postprocess_dq_sm120(
    void const* dq_accum,
    void* dq,
    int batch_size,
    int seqlen_q,
    int num_heads,
    int headdim,
    cudaStream_t stream
) {
    int total = batch_size * seqlen_q * num_heads * headdim;
    constexpr int kThreads = 256;
    dim3 grid((total + kThreads - 1) / kThreads);
    flash_bwd_convert_dq_kernel_sm120<<<grid, kThreads, 0, stream>>>(
        static_cast<float const*>(dq_accum),
        static_cast<__nv_bfloat16*>(dq),
        total
    );
}

}  // namespace flash
