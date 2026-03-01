/******************************************************************************
 * SM120 MXFP8 Block-Scaled Flash Attention Backward Mainloop (dK+dV+dQ)
 *
 * Each CTA owns one N-block (K/V block). Grid: (n_blocks, heads_kv, batch).
 * dK and dV accumulate in FP32 registers across all M-blocks, then write as BF16.
 * dQ is computed per (M-block, N-block) pair and atomicAdded to a FP32 GMEM accumulator.
 *
 * kBlockM=128 matches MMA tile (Blk_MN=128), eliminating all zero-padding.
 * dO is read directly from GMEM (L2 cached) to save 32KB SMEM.
 * All-FP8 block-scaled design:
 *   GEMM-1 (S = Q @ K^T):      FP8 block-scaled MMA (real SFs from GMEM)
 *   GEMM-2 (dP = dO @ V):      FP8 block-scaled MMA (identity SFs)
 *   GEMM-3 (dV += P^T @ dO^T): FP8 block-scaled MMA (identity SFs)
 *   GEMM-4 (dK += dS^T @ Q^T): FP8 block-scaled MMA (identity SFs)
 *   GEMM-5 (dQ += dS @ K):     FP8 block-scaled MMA (identity SFs), atomicAdd to GMEM
 *
 * SMEM budget (~82 KB, fits in SM120's 100 KB):
 *   K [128x128 FP8 SW128]:   16 KB (resident)
 *   V [128x128 FP8 SW128]:   16 KB (resident, swizzled directly)
 *   SFK [512B]:                <1 KB (resident, then repurposed for identity SFs)
 *   Q [128x128 FP8 SW128]:   16 KB (per M-block; reused for P^T, dS^T)
 *   SFQ [512B]:                <1 KB (per M-block; reused for identity SFs)
 *   scratch [128x128 FP8]:    16 KB (dO_fp8, dO^T FP8, dS non-transposed)
 *   Q^T [128x128 FP8 SW128]: 16 KB (scratch)
 *   (dO read from GMEM — not stored in SMEM)
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/arch.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>

#include "cute/tensor.hpp"
#include "cute/arch/mma_sm120.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"

#include "utils.h"
#include "softmax.h"
#include "mainloop_fwd_sm120_tma_mma.hpp"

namespace flash {

using namespace cute;

template <int kStages_, class TileShape_MNK_, class Element_, class ElementAccum_,
          bool Is_causal_>
struct CollectiveMainloopBwdSm120 {

    static constexpr int kStages = kStages_;
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ElementBf16 = cutlass::bfloat16_t;

    static constexpr bool Is_causal = Is_causal_;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});  // 128
    static constexpr int kBlockN = get<1>(TileShape_MNK{});  // 128
    static constexpr int kHeadDim = get<2>(TileShape_MNK{}); // 128

    // MMA tile M dimension: matches kBlockM (no padding needed since kBlockM=128=Blk_MN)
    static constexpr int kBlockM_SF = kBlockN;  // = 128 = kBlockM

    // ====== Block-scaled MMA types (GEMM-1 and GEMM-2) ======
    using ElementSF = cutlass::float_ue8m0_t;
    static constexpr int kSFVecSize = 32;
    static constexpr int kSFCols = kHeadDim / kSFVecSize;

    using MmaAtomOp = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
        cutlass::float_ue8m0_t, kSFVecSize>;
    // 8 warps along M, 1 along N — kBlockM=128 matches MMA tile exactly
    using AtomLayoutMNK_G1 = Layout<Shape<_8, _1, _1>>;
    using PermTileM_G1 = decltype(cute::min(Int<kBlockM_SF>{}, _128{}));  // 128
    using TiledMma_G1 = decltype(cute::make_tiled_mma(
        MMA_Atom<MmaAtomOp>{}, AtomLayoutMNK_G1{},
        Tile<PermTileM_G1, _8, _32>{}));

    // GEMMs 3-4 reuse same MMA (identity SFs)
    using TiledMma_G3 = TiledMma_G1;
    using TiledMma_G4 = TiledMma_G1;

    static constexpr int NumMmaThreads = 256;

    // ====== Swizzled SMEM Layouts ======
    using SmemLayoutAtomSW_FP8 = SM90::GMMA::Layout_K_SW128_Atom<uint8_t>;
    using SmemLayoutAtomSW_BF16 = SM90::GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>;

    // Resident tiles (K and V both [kBlockN x kHeadDim] FP8 swizzled)
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomSW_FP8{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomSW_FP8{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));
    using SmemLayoutV = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;

    // Per M-block tiles (kBlockM=128=kBlockM_SF, no padding)
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomSW_FP8{}, Shape<Int<kBlockM_SF>, Int<kHeadDim>>{}));

    // ====== SF layouts ======
    // SFQ covers kBlockM=128 rows (= Blk_MN, no padding needed).
    static constexpr int MMA_NSF = 32 / kSFVecSize;
    using Sm1xxCfg = cutlass::detail::Sm1xxBlockScaledConfig<kSFVecSize>;
    using Blk_MN = typename Sm1xxCfg::Blk_MN;
    using Blk_SF = typename Sm1xxCfg::Blk_SF;
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
    using mnBBS  = Shape<_32, _4>;
    using mnBBSt = Stride<_16, _4>;
    using kBBS  = Shape<Int<kSFVecSize>, Int<MMA_NSF>>;
    using kBBSt = Stride<_0, _1>;

    using sSFQ_sM = decltype(prepend(Int<kBlockM_SF>{} / Blk_MN{}, mnBBS{}));
    using sSF_sMN = decltype(prepend(Blk_Elems{}, mnBBSt{}));
    using sSF_sK  = decltype(prepend(make_shape(Blk_SF{}/Int<MMA_NSF>{}, Int<kHeadDim>{}/Int<kSFVecSize>{}/Blk_SF{}), kBBS{}));
    using sSFQ_sK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kBlockM_SF>{}/Blk_MN{}*Blk_Elems{}), kBBSt{}));
    using SmemLayoutAtomSFQ = decltype(make_layout(make_shape(sSFQ_sM{}, sSF_sK{}), make_stride(sSF_sMN{}, sSFQ_sK{})));

    using sSFK_sN = decltype(prepend(Int<kBlockN>{} / Blk_MN{}, mnBBS{}));
    using sSFK_sK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kBlockN>{}/Blk_MN{}*Blk_Elems{}), kBBSt{}));
    using SmemLayoutAtomSFK = decltype(make_layout(make_shape(sSFK_sN{}, sSF_sK{}), make_stride(sSF_sMN{}, sSFK_sK{})));

    // ====== Copy atoms ======
    using SmemCopyAtomA_G1 = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;
    using SmemCopyAtomB_G1 = Copy_Atom<SM75_U32x2_LDSM_N, uint8_t>;
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<uint8_t>, uint8_t>;

    // Forward mainloop type for static helpers
    using FwdMainloop = CollectiveMainloopFwdSm120<1,
        cute::Shape<Int<kBlockN>, Int<kBlockN>, Int<kHeadDim>>,
        Element, float, false, false, false>;

    // ====== Shared Memory (~82 KB, fits in SM120's 100 KB) ======
    // dO is read directly from GMEM (L2 cached), saving 32KB SMEM.
    // Total: 16+16+1+16+1+16+16 = ~82 KB
    struct TensorStorage : cute::aligned_struct<128> {
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutK>> smem_k;            // 16 KB
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutVt>> smem_vt;           // 16 KB
        cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutAtomSFK>> smem_sfk;                   // ~512 B
        union {
            alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutV>> smem_v;         // 16 KB
            struct {
                alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutQ>> smem_q;     // 16 KB
                cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutAtomSFQ>> smem_sfq;           // ~512 B
            };
        };
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutQ>> smem_pds;           // 16 KB (FP8 scratch)
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutK>> smem_pt;            // 16 KB (FP8 [128x128])
    };
    static constexpr int SharedStorageSize = sizeof(TensorStorage);

    using index_t = int64_t;

    struct Arguments {
        Element const* ptr_Q;
        int64_t q_batch_stride, q_row_stride, q_head_stride;
        Element const* ptr_K;
        int64_t k_batch_stride, k_row_stride, k_head_stride;
        Element const* ptr_V;
        int64_t v_batch_stride, v_row_stride, v_head_stride;
        uint8_t const* ptr_SFQ; index_t sfq_batch_stride, sfq_head_stride, sfq_row_stride;
        uint8_t const* ptr_SFK; index_t sfk_batch_stride, sfk_head_stride, sfk_row_stride;
        ElementBf16 const* ptr_dO;
        int64_t do_batch_stride, do_row_stride, do_head_stride;
        float const* ptr_LSE_log2;
        float const* ptr_dPsum;
        int64_t lse_batch_stride, lse_head_stride;
        float softmax_scale;
        int seqlen_q, seqlen_k;
        int num_heads, num_heads_kv, batch_size;
        float* ptr_dq_accum;
        int64_t dq_accum_batch_stride, dq_accum_row_stride, dq_accum_head_stride;
    };

    struct Params {
        Element const* ptr_Q;
        int64_t q_batch_stride, q_row_stride, q_head_stride;
        Element const* ptr_K;
        int64_t k_batch_stride, k_row_stride, k_head_stride;
        Element const* ptr_V;
        int64_t v_batch_stride, v_row_stride, v_head_stride;
        uint8_t const* ptr_SFQ; index_t sfq_batch_stride, sfq_head_stride, sfq_row_stride;
        uint8_t const* ptr_SFK; index_t sfk_batch_stride, sfk_head_stride, sfk_row_stride;
        ElementBf16 const* ptr_dO;
        int64_t do_batch_stride, do_row_stride, do_head_stride;
        float const* ptr_LSE_log2;
        float const* ptr_dPsum;
        int64_t lse_batch_stride, lse_head_stride;
        float softmax_scale, softmax_scale_log2;
        int seqlen_q, seqlen_k;
        int num_m_blocks;
        int qhead_per_khead;
        float* ptr_dq_accum;
        int64_t dq_accum_batch_stride, dq_accum_row_stride, dq_accum_head_stride;
    };

    static Params to_underlying_arguments(Arguments const& a) {
        return {a.ptr_Q, a.q_batch_stride, a.q_row_stride, a.q_head_stride,
                a.ptr_K, a.k_batch_stride, a.k_row_stride, a.k_head_stride,
                a.ptr_V, a.v_batch_stride, a.v_row_stride, a.v_head_stride,
                a.ptr_SFQ, a.sfq_batch_stride, a.sfq_head_stride, a.sfq_row_stride,
                a.ptr_SFK, a.sfk_batch_stride, a.sfk_head_stride, a.sfk_row_stride,
                a.ptr_dO, a.do_batch_stride, a.do_row_stride, a.do_head_stride,
                a.ptr_LSE_log2, a.ptr_dPsum,
                a.lse_batch_stride, a.lse_head_stride,
                a.softmax_scale, float(a.softmax_scale * M_LOG2E),
                a.seqlen_q, a.seqlen_k,
                cute::ceil_div(a.seqlen_q, kBlockM),
                a.num_heads / a.num_heads_kv,
                a.ptr_dq_accum, a.dq_accum_batch_stride, a.dq_accum_row_stride, a.dq_accum_head_stride};
    }

    // ====== GMEM BF16 -> FP8 convert + pad: reads BF16 from GMEM, writes FP8 to swizzled SMEM ======
    CUTLASS_DEVICE static void convert_gmem_bf16_to_fp8(
        uint8_t const* __restrict__ gmem_bf16,
        int64_t gmem_row_stride_bytes,
        uint8_t* __restrict__ dst_fp8,
        int rows, int pad_rows, int cols, int row_offset, int seqlen_q,
        int tid, int nthreads)
    {
        int total = pad_rows * cols;
        for (int i = tid; i < total; i += nthreads) {
            int r = i / cols;
            int c = i % cols;
            uint8_t fp8_byte;
            if (r < rows && (row_offset + r) < seqlen_q) {
                cutlass::bfloat16_t bf16_val = *reinterpret_cast<cutlass::bfloat16_t const*>(
                    gmem_bf16 + (row_offset + r) * gmem_row_stride_bytes + c * 2);
                cutlass::float_e4m3_t fp8_val = static_cast<cutlass::float_e4m3_t>(static_cast<float>(bf16_val));
                fp8_byte = reinterpret_cast<uint8_t const&>(fp8_val);
            } else {
                fp8_byte = 0;
            }
            int dst_off = r * cols + c;
            int dst_sw = dst_off ^ (((dst_off >> 7) & 7) << 4);
            dst_fp8[dst_sw] = fp8_byte;
        }
    }

    // ====== GMEM BF16 -> FP8 transpose: reads BF16 from GMEM, writes transposed FP8 to swizzled SMEM ======
    CUTLASS_DEVICE static void transpose_gmem_bf16_to_fp8(
        uint8_t const* __restrict__ gmem_bf16,
        int64_t gmem_row_stride_bytes,
        uint8_t* __restrict__ dst_fp8,
        int rows, int cols, int row_offset, int seqlen_q,
        int tid, int nthreads)
    {
        static constexpr int kPadK = 128;
        int const out_rows = cols;
        int const total = out_rows * kPadK;
        for (int i = tid; i < total; i += nthreads) {
            int const dr = i / kPadK;
            int const dc = i % kPadK;
            uint8_t fp8_byte;
            if (dc < rows && (row_offset + dc) < seqlen_q) {
                // Read src[dc][dr] BF16 from GMEM (transposed access)
                cutlass::bfloat16_t bf16_val = *reinterpret_cast<cutlass::bfloat16_t const*>(
                    gmem_bf16 + (row_offset + dc) * gmem_row_stride_bytes + dr * 2);
                cutlass::float_e4m3_t fp8_val = static_cast<cutlass::float_e4m3_t>(static_cast<float>(bf16_val));
                fp8_byte = reinterpret_cast<uint8_t const&>(fp8_val);
            } else {
                fp8_byte = 0;
            }
            int dst_off = dr * kPadK + dc;
            int dst_sw = dst_off ^ (((dst_off >> 7) & 7) << 4);
            dst_fp8[dst_sw] = fp8_byte;
        }
    }

    // ====== FP8 transpose: [128 x 128] FP8 SW128 -> [128 x 128] FP8 SW128 ======
    CUTLASS_DEVICE static void transpose_fp8_swizzled(
        uint8_t const* __restrict__ src_fp8,
        uint8_t* __restrict__ dst_fp8,
        int tid, int nthreads)
    {
        static constexpr int N = 128;
        static constexpr int total = N * N;
        for (int i = tid; i < total; i += nthreads) {
            int const r = i / N;
            int const c = i % N;
            int src_off = r * N + c;
            int src_sw = src_off ^ (((src_off >> 7) & 7) << 4);
            uint8_t val = src_fp8[src_sw];
            int dst_off = c * N + r;
            int dst_sw = dst_off ^ (((dst_off >> 7) & 7) << 4);
            dst_fp8[dst_sw] = val;
        }
    }

    // ====== Main backward function ======
    template <typename FrgTensorDK, typename FrgTensorDV>
    CUTLASS_DEVICE void mha_bwd(
        Params const& p, TensorStorage& s,
        FrgTensorDK& tOrDK, FrgTensorDV& tOrDV,
        int n_block, int bidb, int bidh_kv, int tid)
    {
        int const n_start = n_block * kBlockN;

        // ====== Load K (FP8 swizzled, resident) ======
        FwdMainloop::load_tile_swizzled(s.smem_k.data(),
            reinterpret_cast<uint8_t const*>(p.ptr_K)
                + bidb * p.k_batch_stride + bidh_kv * p.k_head_stride,
            kBlockN, kHeadDim, p.k_row_stride, n_start, tid, NumMmaThreads);

        // ====== Load V (FP8 swizzled, resident) ======
        FwdMainloop::load_tile_swizzled(s.smem_vt.data(),
            reinterpret_cast<uint8_t const*>(p.ptr_V)
                + bidb * p.v_batch_stride + bidh_kv * p.v_head_stride,
            kBlockN, kHeadDim, p.v_row_stride, n_start, tid, NumMmaThreads);

        // ====== Load SFK (resident) ======
        FwdMainloop::load_sf(s.smem_sfk.data(),
            p.ptr_SFK + bidb * p.sfk_batch_stride + bidh_kv * p.sfk_head_stride,
            kBlockN, kSFCols, p.sfk_row_stride, n_start, tid, NumMmaThreads);
        __syncthreads();

        // ====== Set up GEMM-1 K partition (resident) ======
        TiledMma_G1 tiled_mma_g1;
        auto thread_mma_g1 = tiled_mma_g1.get_thread_slice(tid);
        auto tile_shape_g1 = tile_shape(tiled_mma_g1);

        Tensor sK = make_tensor(make_smem_ptr(s.smem_k.data()), SmemLayoutK{});
        Tensor sK_pi = cute::as_position_independent_swizzle_tensor(sK);
        auto sK_flat = make_tensor(make_smem_ptr(s.smem_k.data()),
            make_layout(make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
        Tensor tCrK = thread_mma_g1.partition_fragment_B(sK_flat);
        auto copy_B_g1 = make_tiled_copy_B(SmemCopyAtomB_G1{}, tiled_mma_g1);
        auto thr_copy_B_g1 = copy_B_g1.get_thread_slice(tid);
        Tensor tCsK = thr_copy_B_g1.partition_S(sK_pi);
        Tensor tCrK_v = thr_copy_B_g1.retile_D(tCrK);

        // SFK partition (resident)
        Tensor sSFK = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(s.smem_sfk.data())),
            SmemLayoutAtomSFK{});
        Tensor tCrSFK = sm120_partition_fragment_SFB(sSFK, thread_mma_g1);
        auto sfk_copy = make_tiled_copy_impl(SmemCopyAtomSF{},
            sm120_get_layoutSFB_TV(tiled_mma_g1),
            make_shape(size<1>(tile_shape_g1), size<2>(tile_shape_g1)));
        auto sfk_thr = sfk_copy.get_thread_slice(tid);
        Tensor tCsSFK = sfk_thr.partition_S(sSFK);
        Tensor tCrSFK_v = sfk_thr.retile_D(tCrSFK);

        // ====== Set up GEMM-2 V partition (resident) ======
        Tensor sV = make_tensor(make_smem_ptr(s.smem_vt.data()), SmemLayoutVt{});
        Tensor sV_pi = cute::as_position_independent_swizzle_tensor(sV);
        auto sV_flat = make_tensor(make_smem_ptr(s.smem_vt.data()),
            make_layout(make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
        Tensor tCrV = thread_mma_g1.partition_fragment_B(sV_flat);
        auto copy_B_V = make_tiled_copy_B(SmemCopyAtomB_G1{}, tiled_mma_g1);
        auto thr_copy_B_V = copy_B_V.get_thread_slice(tid);
        Tensor tCsV = thr_copy_B_V.partition_S(sV_pi);
        Tensor tCrV_v = thr_copy_B_V.retile_D(tCrV);
        for (int k = 0; k < size<2>(tCrV_v); ++k) copy(copy_B_V, tCsV(_,_,k), tCrV_v(_,_,k));

        // ====== V identity SFs (resident) ======
        FwdMainloop::fill_identity_sf(s.smem_sfk.data(), cute::cosize_v<SmemLayoutAtomSFK>,
            tid, NumMmaThreads);
        __syncthreads();

        Tensor sSFV = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(s.smem_sfk.data())),
            SmemLayoutAtomSFK{});
        Tensor tCrSFV = sm120_partition_fragment_SFB(sSFV, thread_mma_g1);
        auto sfv_copy = make_tiled_copy_impl(SmemCopyAtomSF{},
            sm120_get_layoutSFB_TV(tiled_mma_g1),
            make_shape(size<1>(tile_shape_g1), size<2>(tile_shape_g1)));
        auto sfv_thr = sfv_copy.get_thread_slice(tid);
        Tensor tCsSFV = sfv_thr.partition_S(sSFV);
        Tensor tCrSFV_v = sfv_thr.retile_D(tCrSFV);
        for (int k = 0; k < size<2>(tCrSFV_v); ++k) copy(sfv_copy, tCsSFV(_,_,k), tCrSFV_v(_,_,k));

        // ====== Set up shared A-partition from smem_q ======
        Tensor sQ = make_tensor(make_smem_ptr(s.smem_q.data()), SmemLayoutQ{});
        Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
        auto sQ_flat = make_tensor(make_smem_ptr(s.smem_q.data()),
            make_layout(make_shape(Int<kBlockM_SF>{}, Int<kHeadDim>{})));
        Tensor tCrQ = thread_mma_g1.partition_fragment_A(sQ_flat);
        auto copy_A_g1 = make_tiled_copy_A(SmemCopyAtomA_G1{}, tiled_mma_g1);
        auto thr_copy_A_g1 = copy_A_g1.get_thread_slice(tid);
        Tensor tCsQ = thr_copy_A_g1.partition_S(sQ_pi);
        Tensor tCrQ_v = thr_copy_A_g1.retile_D(tCrQ);

        // ====== Set up SFQ partition from smem_sfq ======
        Tensor sSFQ = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(s.smem_sfq.data())),
            SmemLayoutAtomSFQ{});
        Tensor tCrSFQ = sm120_partition_fragment_SFA(sSFQ, thread_mma_g1);
        auto sfq_copy = make_tiled_copy_impl(SmemCopyAtomSF{},
            sm120_get_layoutSFA_TV(tiled_mma_g1),
            make_shape(size<0>(tile_shape_g1), size<2>(tile_shape_g1)));
        auto sfq_thr = sfq_copy.get_thread_slice(tid);
        Tensor tCsSFQ = sfq_thr.partition_S(sSFQ);
        Tensor tCrSFQ_v = sfq_thr.retile_D(tCrSFQ);

        // ====== Set up shared B-partition for GEMM-3/4 ======
        Tensor sPds_fp8 = make_tensor(make_smem_ptr(s.smem_pds.data()), SmemLayoutK{});
        Tensor sPds_fp8_pi = cute::as_position_independent_swizzle_tensor(sPds_fp8);
        auto sPds_flat = make_tensor(make_smem_ptr(s.smem_pds.data()),
            make_layout(make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
        Tensor tCrB34 = thread_mma_g1.partition_fragment_B(sPds_flat);
        auto copy_B_g34 = make_tiled_copy_B(SmemCopyAtomB_G1{}, tiled_mma_g1);
        auto thr_copy_B_g34 = copy_B_g34.get_thread_slice(tid);
        Tensor tCsPds = thr_copy_B_g34.partition_S(sPds_fp8_pi);
        Tensor tCrB34_v = thr_copy_B_g34.retile_D(tCrB34);

        Tensor sPt_fp8 = make_tensor(make_smem_ptr(s.smem_pt.data()), SmemLayoutK{});
        Tensor sPt_fp8_pi = cute::as_position_independent_swizzle_tensor(sPt_fp8);
        Tensor tCsPt = thr_copy_B_g34.partition_S(sPt_fp8_pi);

        // ====== Set up A-partition for GEMM-5: dS from smem_pds ======
        Tensor sPds_A = make_tensor(make_smem_ptr(s.smem_pds.data()), SmemLayoutQ{});
        Tensor sPds_A_pi = cute::as_position_independent_swizzle_tensor(sPds_A);
        Tensor tCsPds_A = thr_copy_A_g1.partition_S(sPds_A_pi);

        int const warp_idx = tid / 32;
        int const lane_idx = tid % 32;
        int const lane_row = lane_idx / 4;
        int const lane_col = lane_idx % 4;

        // ====== M-block loop ======
        int const m_block_min = Is_causal ? (n_block * kBlockN / kBlockM) : 0;
        for (int m_blk = m_block_min; m_blk < p.num_m_blocks; ++m_blk) {
            int const m_start = m_blk * kBlockM;

            for (int qh_off = 0; qh_off < p.qhead_per_khead; ++qh_off) {
                int const bidh = bidh_kv * p.qhead_per_khead + qh_off;

                // ====== Phase A: Load Q + SFQ (no dO — read from GMEM later) ======
                FwdMainloop::load_tile_swizzled(s.smem_q.data(),
                    reinterpret_cast<uint8_t const*>(p.ptr_Q)
                        + bidb * p.q_batch_stride + bidh * p.q_head_stride,
                    kBlockM, kHeadDim, p.q_row_stride, m_start, tid, NumMmaThreads);

                FwdMainloop::fill_identity_sf(s.smem_sfq.data(), cute::cosize_v<SmemLayoutAtomSFQ>,
                    tid, NumMmaThreads);
                FwdMainloop::load_sf(s.smem_sfq.data(),
                    p.ptr_SFQ + bidb * p.sfq_batch_stride + bidh * p.sfq_head_stride,
                    kBlockM, kSFCols, p.sfq_row_stride, m_start, tid, NumMmaThreads);

                __syncthreads();  // sync #1: Q + SFQ visible

                // ====== Phase B: GEMM-1 (S = Q@K^T) ======
                for (int k = 0; k < size<2>(tCrQ_v); ++k) copy(copy_A_g1, tCsQ(_,_,k), tCrQ_v(_,_,k));
                for (int k = 0; k < size<2>(tCrSFQ_v); ++k) copy(sfq_copy, tCsSFQ(_,_,k), tCrSFQ_v(_,_,k));
                for (int k = 0; k < size<2>(tCrK_v); ++k) copy(copy_B_g1, tCsK(_,_,k), tCrK_v(_,_,k));
                for (int k = 0; k < size<2>(tCrSFK_v); ++k) copy(sfk_copy, tCsSFK(_,_,k), tCrSFK_v(_,_,k));

                Tensor acc_s = partition_fragment_C(tiled_mma_g1, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
                clear(acc_s);
                for (int k = 0; k < size<2>(tCrQ); ++k) {
                    cute::gemm(tiled_mma_g1,
                        make_zip_tensor(tCrQ(_,_,k), tCrSFQ(_,_,k)),
                        make_zip_tensor(tCrK(_,_,k), tCrSFK(_,_,k)),
                        acc_s);
                }

                // ====== Phase B': Fill smem_sfq with identity SFs ======
                FwdMainloop::fill_identity_sf(s.smem_sfq.data(), cute::cosize_v<SmemLayoutAtomSFQ>,
                    tid, NumMmaThreads);

                // ====== Phase C: Causal mask + seqlen mask + softmax ======
                if constexpr (Is_causal) {
                    if (n_start >= m_start) {
                        Tensor acc_s_rc = make_tensor(acc_s.data(),
                            flash::convert_layout_acc_rowcol(acc_s.layout()));
                        int const wm = warp_idx;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(acc_s_rc); ++mi) {
                            int const row = m_start + wm * 16 + (mi / 2) * 16 + lane_row + (mi % 2) * 8;
                            #pragma unroll
                            for (int ni = 0; ni < size<1>(acc_s_rc); ni += 2) {
                                int const col = n_start + (ni / 2) * 8 + lane_col * 2;
                                if (col > row) { acc_s_rc(mi, ni) = -INFINITY; acc_s_rc(mi, ni + 1) = -INFINITY; }
                                else if (col + 1 > row) { acc_s_rc(mi, ni + 1) = -INFINITY; }
                            }
                        }
                    }
                }

                // Mask out-of-bounds rows (>= seqlen_q) to -inf
                {
                    Tensor acc_mask_rc = make_tensor(acc_s.data(),
                        flash::convert_layout_acc_rowcol(acc_s.layout()));
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(acc_mask_rc); ++mi) {
                        int const row = m_start + warp_idx * 16 + (mi / 2) * 16 + lane_row + (mi % 2) * 8;
                        if (row >= p.seqlen_q) {
                            #pragma unroll
                            for (int ni = 0; ni < size<1>(acc_mask_rc); ++ni) {
                                acc_mask_rc(mi, ni) = -INFINITY;
                            }
                        }
                    }
                }

                // Softmax: P = exp2(S * scale_log2 - LSE_log2)
                Tensor acc_s_rc = make_tensor(acc_s.data(),
                    flash::convert_layout_acc_rowcol(acc_s.layout()));
                static constexpr int nrow_s = decltype(size<0>(acc_s_rc))::value;
                static constexpr int ncol_s = decltype(size<1>(acc_s_rc))::value;

                float lse_log2_regs[nrow_s];
                float dpsum_regs[nrow_s];
                {
                    int const wm = warp_idx;
                    #pragma unroll
                    for (int mi = 0; mi < nrow_s; ++mi) {
                        int const row = m_start + wm * 16 + (mi / 2) * 16 + lane_row + (mi % 2) * 8;
                        int const idx = bidb * p.lse_batch_stride + bidh * p.lse_head_stride + row;
                        bool valid = (row < p.seqlen_q);
                        lse_log2_regs[mi] = valid ? p.ptr_LSE_log2[idx] : INFINITY;
                        dpsum_regs[mi] = valid ? p.ptr_dPsum[idx] : 0.0f;
                    }
                }

                #pragma unroll
                for (int mi = 0; mi < nrow_s; ++mi) {
                    #pragma unroll
                    for (int ni = 0; ni < ncol_s; ++ni) {
                        float sv = acc_s_rc(mi, ni);
                        float pv = (sv == -INFINITY) ? 0.0f : exp2f(sv * p.softmax_scale_log2 - lse_log2_regs[mi]);
                        acc_s_rc(mi, ni) = pv;
                    }
                }

                // ====== Phase D: Q→Q^T (smem_q → smem_pt) ======
                transpose_fp8_swizzled(
                    s.smem_q.data(), s.smem_pt.data(),
                    tid, NumMmaThreads);

                // ====== Phase H: dO BF16 (GMEM) → dO_fp8 (smem_pds) ======
                {
                    auto const* dO_gmem = reinterpret_cast<uint8_t const*>(p.ptr_dO)
                        + bidb * p.do_batch_stride * 2 + bidh * p.do_head_stride * 2;
                    int64_t dO_row_stride_bytes = p.do_row_stride * 2;
                    convert_gmem_bf16_to_fp8(dO_gmem, dO_row_stride_bytes, s.smem_pds.data(),
                        kBlockM, kBlockM_SF, kHeadDim, m_start, p.seqlen_q, tid, NumMmaThreads);
                }
                __syncthreads();  // sync #2: D+H done (smem_pt has Q^T, smem_pds has dO_fp8)

                // Reload identity SFs
                for (int k = 0; k < size<2>(tCrSFQ_v); ++k) copy(sfq_copy, tCsSFQ(_,_,k), tCrSFQ_v(_,_,k));

                // ====== Phase J: GEMM-2: dP = dO_fp8(smem_pds) @ V(regs) ======
                {
                    for (int k = 0; k < size<2>(tCrQ_v); ++k) copy(copy_A_g1, tCsPds_A(_,_,k), tCrQ_v(_,_,k));

                    Tensor acc_dp = partition_fragment_C(tiled_mma_g1, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
                    clear(acc_dp);
                    for (int k = 0; k < size<2>(tCrQ); ++k) {
                        cute::gemm(tiled_mma_g1,
                            make_zip_tensor(tCrQ(_,_,k), tCrSFQ(_,_,k)),
                            make_zip_tensor(tCrV(_,_,k), tCrSFV(_,_,k)),
                            acc_dp);
                    }

                    // ====== Phase E: Scatter P → smem_q as FP8 transposed ======
                    {
                        int const wm = warp_idx;
                        auto* dst = s.smem_q.data();
                        #pragma unroll
                        for (int mi = 0; mi < nrow_s; ++mi) {
                            int const row = wm * 16 + (mi / 2) * 16 + lane_row + (mi % 2) * 8;
                            #pragma unroll
                            for (int ni = 0; ni < ncol_s; ni += 2) {
                                int const col = (ni / 2) * 8 + lane_col * 2;
                                #pragma unroll
                                for (int i = 0; i < 2; ++i) {
                                    cutlass::float_e4m3_t fp8_val = static_cast<cutlass::float_e4m3_t>(acc_s_rc(mi, ni + i));
                                    uint8_t fp8_byte = reinterpret_cast<uint8_t const&>(fp8_val);
                                    int dst_off = (col + i) * 128 + row;
                                    int dst_sw = dst_off ^ (((dst_off >> 7) & 7) << 4);
                                    dst[dst_sw] = fp8_byte;
                                }
                            }
                        }
                    }

                    // ====== Phase K: dS = P * (dP - dPsum) * scale ======
                    Tensor acc_dp_rc = make_tensor(acc_dp.data(),
                        flash::convert_layout_acc_rowcol(acc_dp.layout()));
                    #pragma unroll
                    for (int mi = 0; mi < nrow_s; ++mi) {
                        #pragma unroll
                        for (int ni = 0; ni < ncol_s; ++ni) {
                            float pv = acc_s_rc(mi, ni);
                            float dpv = acc_dp_rc(mi, ni);
                            acc_s_rc(mi, ni) = pv * (dpv - dpsum_regs[mi]) * p.softmax_scale;
                        }
                    }
                }

                // ====== Phase F: dO BF16 (GMEM) → dO^T FP8 → smem_pds ======
                {
                    auto const* dO_gmem = reinterpret_cast<uint8_t const*>(p.ptr_dO)
                        + bidb * p.do_batch_stride * 2 + bidh * p.do_head_stride * 2;
                    int64_t dO_row_stride_bytes = p.do_row_stride * 2;
                    transpose_gmem_bf16_to_fp8(dO_gmem, dO_row_stride_bytes, s.smem_pds.data(),
                        kBlockM, kHeadDim, m_start, p.seqlen_q, tid, NumMmaThreads);
                }
                __syncthreads();  // sync #3: E+F done (smem_q has P^T, smem_pds has dO^T)

                // ====== Phase G: GEMM-3: dV += P^T_fp8(smem_q) @ dO^T_fp8(smem_pds) ======
                static constexpr int kIters34 = kBlockM / 32;  // = 4 (all K-chunks valid)
                {
                    for (int k = 0; k < kIters34; ++k) copy(copy_A_g1, tCsQ(_,_,k), tCrQ_v(_,_,k));
                    for (int k = 0; k < kIters34; ++k) copy(copy_B_g34, tCsPds(_,_,k), tCrB34_v(_,_,k));
                    for (int k = 0; k < kIters34; ++k) {
                        cute::gemm(tiled_mma_g1,
                            make_zip_tensor(tCrQ(_,_,k), tCrSFQ(_,_,k)),
                            make_zip_tensor(tCrB34(_,_,k), tCrSFV(_,_,k)),
                            tOrDV);
                    }
                }
                __syncthreads();  // sync #4: GEMM-3 reads done

                // ====== Phase LM (dual scatter): dS → smem_q (transposed) + smem_pds (non-transposed) ======
                {
                    int const wm = warp_idx;
                    auto* dst_q = s.smem_q.data();
                    auto* dst_pds = s.smem_pds.data();
                    #pragma unroll
                    for (int mi = 0; mi < nrow_s; ++mi) {
                        int const row = wm * 16 + (mi / 2) * 16 + lane_row + (mi % 2) * 8;
                        #pragma unroll
                        for (int ni = 0; ni < ncol_s; ni += 2) {
                            int const col = (ni / 2) * 8 + lane_col * 2;
                            #pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                cutlass::float_e4m3_t fp8_val = static_cast<cutlass::float_e4m3_t>(acc_s_rc(mi, ni + i));
                                uint8_t fp8_byte = reinterpret_cast<uint8_t const&>(fp8_val);
                                int dst_off_t = (col + i) * 128 + row;
                                int dst_sw_t = dst_off_t ^ (((dst_off_t >> 7) & 7) << 4);
                                dst_q[dst_sw_t] = fp8_byte;
                                int dst_off_n = row * 128 + (col + i);
                                int dst_sw_n = dst_off_n ^ (((dst_off_n >> 7) & 7) << 4);
                                dst_pds[dst_sw_n] = fp8_byte;
                            }
                        }
                    }
                }
                __syncthreads();  // sync #5: dS^T in smem_q + dS in smem_pds visible

                // ====== GEMM-5: dQ_local = dS_fp8(smem_pds) @ K_fp8(smem_k), then atomicAdd ======
                {
                    for (int k = 0; k < size<2>(tCrQ_v); ++k) copy(copy_A_g1, tCsPds_A(_,_,k), tCrQ_v(_,_,k));
                    for (int k = 0; k < size<2>(tCrK_v); ++k) copy(copy_B_g1, tCsK(_,_,k), tCrK_v(_,_,k));

                    Tensor acc_dq = partition_fragment_C(tiled_mma_g1, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));
                    clear(acc_dq);
                    for (int k = 0; k < size<2>(tCrQ); ++k) {
                        cute::gemm(tiled_mma_g1,
                            make_zip_tensor(tCrQ(_,_,k), tCrSFQ(_,_,k)),
                            make_zip_tensor(tCrK(_,_,k), tCrSFV(_,_,k)),
                            acc_dq);
                    }

                    Tensor acc_dq_rc = make_tensor(acc_dq.data(),
                        flash::convert_layout_acc_rowcol(acc_dq.layout()));
                    float* dq_base = p.ptr_dq_accum + bidb * p.dq_accum_batch_stride + bidh * p.dq_accum_head_stride;
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(acc_dq_rc); ++mi) {
                        int const row = m_start + warp_idx * 16 + (mi / 2) * 16 + lane_row + (mi % 2) * 8;
                        if (row >= p.seqlen_q) continue;
                        #pragma unroll
                        for (int ni = 0; ni < size<1>(acc_dq_rc); ++ni) {
                            int const col = (ni / 2) * 8 + lane_col * 2 + (ni % 2);
                            if (col < kHeadDim) {
                                atomicAdd(&dq_base[row * p.dq_accum_row_stride + col], acc_dq_rc(mi, ni));
                            }
                        }
                    }
                }

                // ====== Phase N: GEMM-4: dK += dS^T_fp8(smem_q) @ Q^T_fp8(smem_pt) ======
                {
                    for (int k = 0; k < kIters34; ++k) copy(copy_A_g1, tCsQ(_,_,k), tCrQ_v(_,_,k));
                    for (int k = 0; k < kIters34; ++k) copy(copy_B_g34, tCsPt(_,_,k), tCrB34_v(_,_,k));
                    for (int k = 0; k < kIters34; ++k) {
                        cute::gemm(tiled_mma_g1,
                            make_zip_tensor(tCrQ(_,_,k), tCrSFQ(_,_,k)),
                            make_zip_tensor(tCrB34(_,_,k), tCrSFV(_,_,k)),
                            tOrDK);
                    }
                }

                __syncthreads();  // sync #6: GEMM-4 done, safe to overwrite smem_q/smem_pt next iteration
            }  // GQA head loop
        }  // M-block loop
    }
};

}  // namespace flash
