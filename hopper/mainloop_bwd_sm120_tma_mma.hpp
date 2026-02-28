/******************************************************************************
 * SM120 MXFP8 Block-Scaled Flash Attention Backward Mainloop (Phase 1: dK+dV)
 *
 * Each CTA owns one N-block (K/V block). Grid: (n_blocks, heads_kv, batch).
 * dK and dV accumulate in FP32 registers across all M-blocks, then write as BF16.
 *
 * Hybrid FP8/BF16 design:
 *   GEMM-1 (S = Q @ K^T): FP8 block-scaled MMA (must match forward for exact P)
 *   GEMM-2 (dP = dO @ V^T): SM80-style BF16 MMA (V^T FP8→BF16 in regs)
 *   GEMM-3 (dV += P^T @ dO): SM80-style BF16 MMA (P^T from transposed SMEM)
 *   GEMM-4 (dK += dS^T @ Q): SM80-style BF16 MMA (Q FP8→BF16 in regs)
 *
 * SMEM budget (~88 KB, fits in 101 KB):
 *   K [128x128 FP8 SW128]:  16 KB (resident)
 *   V^T [128x128 FP8 SW128]: 16 KB (resident, transposed from V at start)
 *   SFK [256B]:               <1 KB (resident)
 *   Q [64x128 FP8 SW128]:    8 KB (per M-block)
 *   SFQ [256B]:               <1 KB (per M-block)
 *   dO [64x128 BF16 SW128]: 16 KB (per M-block)
 *   P/dS [64x128 BF16 SW128]: 16 KB (scratch, per M-block)
 *   P^T/dS^T [128x64 BF16 SW128]: 16 KB (scratch, per M-block)
 *
 * FP8→BF16 for V^T and Q: done via register conversion during LDSM
 * (load FP8 bytes, convert in regs, feed to BF16 MMA) — no extra SMEM needed.
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

    static constexpr int kBlockM = get<0>(TileShape_MNK{});  // 64
    static constexpr int kBlockN = get<1>(TileShape_MNK{});  // 128
    static constexpr int kHeadDim = get<2>(TileShape_MNK{}); // 128

    // Padded M for GEMM-1 block-scaled MMA: must be >= 128 (Blk_MN constraint)
    static constexpr int kBlockM_SF = kBlockN;  // = 128

    // ====== Block-scaled MMA types (GEMM-1: S = Q @ K^T) ======
    using ElementSF = cutlass::float_ue8m0_t;
    static constexpr int kSFVecSize = 32;
    static constexpr int kSFCols = kHeadDim / kSFVecSize;

    using MmaAtomOp = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
        cutlass::float_ue8m0_t, kSFVecSize>;
    // 8 warps along M, 1 along N — matches forward kernel for SF compatibility
    // GEMM-1 uses padded kBlockM_SF=128 for MMA (upper 64 rows produce unused S values)
    using AtomLayoutMNK_G1 = Layout<Shape<_8, _1, _1>>;
    using PermTileM_G1 = decltype(cute::min(Int<kBlockM_SF>{}, _128{}));  // 128
    using TiledMma_G1 = decltype(cute::make_tiled_mma(
        MMA_Atom<MmaAtomOp>{}, AtomLayoutMNK_G1{},
        Tile<PermTileM_G1, _8, _32>{}));

    // SM80 BF16 MMA for GEMMs 2-4
    using MmaAtomOp_BF16 = SM80_16x8x16_F32BF16BF16F32_TN;

    // GEMM-2: dP[kBlockM×kBlockN] = dO[kBlockM×D] @ V[kBlockN×D]^T
    // 4×2×1 warp layout: 4 warps along M (4×16=64=kBlockM), 2 along N.
    // All 8 warps active. Different layout from GEMM-1 (8×1×1), so dS computation
    // reads P from saved P^T in smem_q.
    using TiledMma_G2 = decltype(cute::make_tiled_mma(
        MMA_Atom<MmaAtomOp_BF16>{}, Layout<Shape<_4, _2, _1>>{},
        Tile<Int<kBlockM>, _16, _16>{}));

    // GEMM-3: dV[128xD] += P^T[128x64] @ dO[64xD] — 8x1x1
    using TiledMma_G3 = decltype(cute::make_tiled_mma(
        MMA_Atom<MmaAtomOp_BF16>{}, Layout<Shape<_8, _1, _1>>{},
        Tile<_128, _16, _16>{}));

    // GEMM-4: dK[128xD] += dS^T[128x64] @ Q[64xD] — 8x1x1
    using TiledMma_G4 = decltype(cute::make_tiled_mma(
        MMA_Atom<MmaAtomOp_BF16>{}, Layout<Shape<_8, _1, _1>>{},
        Tile<_128, _16, _16>{}));

    static constexpr int NumMmaThreads = 256;

    // ====== Swizzled SMEM Layouts ======
    using SmemLayoutAtomSW_FP8 = SM90::GMMA::Layout_K_SW128_Atom<uint8_t>;
    using SmemLayoutAtomSW_BF16 = SM90::GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>;

    // Resident tiles
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomSW_FP8{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomSW_FP8{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));
    using SmemLayoutV = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;

    // Per M-block tiles
    // Q SMEM: padded to kBlockM_SF=128 rows for GEMM-1 block-scaled MMA compatibility
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomSW_FP8{}, Shape<Int<kBlockM_SF>, Int<kHeadDim>>{}));
    using SmemLayoutdO = decltype(tile_to_shape(SmemLayoutAtomSW_BF16{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutPdS = decltype(tile_to_shape(SmemLayoutAtomSW_BF16{}, Shape<Int<kBlockM>, Int<kBlockN>>{}));
    using SmemLayoutPt = decltype(tile_to_shape(SmemLayoutAtomSW_BF16{}, Shape<Int<kBlockN>, Int<kBlockM>>{}));
    // Transposed dO and Q layouts for GEMM-3/4 B-operands
    // dO^T [kHeadDim × kBlockM BF16] and Q^T [kHeadDim × kBlockM BF16]
    using SmemLayoutdOt = decltype(tile_to_shape(SmemLayoutAtomSW_BF16{}, Shape<Int<kHeadDim>, Int<kBlockM>>{}));
    using SmemLayoutQt = SmemLayoutdOt;  // Same shape: [128 × 64 BF16]

    // V BF16 [kBlockN × kHeadDim] for GEMM-2 B-operand (32 KB, spans smem_pds + smem_pt)
    using SmemLayoutVBF16 = decltype(tile_to_shape(SmemLayoutAtomSW_BF16{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    // dO padded to kBlockM_SF=128 rows for GEMM-2 A-operand (same M as GEMM-1)
    using SmemLayoutdO_padded = decltype(tile_to_shape(SmemLayoutAtomSW_BF16{}, Shape<Int<kBlockM_SF>, Int<kHeadDim>>{}));

    // ====== SF layouts ======
    // SFQ uses padded kBlockM_SF=128 to satisfy Blk_MN=128 constraint.
    // We allocate SF SMEM for 128 rows but only load SFs for kBlockM=64 valid rows;
    // the extra rows are filled with identity SF (127 = 2^0 = 1.0).
    static constexpr int MMA_NSF = 32 / kSFVecSize;
    using Sm1xxCfg = cutlass::detail::Sm1xxBlockScaledConfig<kSFVecSize>;
    using Blk_MN = typename Sm1xxCfg::Blk_MN;
    using Blk_SF = typename Sm1xxCfg::Blk_SF;
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
    using mnBBS  = Shape<_32, _4>;
    using mnBBSt = Stride<_16, _4>;
    using kBBS  = Shape<Int<kSFVecSize>, Int<MMA_NSF>>;
    using kBBSt = Stride<_0, _1>;

    // SFQ uses padded kBlockM_SF=128 (= kBlockN) to satisfy Blk_MN=128
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
    using SmemCopyAtomA_BF16 = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::bfloat16_t>;
    using SmemCopyAtomB_BF16 = Copy_Atom<SM75_U32x2_LDSM_N, cutlass::bfloat16_t>;

    // Forward mainloop type for static helpers
    using FwdMainloop = CollectiveMainloopFwdSm120<1,
        cute::Shape<Int<kBlockN>, Int<kBlockN>, Int<kHeadDim>>,  // Use kBlockN for both M,N (for load helpers)
        Element, float, false, false, false>;

    // ====== Shared Memory (must fit in 100 KB) ======
    // V (16K) is loaded once and transposed to V^T before M-loop, then freed.
    // Q (16K padded) is loaded per M-block during the M-loop.
    // Since V and Q are never needed simultaneously, they share memory via union.
    // Total: 16+16+1+16+1+16+16+16 = 98 KB (fits!)
    struct TensorStorage : cute::aligned_struct<128> {
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutK>> smem_k;            // 16 KB
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutVt>> smem_vt;           // 16 KB
        cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutAtomSFK>> smem_sfk;                   // ~512 B
        union {
            // Before M-loop: V loaded here for transpose
            alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutV>> smem_v;         // 16 KB
            // During M-loop: Q (padded to 128 rows) + SFQ
            struct {
                alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutQ>> smem_q;     // 16 KB
                cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutAtomSFQ>> smem_sfq;           // ~512 B
            };
        };
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutdO> * 2> smem_do;       // 16 KB (BF16)
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutPdS> * 2> smem_pds;     // 16 KB (BF16)
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutPt> * 2> smem_pt;       // 16 KB (BF16)
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
                a.num_heads / a.num_heads_kv};
    }

    // ====== BF16 transpose: [rows x cols] → [cols x rows] (both SW128 swizzled) ======
    CUTLASS_DEVICE static void transpose_bf16_smem(
        uint8_t const* __restrict__ src, uint8_t* __restrict__ dst,
        int rows, int cols, int tid, int nthreads)
    {
        int blocks_r = rows / 2;
        int blocks_c = cols / 2;
        int total = blocks_r * blocks_c;
        for (int blk = tid; blk < total; blk += nthreads) {
            int br = blk / blocks_c, bc = blk % blocks_c;
            int r = br * 2, c = bc * 2;
            uint16_t vals[2][2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int sb = ((r + i) * cols + c) * 2;
                int sw = sb ^ (((sb >> 7) & 7) << 4);
                uint32_t pair = *reinterpret_cast<uint32_t const*>(src + sw);
                vals[i][0] = pair & 0xFFFF;
                vals[i][1] = pair >> 16;
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int db = ((c + i) * rows + r) * 2;
                int dw = db ^ (((db >> 7) & 7) << 4);
                *reinterpret_cast<uint32_t*>(dst + dw) =
                    uint32_t(vals[0][i]) | (uint32_t(vals[1][i]) << 16);
            }
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

        // ====== Load V (FP8 non-swizzled) and transpose to V^T (FP8 swizzled) ======
        FwdMainloop::load_tile(s.smem_v.data(),
            reinterpret_cast<uint8_t const*>(p.ptr_V)
                + bidb * p.v_batch_stride + bidh_kv * p.v_head_stride,
            kBlockN, kHeadDim, p.v_row_stride, n_start, tid, NumMmaThreads);
        __syncthreads();
        FwdMainloop::transpose_v_smem(s.smem_v.data(), s.smem_vt.data(), tid, NumMmaThreads);

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
        for (int k = 0; k < size<2>(tCrSFK_v); ++k) copy(sfk_copy, tCsSFK(_,_,k), tCrSFK_v(_,_,k));
        for (int k = 0; k < size<2>(tCrK_v); ++k) copy(copy_B_g1, tCsK(_,_,k), tCrK_v(_,_,k));

        int const warp_idx = tid / 32;
        int const lane_idx = tid % 32;
        int const lane_row = lane_idx / 4;
        int const lane_col = lane_idx % 4;

        // (acc_ref removed — nrow_s/ncol_s computed inside M-block loop from acc_s_rc)

        // ====== M-block loop ======
        for (int m_blk = 0; m_blk < p.num_m_blocks; ++m_blk) {
            int const m_start = m_blk * kBlockM;

            for (int qh_off = 0; qh_off < p.qhead_per_khead; ++qh_off) {
                int const bidh = bidh_kv * p.qhead_per_khead + qh_off;

                // ====== Load Q (padded to kBlockM_SF=128 rows) + SFQ + dO ======
                // First zero the entire Q SMEM (128 rows) so padded rows are 0
                {
                    int total16 = cute::cosize_v<SmemLayoutQ> / 16;
                    for (int i = tid; i < total16; i += NumMmaThreads)
                        reinterpret_cast<uint4*>(s.smem_q.data())[i] = make_uint4(0,0,0,0);
                }
                // Load only kBlockM=64 valid rows (swizzled)
                FwdMainloop::load_tile_swizzled(s.smem_q.data(),
                    reinterpret_cast<uint8_t const*>(p.ptr_Q)
                        + bidb * p.q_batch_stride + bidh * p.q_head_stride,
                    kBlockM, kHeadDim, p.q_row_stride, m_start, tid, NumMmaThreads);

                // SFQ: fill entire 128-row buffer with identity SF (127), then load 64 valid rows
                FwdMainloop::fill_identity_sf(s.smem_sfq.data(), cute::cosize_v<SmemLayoutAtomSFQ>,
                    tid, NumMmaThreads);
                FwdMainloop::load_sf(s.smem_sfq.data(),
                    p.ptr_SFQ + bidb * p.sfq_batch_stride + bidh * p.sfq_head_stride,
                    kBlockM, kSFCols, p.sfq_row_stride, m_start, tid, NumMmaThreads);

                // Load dO (BF16 swizzled)
                {
                    auto const* src = reinterpret_cast<uint8_t const*>(p.ptr_dO)
                        + bidb * p.do_batch_stride * 2 + bidh * p.do_head_stride * 2;
                    int total = kBlockM * (kHeadDim * 2 / 16);  // 16 bytes per uint4
                    for (int i = tid; i < total; i += NumMmaThreads) {
                        int r = i / (kHeadDim * 2 / 16), c16 = i % (kHeadDim * 2 / 16);
                        int byte_offset = r * kHeadDim * 2 + c16 * 16;
                        int swizzled = byte_offset ^ (((byte_offset >> 7) & 7) << 4);
                        *reinterpret_cast<uint4*>(s.smem_do.data() + swizzled) =
                            *reinterpret_cast<uint4 const*>(src + (m_start + r) * p.do_row_stride * 2 + c16 * 16);
                    }
                }
                __syncthreads();

                // ====== Partition Q + SFQ (using padded 128 rows) ======
                Tensor sQ = make_tensor(make_smem_ptr(s.smem_q.data()), SmemLayoutQ{});
                Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
                auto sQ_flat = make_tensor(make_smem_ptr(s.smem_q.data()),
                    make_layout(make_shape(Int<kBlockM_SF>{}, Int<kHeadDim>{})));
                Tensor tCrQ = thread_mma_g1.partition_fragment_A(sQ_flat);
                auto copy_A_g1 = make_tiled_copy_A(SmemCopyAtomA_G1{}, tiled_mma_g1);
                auto thr_copy_A_g1 = copy_A_g1.get_thread_slice(tid);
                Tensor tCsQ = thr_copy_A_g1.partition_S(sQ_pi);
                Tensor tCrQ_v = thr_copy_A_g1.retile_D(tCrQ);
                for (int k = 0; k < size<2>(tCrQ_v); ++k) copy(copy_A_g1, tCsQ(_,_,k), tCrQ_v(_,_,k));

                Tensor sSFQ = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(s.smem_sfq.data())),
                    SmemLayoutAtomSFQ{});
                Tensor tCrSFQ = sm120_partition_fragment_SFA(sSFQ, thread_mma_g1);
                auto sfq_copy = make_tiled_copy_impl(SmemCopyAtomSF{},
                    sm120_get_layoutSFA_TV(tiled_mma_g1),
                    make_shape(size<0>(tile_shape_g1), size<2>(tile_shape_g1)));
                auto sfq_thr = sfq_copy.get_thread_slice(tid);
                Tensor tCsSFQ = sfq_thr.partition_S(sSFQ);
                Tensor tCrSFQ_v = sfq_thr.retile_D(tCrSFQ);
                for (int k = 0; k < size<2>(tCrSFQ_v); ++k) copy(sfq_copy, tCsSFQ(_,_,k), tCrSFQ_v(_,_,k));

                // ====== GEMM-1: S = Q @ K^T ======
                Tensor acc_s = partition_fragment_C(tiled_mma_g1, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
                clear(acc_s);
                for (int k = 0; k < size<2>(tCrQ); ++k) {
                    cute::gemm(tiled_mma_g1,
                        make_zip_tensor(tCrQ(_,_,k), tCrSFQ(_,_,k)),
                        make_zip_tensor(tCrK(_,_,k), tCrSFK(_,_,k)),
                        acc_s);
                }

                // ====== Causal mask ======
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

                // ====== Mask padded rows (rows >= m_start + kBlockM) to -inf ======
                // GEMM-1 computed S[128×128] but only first kBlockM=64 rows are valid
                {
                    Tensor acc_mask_rc = make_tensor(acc_s.data(),
                        flash::convert_layout_acc_rowcol(acc_s.layout()));
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(acc_mask_rc); ++mi) {
                        int const row = m_start + warp_idx * 16 + (mi / 2) * 16 + lane_row + (mi % 2) * 8;
                        if (row >= m_start + kBlockM || row >= p.seqlen_q) {
                            #pragma unroll
                            for (int ni = 0; ni < size<1>(acc_mask_rc); ++ni) {
                                acc_mask_rc(mi, ni) = -INFINITY;
                            }
                        }
                    }
                }

                // ====== Softmax fwd: P = exp2(S * scale_log2 - LSE_log2) ======
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
                        // For padded rows (>= kBlockM from start), use large LSE so P→0
                        bool valid = (row < m_start + kBlockM) && (row < p.seqlen_q);
                        lse_log2_regs[mi] = valid ? p.ptr_LSE_log2[idx] : INFINITY;
                        dpsum_regs[mi] = valid ? p.ptr_dPsum[idx] : 0.0f;
                    }
                }

                // Compute P and write as BF16 to smem_pds
                #pragma unroll
                for (int mi = 0; mi < nrow_s; ++mi) {
                    #pragma unroll
                    for (int ni = 0; ni < ncol_s; ++ni) {
                        float sv = acc_s_rc(mi, ni);
                        float pv = (sv == -INFINITY) ? 0.0f : exp2f(sv * p.softmax_scale_log2 - lse_log2_regs[mi]);
                        acc_s_rc(mi, ni) = pv;  // P in registers for softmax bwd
                    }
                }

                // Scatter P (BF16) to smem_pds — only kBlockM=64 valid rows
                {
                    int const wm = warp_idx;
                    auto* pds = s.smem_pds.data();
                    #pragma unroll
                    for (int mi = 0; mi < nrow_s; ++mi) {
                        int const row = wm * 16 + (mi / 2) * 16 + lane_row + (mi % 2) * 8;
                        if (row >= kBlockM) continue;  // Skip padded rows
                        #pragma unroll
                        for (int ni = 0; ni < ncol_s; ni += 2) {
                            int const col = (ni / 2) * 8 + lane_col * 2;
                            ElementBf16 b0 = static_cast<ElementBf16>(acc_s_rc(mi, ni));
                            ElementBf16 b1 = static_cast<ElementBf16>(acc_s_rc(mi, ni + 1));
                            int bo = (row * kBlockN + col) * 2;
                            int sw = bo ^ (((bo >> 7) & 7) << 4);
                            *reinterpret_cast<uint32_t*>(pds + sw) =
                                uint32_t(reinterpret_cast<uint16_t const&>(b0)) |
                                (uint32_t(reinterpret_cast<uint16_t const&>(b1)) << 16);
                        }
                    }
                }
                __syncthreads();

                // Transpose P → P^T into smem_pt
                transpose_bf16_smem(s.smem_pds.data(), s.smem_pt.data(),
                    kBlockM, kBlockN, tid, NumMmaThreads);

                // ====== GEMM-3: dV[128×D] += P^T[128×64] @ dO[64×D] ======
                // B-operand dO needs TN format: [N=D × K=64] = dO^T
                // Transpose dO [64×D BF16] → dO^T [D×64 BF16] into smem_pds (16 KB scratch)
                transpose_bf16_smem(s.smem_do.data(), s.smem_pds.data(),
                    kBlockM, kHeadDim, tid, NumMmaThreads);
                __syncthreads();

                {
                    TiledMma_G3 tiled_mma_g3;
                    auto thr_g3 = tiled_mma_g3.get_thread_slice(tid);

                    // A-operand: P^T [kBlockN × kBlockM BF16 swizzled] in smem_pt
                    Tensor sPt = make_tensor(make_smem_ptr(reinterpret_cast<ElementBf16*>(s.smem_pt.data())), SmemLayoutPt{});
                    Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
                    auto sPt_flat = make_tensor(make_smem_ptr(reinterpret_cast<ElementBf16*>(s.smem_pt.data())),
                        make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{})));
                    Tensor tCrPt = thr_g3.partition_fragment_A(sPt_flat);
                    auto cpA3 = make_tiled_copy_A(SmemCopyAtomA_BF16{}, tiled_mma_g3);
                    auto thrA3 = cpA3.get_thread_slice(tid);
                    Tensor tCsPt = thrA3.partition_S(sPt_pi);
                    Tensor tCrPt_v = thrA3.retile_D(tCrPt);

                    // B-operand: dO^T [kHeadDim × kBlockM BF16 swizzled] in smem_pds
                    Tensor sdOt = make_tensor(make_smem_ptr(reinterpret_cast<ElementBf16*>(s.smem_pds.data())), SmemLayoutdOt{});
                    Tensor sdOt_pi = cute::as_position_independent_swizzle_tensor(sdOt);
                    auto sdOt_flat = make_tensor(make_smem_ptr(reinterpret_cast<ElementBf16*>(s.smem_pds.data())),
                        make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));
                    Tensor tCrdOt = thr_g3.partition_fragment_B(sdOt_flat);
                    auto cpB3 = make_tiled_copy_B(SmemCopyAtomB_BF16{}, tiled_mma_g3);
                    auto thrB3 = cpB3.get_thread_slice(tid);
                    Tensor tCsdOt = thrB3.partition_S(sdOt_pi);
                    Tensor tCrdOt_v = thrB3.retile_D(tCrdOt);

                    for (int k = 0; k < size<2>(tCrPt); ++k) {
                        copy(cpA3, tCsPt(_,_,k), tCrPt_v(_,_,k));
                        copy(cpB3, tCsdOt(_,_,k), tCrdOt_v(_,_,k));
                        cute::gemm(tiled_mma_g3, tCrPt(_,_,k), tCrdOt(_,_,k), tOrDV);
                    }
                }


                // ====== GEMM-2 (scalar): dP + dS computation ======
                // Compute dP[m,n] = sum_d(dO[m,d] * V[n,d]) and dS = P * (dP - dPsum) * scale
                // Using scalar GMEM reads. P is in acc_s_rc (8x1x1 layout, matching this loop).
                {
                    int const wm = warp_idx;
                    ElementBf16 const* dO_base = p.ptr_dO + bidb * p.do_batch_stride + bidh * p.do_head_stride;
                    Element const* V_base = p.ptr_V + bidb * p.v_batch_stride + bidh_kv * p.v_head_stride;

                    #pragma unroll
                    for (int mi = 0; mi < nrow_s; ++mi) {
                        int const local_row = wm * 16 + (mi / 2) * 16 + lane_row + (mi % 2) * 8;
                        int const row = m_start + local_row;
                        if (local_row >= kBlockM || row >= p.seqlen_q) continue;
                        #pragma unroll
                        for (int ni = 0; ni < ncol_s; ni += 2) {
                            int const col0 = n_start + (ni / 2) * 8 + lane_col * 2;
                            float dp0 = 0.0f, dp1 = 0.0f;
                            ElementBf16 const* dO_ptr = dO_base + row * p.do_row_stride;
                            for (int d = 0; d < kHeadDim; d += 4) {
                                float dO_0 = static_cast<float>(dO_ptr[d]);
                                float dO_1 = static_cast<float>(dO_ptr[d+1]);
                                float dO_2 = static_cast<float>(dO_ptr[d+2]);
                                float dO_3 = static_cast<float>(dO_ptr[d+3]);
                                if (col0 < p.seqlen_k) {
                                    Element const* V0 = V_base + col0 * p.v_row_stride;
                                    dp0 += dO_0 * float(V0[d]) + dO_1 * float(V0[d+1])
                                         + dO_2 * float(V0[d+2]) + dO_3 * float(V0[d+3]);
                                }
                                if (col0 + 1 < p.seqlen_k) {
                                    Element const* V1 = V_base + (col0 + 1) * p.v_row_stride;
                                    dp1 += dO_0 * float(V1[d]) + dO_1 * float(V1[d+1])
                                         + dO_2 * float(V1[d+2]) + dO_3 * float(V1[d+3]);
                                }
                            }
                            float p0 = acc_s_rc(mi, ni);
                            float p1 = acc_s_rc(mi, ni + 1);
                            float ds0 = p0 * (dp0 - dpsum_regs[mi]) * p.softmax_scale;
                            float ds1 = p1 * (dp1 - dpsum_regs[mi]) * p.softmax_scale;

                            int const pds_row = local_row;
                            int const pds_col = (ni / 2) * 8 + lane_col * 2;
                            ElementBf16 b0 = static_cast<ElementBf16>(ds0);
                            ElementBf16 b1 = static_cast<ElementBf16>(ds1);
                            int bo = (pds_row * kBlockN + pds_col) * 2;
                            int sw = bo ^ (((bo >> 7) & 7) << 4);
                            *reinterpret_cast<uint32_t*>(s.smem_pds.data() + sw) =
                                uint32_t(reinterpret_cast<uint16_t const&>(b0)) |
                                (uint32_t(reinterpret_cast<uint16_t const&>(b1)) << 16);
                        }
                    }
                }

                // Transpose dS → dS^T
                __syncthreads();
                transpose_bf16_smem(s.smem_pds.data(), s.smem_pt.data(),
                    kBlockM, kBlockN, tid, NumMmaThreads);

                // Convert Q FP8 [64×128 swizzled] → Q^T BF16 [128×64 swizzled] in smem_pds
                // Combined convert + transpose: read FP8 Q, convert to BF16, write transposed
                {
                    int total = kBlockM * kHeadDim;  // = 64 * 128 = 8192 elements
                    for (int i = tid; i < total; i += NumMmaThreads) {
                        int row = i / kHeadDim;  // Q row [0, 64)
                        int col = i % kHeadDim;  // Q col [0, 128)
                        // Read 1 FP8 from swizzled smem_q
                        int src_off = row * kHeadDim + col;
                        int src_sw = src_off ^ (((src_off >> 7) & 7) << 4);
                        uint8_t fp8_byte = s.smem_q.data()[src_sw];
                        cutlass::bfloat16_t bf16_val = static_cast<cutlass::bfloat16_t>(
                            static_cast<float>(cutlass::float_e4m3_t::bitcast(fp8_byte)));
                        // Write to transposed position in smem_pds: Q^T[col, row]
                        int dst_off = (col * kBlockM + row) * 2;  // BF16 bytes
                        int dst_sw = dst_off ^ (((dst_off >> 7) & 7) << 4);
                        *reinterpret_cast<uint16_t*>(s.smem_pds.data() + dst_sw) =
                            reinterpret_cast<uint16_t const&>(bf16_val);
                    }
                }
                __syncthreads();

                // ====== GEMM-4: dK[128×D] += dS^T[128×64] @ Q^T[D×64] ======
                // Wait — dK[128×D] = dS^T[128×64] @ Q[64×D]
                // A = dS^T: M=128, K=64 → [128×64] in smem_pt ✓
                // B = Q: N=D=128, K=64 → B in TN format: [N×K] = [128×64] = Q^T
                // Q^T is in smem_pds as [kHeadDim × kBlockM] = [128×64] ✓
                {
                    TiledMma_G4 tiled_mma_g4;
                    auto thr_g4 = tiled_mma_g4.get_thread_slice(tid);

                    // A-operand: dS^T [kBlockN × kBlockM BF16 swizzled] in smem_pt
                    Tensor sdSt = make_tensor(make_smem_ptr(reinterpret_cast<ElementBf16*>(s.smem_pt.data())), SmemLayoutPt{});
                    Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);
                    auto sdSt_flat = make_tensor(make_smem_ptr(reinterpret_cast<ElementBf16*>(s.smem_pt.data())),
                        make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{})));
                    Tensor tCrdSt = thr_g4.partition_fragment_A(sdSt_flat);
                    auto cpA4 = make_tiled_copy_A(SmemCopyAtomA_BF16{}, tiled_mma_g4);
                    auto thrA4 = cpA4.get_thread_slice(tid);
                    Tensor tCsdSt = thrA4.partition_S(sdSt_pi);
                    Tensor tCrdSt_v = thrA4.retile_D(tCrdSt);

                    // B-operand: Q^T [kHeadDim × kBlockM BF16 swizzled] in smem_pds
                    Tensor sQt = make_tensor(make_smem_ptr(reinterpret_cast<ElementBf16*>(s.smem_pds.data())), SmemLayoutQt{});
                    Tensor sQt_pi = cute::as_position_independent_swizzle_tensor(sQt);
                    auto sQt_flat = make_tensor(make_smem_ptr(reinterpret_cast<ElementBf16*>(s.smem_pds.data())),
                        make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));
                    Tensor tCrQt = thr_g4.partition_fragment_B(sQt_flat);
                    auto cpB4 = make_tiled_copy_B(SmemCopyAtomB_BF16{}, tiled_mma_g4);
                    auto thrB4 = cpB4.get_thread_slice(tid);
                    Tensor tCsQt = thrB4.partition_S(sQt_pi);
                    Tensor tCrQt_v = thrB4.retile_D(tCrQt);

                    for (int k = 0; k < size<2>(tCrdSt); ++k) {
                        copy(cpA4, tCsdSt(_,_,k), tCrdSt_v(_,_,k));
                        copy(cpB4, tCsQt(_,_,k), tCrQt_v(_,_,k));
                        cute::gemm(tiled_mma_g4, tCrdSt(_,_,k), tCrQt(_,_,k), tOrDK);
                    }
                }

                __syncthreads();
            }  // GQA head loop
        }  // M-block loop
    }
};

}  // namespace flash
