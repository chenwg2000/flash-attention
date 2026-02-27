/******************************************************************************
 * SM120 MXFP8 Block-Scaled Flash Attention Forward Mainloop
 * With TMA loads for K/V and pipelined N-block loop
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
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"  // SM90::GMMA::Layout_K_SW128_Atom

#include "cutlass/pipeline/pipeline.hpp"
#include "sm90_pipeline_no_cluster.hpp"

#include "utils.h"
#include "softmax.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Scale factor partitioning helpers - adapted from
// cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
sm120_thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    using AtomShape_MNK = typename Atom::Shape_MNK;
    using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;
    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfatensor, t_tile);
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);
    auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);
    auto thr_tile = make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    return zipped_divide(tv_tensor, thr_tile);
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
sm120_thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    using AtomShape_MNK = typename Atom::Shape_MNK;
    using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;
    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfbtensor, t_tile);
    auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);
    auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, _);
    auto thr_tile = make_tile(_, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    return zipped_divide(tv_tensor, thr_tile);
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
sm120_partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
        sm120_thrfrg_SFA(sfatensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition);
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
sm120_partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
        sm120_thrfrg_SFB(sfbtensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition);
}

template <class TiledMma_>
CUTE_HOST_DEVICE constexpr auto
sm120_get_layoutSFA_TV(TiledMma_& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto atile = make_tile(_,
        make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                              make_stride(Int<1>{}, Int<0>{})), _));
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    return sm120_thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

template <class TiledMma_>
CUTE_HOST_DEVICE constexpr auto
sm120_get_layoutSFB_TV(TiledMma_& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto btile = make_tile(_,
        make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                              make_stride(Int<0>{}, Int<1>{})), _));
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    return sm120_thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kStages_, class TileShape_MNK_, class Element_, class ElementAccum_,
          bool Is_causal_, bool Is_local_, bool Has_softcap_>
struct CollectiveMainloopFwdSm120 {

    static constexpr int kStages = kStages_;
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = cutlass::arch::Sm90;

    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    using ElementSF = cutlass::float_ue8m0_t;
    static constexpr int kSFVecSize = 32;
    static constexpr int kSFCols = kHeadDim / kSFVecSize;
    // SF cols for GEMM-II K dimension (kBlockN)
    static constexpr int kSFColsPV = kBlockN / kSFVecSize;

    // Block-scaled MMA
    using MmaAtomOp = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
        cutlass::float_ue8m0_t, kSFVecSize>;
    // 8 warps all along M, 1 along N — ensures quad_allreduce covers all N-columns
    // for correct softmax row reduction
    using AtomLayoutMNK = Layout<Shape<_8, _1, _1>>;
    using PermTileM = decltype(cute::min(Int<kBlockM>{}, _128{}));
    using PermTileN = _8;
    using PermTileK = _32;
    using TiledMma = decltype(cute::make_tiled_mma(
        MMA_Atom<MmaAtomOp>{}, AtomLayoutMNK{},
        Tile<PermTileM, PermTileN, PermTileK>{}));

    static constexpr int NumMmaThreads = size(TiledMma{});

    // Swizzled layouts for GEMM-I operands (K and Q) — SW128 for bank-conflict-free LDSM
    using SmemLayoutAtomSW = SM90::GMMA::Layout_K_SW128_Atom<uint8_t>;  // Swizzle<3,4,3>
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomSW{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomSW{}, Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));

    // Non-swizzled V layout (V needs raw-pointer reads for byte-level transpose)
    using SmemLayoutV = Layout<Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>,
                               Stride<Int<kHeadDim>, _1, Int<kBlockN * kHeadDim>>>;

    // Swizzled P layout for GEMM-II A-operand [kBlockM, kBlockN]
    using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomSW{}, Shape<Int<kBlockM>, Int<kBlockN>>{}));

    // Swizzled transposed V layout: [kHeadDim, kBlockN] for GEMM-II B-operand
    using SmemLayoutVtSingleStage = decltype(tile_to_shape(SmemLayoutAtomSW{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    // ====== TMA types ======
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
    using ShapeKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, hdim, head, batch)
    using StrideKV = cute::Stride<int64_t, _1, int64_t, int64_t>;

    // 2D SMEM layouts for one pipeline stage (K swizzled, V non-swizzled)
    using SmemLayoutKSingleStage = decltype(take<0, 2>(SmemLayoutK{}));  // Swizzled 2D
    using SmemLayoutVSingleStage = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>,
                                          Stride<Int<kHeadDim>, _1>>;   // Non-swizzled 2D

    using TMA_K = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeKV{}, StrideKV{}),
        SmemLayoutKSingleStage{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
        _1{}));
    using TMA_V = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeKV{}, StrideKV{}),
        SmemLayoutVSingleStage{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
        _1{}));

    static constexpr uint32_t TmaTransactionBytesK =
        static_cast<uint32_t>(size(SmemLayoutKSingleStage{}) * sizeof(Element));
    static constexpr uint32_t TmaTransactionBytesV =
        static_cast<uint32_t>(size(SmemLayoutVSingleStage{}) * sizeof(Element));
    static constexpr uint32_t TmaTransactionBytesKV = TmaTransactionBytesK + TmaTransactionBytesV;

    // ====== Pipeline types ======
    using MainloopPipeline = cutlass::PipelineTmaAsyncNoCluster<kStages>;
    using PipelineState = cutlass::PipelineState<kStages>;

    // SMEM layouts for scale factors (CUTLASS block-scaled format)
    static constexpr int MMA_NSF = 32 / kSFVecSize;
    using Sm1xxCfg = cutlass::detail::Sm1xxBlockScaledConfig<kSFVecSize>;
    using Blk_MN = typename Sm1xxCfg::Blk_MN;
    using Blk_SF = typename Sm1xxCfg::Blk_SF;
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

    using mnBBS  = Shape<_32, _4>;
    using mnBBSt = Stride<_16, _4>;
    using kBBS  = Shape<Int<kSFVecSize>, Int<MMA_NSF>>;
    using kBBSt = Stride<_0, _1>;

    // SFQ: [kBlockM, kHeadDim/32] - scales for Q along headDim
    using sSFQ_sM = decltype(prepend(Int<kBlockM>{} / Blk_MN{}, mnBBS{}));
    using sSF_sMN = decltype(prepend(Blk_Elems{}, mnBBSt{}));
    using sSF_sK  = decltype(prepend(make_shape(Blk_SF{}/Int<MMA_NSF>{}, Int<kHeadDim>{}/Int<kSFVecSize>{}/Blk_SF{}), kBBS{}));
    using sSFQ_sK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kBlockM>{}/Blk_MN{}*Blk_Elems{}), kBBSt{}));
    using SmemLayoutAtomSFQ = decltype(make_layout(make_shape(sSFQ_sM{}, sSF_sK{}), make_stride(sSF_sMN{}, sSFQ_sK{})));

    // SFK: [kBlockN, kHeadDim/32] - scales for K along headDim
    using sSFK_sN = decltype(prepend(Int<kBlockN>{} / Blk_MN{}, mnBBS{}));
    using sSFK_sK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kBlockN>{}/Blk_MN{}*Blk_Elems{}), kBBSt{}));
    using SmemLayoutAtomSFK = decltype(make_layout(make_shape(sSFK_sN{}, sSF_sK{}), make_stride(sSF_sMN{}, sSFK_sK{})));

    // SFP: [kBlockM, kBlockN/32] - identity scales for P (GEMM-II A-operand)
    using sSF_sK_pv = decltype(prepend(make_shape(Blk_SF{}/Int<MMA_NSF>{}, Int<kBlockN>{}/Int<kSFVecSize>{}/Blk_SF{}), kBBS{}));
    using sSFP_sK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kBlockM>{}/Blk_MN{}*Blk_Elems{}), kBBSt{}));
    using SmemLayoutAtomSFP = decltype(make_layout(make_shape(sSFQ_sM{}, sSF_sK_pv{}), make_stride(sSF_sMN{}, sSFP_sK{})));

    // SFV for GEMM-II: [kHeadDim, kBlockN/32] - scales for V^T along kBlockN
    using sSFVt_sN = decltype(prepend(Int<kHeadDim>{} / Blk_MN{}, mnBBS{}));
    using sSFVt_sK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kHeadDim>{}/Blk_MN{}*Blk_Elems{}), kBBSt{}));
    using SmemLayoutAtomSFVt = decltype(make_layout(make_shape(sSFVt_sN{}, sSF_sK_pv{}), make_stride(sSF_sMN{}, sSFVt_sK{})));

    using SmemLayoutSFQ = decltype(make_layout(
        append(shape(SmemLayoutAtomSFQ{}), Int<1>{}),
        append(stride(SmemLayoutAtomSFQ{}), size(filter_zeros(SmemLayoutAtomSFQ{})))));
    using SmemLayoutSFK = decltype(make_layout(
        append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFK{}), size(filter_zeros(SmemLayoutAtomSFK{})))));
    using SmemLayoutSFV = SmemLayoutSFK;

    // GEMM-I copy atoms: LDSM for swizzled Q and K operands
    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;   // Q (A-operand)
    using SmemCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, uint8_t>;   // K (B-operand, U32x2 for 8x1x1 warp layout)

    // GEMM-II copy atoms: LDSM for swizzled P and V^T operands
    using SmemCopyAtomP  = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;   // P (A-operand)
    using SmemCopyAtomVt = Copy_Atom<SM75_U32x2_LDSM_N, uint8_t>;   // V^T (B-operand)

    // Scale factors: always UniversalCopy (non-swizzled block-scaled format)
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<uint8_t>, uint8_t>;

    // Shared memory layout — aggressive overlapping to fit within 101 KB limit
    struct TensorStorage : cute::aligned_struct<128> {
        union {
            alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutQ>> smem_q;
            alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutP>> smem_p;
        };
        union {
            alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutK>> smem_k;
            alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutVtSingleStage>> smem_vt;
        };
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutSFQ>> smem_sfq;
        union {
            struct {
                cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutSFK>> smem_sfk;
                cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutSFV>> smem_sfv;
            };
            struct {
                cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutAtomSFP>> smem_sfp;
                cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutAtomSFVt>> smem_sfvt;
            };
        };
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
        uint8_t const* ptr_SFV; index_t sfv_batch_stride, sfv_head_stride, sfv_row_stride;
        float softmax_scale, softcap_val;
        int window_size_left, window_size_right, seqlen_q, seqlen_k;
        int num_heads_kv, batch_size;
    };

    struct Params {
        Element const* ptr_Q;
        int64_t q_batch_stride, q_row_stride, q_head_stride;
        ShapeKV shape_KV;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        uint8_t const* ptr_SFQ; index_t sfq_batch_stride, sfq_head_stride, sfq_row_stride;
        uint8_t const* ptr_SFK; index_t sfk_batch_stride, sfk_head_stride, sfk_row_stride;
        uint8_t const* ptr_SFV; index_t sfv_batch_stride, sfv_head_stride, sfv_row_stride;
        float softmax_scale, softmax_scale_log2, softcap_val;
        int window_size_left, window_size_right, seqlen_q, seqlen_k;
    };

    static Params to_underlying_arguments(Arguments const& a) {
        auto shape_KV = make_shape(int32_t(a.seqlen_k), int32_t(kHeadDim),
                                   int32_t(a.num_heads_kv), int32_t(a.batch_size));
        auto stride_K = make_stride(int64_t(a.k_row_stride), _1{},
                                    int64_t(a.k_head_stride), int64_t(a.k_batch_stride));
        auto stride_V = make_stride(int64_t(a.v_row_stride), _1{},
                                    int64_t(a.v_head_stride), int64_t(a.v_batch_stride));

        Tensor mK = make_tensor(make_gmem_ptr(a.ptr_K), shape_KV, stride_K);
        TMA_K tma_load_K = make_tma_copy(GmemTiledCopyKV{}, mK, SmemLayoutKSingleStage{},
                                          make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), _1{});

        Tensor mV = make_tensor(make_gmem_ptr(a.ptr_V), shape_KV, stride_V);
        TMA_V tma_load_V = make_tma_copy(GmemTiledCopyKV{}, mV, SmemLayoutVSingleStage{},
                                          make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), _1{});

        return {a.ptr_Q, a.q_batch_stride, a.q_row_stride, a.q_head_stride,
                shape_KV, tma_load_K, tma_load_V,
                a.ptr_SFQ, a.sfq_batch_stride, a.sfq_head_stride, a.sfq_row_stride,
                a.ptr_SFK, a.sfk_batch_stride, a.sfk_head_stride, a.sfk_row_stride,
                a.ptr_SFV, a.sfv_batch_stride, a.sfv_head_stride, a.sfv_row_stride,
                a.softmax_scale, float(a.softmax_scale * M_LOG2E), a.softcap_val,
                a.window_size_left, a.window_size_right, a.seqlen_q, a.seqlen_k};
    }

    CUTLASS_DEVICE static void load_sf(uint8_t* dst, uint8_t const* src,
        int rows, int cols, index_t stride, int row_off, int tid, int nthreads) {
        if (src == nullptr) {
            for (int i = tid; i < rows*cols; i += nthreads) { dst[i] = 127; }
            return;
        }
        for (int i = tid; i < rows*cols; i += nthreads) {
            dst[i] = src[(row_off + i/cols)*stride + i%cols];
        }
    }

    CUTLASS_DEVICE static void load_tile(uint8_t* dst, uint8_t const* src,
        int rows, int cols, int64_t stride, int row_off, int tid, int nthreads) {
        int total = rows * (cols / 16);
        for (int i = tid; i < total; i += nthreads) {
            int r = i / (cols/16), c = i % (cols/16);
            *reinterpret_cast<uint4*>(dst + r*cols + c*16) =
                *reinterpret_cast<uint4 const*>(src + (row_off+r)*stride + c*16);
        }
    }

    /// Load tile from GMEM to SMEM with SW128 swizzle applied (Swizzle<3,4,3>)
    CUTLASS_DEVICE static void load_tile_swizzled(uint8_t* dst, uint8_t const* src,
        int rows, int cols, int64_t stride, int row_off, int tid, int nthreads) {
        int total = rows * (cols / 16);
        for (int i = tid; i < total; i += nthreads) {
            int r = i / (cols/16), c16 = i % (cols/16);
            int byte_offset = r * cols + c16 * 16;
            // Swizzle<3,4,3>: XOR bits [4,7) with bits [7,10) shifted right by 3
            int swizzled_offset = byte_offset ^ (((byte_offset >> 7) & 7) << 4);
            *reinterpret_cast<uint4*>(dst + swizzled_offset) =
                *reinterpret_cast<uint4 const*>(src + (row_off+r)*stride + c16*16);
        }
    }

    CUTLASS_DEVICE static void fill_identity_sf(uint8_t* dst, int count, int tid, int nthreads) {
        // Vectorized fill: 16 bytes at a time via uint4
        constexpr uint32_t fill_word = 0x7F7F7F7Fu;  // 127 repeated
        uint4 fill_vec;
        fill_vec.x = fill_word; fill_vec.y = fill_word;
        fill_vec.z = fill_word; fill_vec.w = fill_word;
        int total16 = count / 16;
        for (int i = tid; i < total16; i += nthreads) {
            reinterpret_cast<uint4*>(dst)[i] = fill_vec;
        }
        // Handle remainder
        int done = total16 * 16;
        for (int i = done + tid; i < count; i += nthreads) {
            dst[i] = 127;
        }
    }

    /// Vectorized 4x4 block transpose with SW128 swizzle on destination
    CUTLASS_DEVICE static void transpose_v_smem(
        uint8_t const* __restrict__ src, uint8_t* __restrict__ dst,
        int tid, int nthreads)
    {
        constexpr int rows = kBlockN;   // 128
        constexpr int cols = kHeadDim;  // 128
        constexpr int blocks_r = rows / 4;
        constexpr int blocks_c = cols / 4;
        constexpr int total_blocks = blocks_r * blocks_c;

        for (int blk = tid; blk < total_blocks; blk += nthreads) {
            int br = blk / blocks_c;
            int bc = blk % blocks_c;
            int r = br * 4;
            int c = bc * 4;

            // Read 4 rows of 4 bytes each (non-swizzled V source)
            uint32_t r0 = *reinterpret_cast<uint32_t const*>(src + (r+0)*cols + c);
            uint32_t r1 = *reinterpret_cast<uint32_t const*>(src + (r+1)*cols + c);
            uint32_t r2 = *reinterpret_cast<uint32_t const*>(src + (r+2)*cols + c);
            uint32_t r3 = *reinterpret_cast<uint32_t const*>(src + (r+3)*cols + c);

            // 4x4 byte transpose via __byte_perm
            uint32_t t0 = __byte_perm(r0, r1, 0x5140);
            uint32_t t1 = __byte_perm(r0, r1, 0x7362);
            uint32_t t2 = __byte_perm(r2, r3, 0x5140);
            uint32_t t3 = __byte_perm(r2, r3, 0x7362);

            // Write 4 cols of 4 bytes each (transposed) with SW128 swizzle on dst
            // Swizzle<3,4,3>: addr ^ (((addr >> 7) & 7) << 4)
            // r is 4-byte aligned, so all 4 bytes in each write are in the same 16-byte block
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int off = (c + i) * rows + r;
                int sw  = off ^ (((off >> 7) & 7) << 4);
                uint32_t val = (i == 0) ? __byte_perm(t0, t2, 0x5410) :
                               (i == 1) ? __byte_perm(t0, t2, 0x7632) :
                               (i == 2) ? __byte_perm(t1, t3, 0x5410) :
                                          __byte_perm(t1, t3, 0x7632);
                *reinterpret_cast<uint32_t*>(dst + sw) = val;
            }
        }
    }

    template <typename FrgTensorO>
    CUTLASS_DEVICE void mha_fwd(
        Params const& p, TensorStorage& s,
        MainloopPipeline& pipeline, PipelineState& smem_pipe_read, PipelineState& smem_pipe_write,
        FrgTensorO& tOrO, float* lse_arr,
        int m_block, int bidb, int bidh, int bidh_kv, int tid)
    {
        int const block_n = kBlockN;
        int const m_start = m_block * kBlockM;
        int n_block_max = cute::ceil_div(p.seqlen_k, block_n);
        // Causal: skip N-blocks that are entirely above the diagonal
        if constexpr (Is_causal) {
            n_block_max = cute::min(n_block_max, cute::ceil_div(m_start + kBlockM, block_n));
        }

        TiledMma tiled_mma;
        auto thread_mma = tiled_mma.get_thread_slice(tid);
        auto tile_shape_mnk = tile_shape(tiled_mma);

        Tensor acc_ref = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
        static constexpr int kAccRows = decltype(size<0>(flash::convert_layout_acc_rowcol(acc_ref.layout())))::value;
        flash::Softmax<kAccRows, 8> softmax(p.softmax_scale_log2);
        clear(tOrO);

        // ====== Set up TMA tensor views ======
        Tensor mK_tma = p.tma_load_K.get_tma_tensor(p.shape_KV);
        Tensor mV_tma = p.tma_load_V.get_tma_tensor(p.shape_KV);

        // Select head and batch, then tile along seqlen
        Tensor gK = local_tile(mK_tma(_, _, bidh_kv, bidb),
            make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
        Tensor gV = local_tile(mV_tma(_, _, bidh_kv, bidb),
            make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}));

        // TMA partition: GMEM source
        auto block_tma_K = p.tma_load_K.get_slice(_0{});
        auto block_tma_V = p.tma_load_V.get_slice(_0{});
        Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK));  // (TMA, num_n_blocks)
        Tensor tVgV = group_modes<0, 3>(block_tma_V.partition_S(gV));  // (TMA, num_n_blocks)

        // TMA partition: SMEM destination (3D with pipeline stages; K swizzled, V non-swizzled)
        Tensor sK_tma = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(s.smem_k.data())), SmemLayoutK{});
        Tensor sV_tma = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(s.smem_v.data())), SmemLayoutV{});
        Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK_tma));  // (TMA, kStages)
        Tensor tVsV = group_modes<0, 3>(block_tma_V.partition_D(sV_tma));  // (TMA, kStages)

        // ====== Load Q + SFQ to SMEM (once per M-block, cooperative, swizzled) ======
        load_tile_swizzled(s.smem_q.data(), reinterpret_cast<uint8_t const*>(p.ptr_Q)
            + bidb*p.q_batch_stride + bidh*p.q_head_stride,
            kBlockM, kHeadDim, p.q_row_stride, m_start, tid, NumMmaThreads);
        load_sf(s.smem_sfq.data(), p.ptr_SFQ
            + bidb*p.sfq_batch_stride + bidh*p.sfq_head_stride,
            kBlockM, kSFCols, p.sfq_row_stride, m_start, tid, NumMmaThreads);
        fill_identity_sf(s.smem_sfp.data(), cute::cosize_v<SmemLayoutAtomSFP>, tid, NumMmaThreads);
        __syncthreads();

        // ====== Partition Q + SFQ from SMEM (swizzled + LDSM) ======
        Tensor sQ = make_tensor(make_smem_ptr(s.smem_q.data()), SmemLayoutQ{});
        Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
        // Use flat layout for fragment creation (avoids composed-layout issues in MMA partition)
        auto sQ_flat = make_tensor(make_smem_ptr(s.smem_q.data()),
            make_layout(make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
        Tensor tCrQ = thread_mma.partition_fragment_A(sQ_flat);
        auto copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto thr_copy_A = copy_A.get_thread_slice(tid);
        Tensor tCsQ = thr_copy_A.partition_S(sQ_pi);
        Tensor tCrQ_v = thr_copy_A.retile_D(tCrQ);
        for (int k = 0; k < size<2>(tCrQ_v); ++k) copy(copy_A, tCsQ(_,_,k), tCrQ_v(_,_,k));

        Tensor sSFQ = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(s.smem_sfq.data())),
            SmemLayoutAtomSFQ{});
        Tensor tCrSFQ = sm120_partition_fragment_SFA(sSFQ, thread_mma);
        auto sfq_copy = make_tiled_copy_impl(SmemCopyAtomSF{},
            sm120_get_layoutSFA_TV(tiled_mma),
            make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto sfq_thr = sfq_copy.get_thread_slice(tid);
        Tensor tCsSFQ = sfq_thr.partition_S((sSFQ));
        Tensor tCrSFQ_v = sfq_thr.retile_D(tCrSFQ);
        for (int k = 0; k < size<2>(tCrSFQ_v); ++k) copy(sfq_copy, tCsSFQ(_,_,k), tCrSFQ_v(_,_,k));

        // ====== Pre-build GEMM-I K partition (3D swizzled, indexed by stage inside loop) ======
        Tensor sK_3d = make_tensor(make_smem_ptr(s.smem_k.data()), SmemLayoutK{});
        Tensor sK_pi = cute::as_position_independent_swizzle_tensor(sK_3d);
        auto copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto thr_copy_B = copy_B.get_thread_slice(tid);
        Tensor tCsK_all = thr_copy_B.partition_S(sK_pi);  // (CPY, CPY_N, CPY_K, kStages)
        // Fragment from flat layout (shape only, not swizzled)
        auto sK_flat = make_tensor(make_smem_ptr(s.smem_k.data()),
            make_layout(make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
        Tensor tCrK = thread_mma.partition_fragment_B(sK_flat);
        Tensor tCrK_v = thr_copy_B.retile_D(tCrK);

        // ====== TMA Prologue: prefetch first kStages blocks ======
        int const num_stages = kStages_;  // Local var to avoid ODR-use of static constexpr
        int const prefetch_count = cute::min(num_stages, n_block_max);
        for (int pf = 0; pf < prefetch_count; ++pf) {
            int const n_blk_pf = n_block_max - 1 - pf;
            pipeline.producer_acquire(smem_pipe_write);
            if (tid == 0) {
                copy(p.tma_load_K.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                     tKgK(_, n_blk_pf), tKsK(_, smem_pipe_write.index()));
                copy(p.tma_load_V.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                     tVgV(_, n_blk_pf), tVsV(_, smem_pipe_write.index()));
            }
            ++smem_pipe_write;
        }

        // ====== N-block loop (pipelined) ======
        for (int n_blk = n_block_max-1; n_blk >= 0; --n_blk) {
            int const n_start = n_blk * block_n;
            int const stg = smem_pipe_read.index();

            // ------ Load SFK cooperatively (overlaps with TMA barrier wait) ------
            load_sf(s.smem_sfk.data() + stg*cute::cosize_v<SmemLayoutAtomSFK>,
                p.ptr_SFK + bidb*p.sfk_batch_stride + bidh_kv*p.sfk_head_stride,
                kBlockN, kSFCols, p.sfk_row_stride, n_start, tid, NumMmaThreads);

            // ------ Wait for K+V TMA + SFK load visibility ------
            pipeline.consumer_wait(smem_pipe_read);
            __syncthreads();

            // ====== GEMM-I: S = Q @ K^T (swizzled K + LDSM) ======
            Tensor tCsK = tCsK_all(_,_,_,stg);  // Select current pipeline stage

            Tensor sSFK = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(
                s.smem_sfk.data() + stg*cute::cosize_v<SmemLayoutAtomSFK>)), SmemLayoutAtomSFK{});
            Tensor tCrSFK = sm120_partition_fragment_SFB(sSFK, thread_mma);
            auto sfk_copy = make_tiled_copy_impl(SmemCopyAtomSF{},
                sm120_get_layoutSFB_TV(tiled_mma),
                make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
            auto sfk_thr = sfk_copy.get_thread_slice(tid);
            Tensor tCsSFK = sfk_thr.partition_S((sSFK));
            Tensor tCrSFK_v = sfk_thr.retile_D(tCrSFK);

            Tensor acc_s = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
            clear(acc_s);

            auto K_MAX = size<2>(tCrQ);
            for (int k = 0; k < K_MAX; ++k) {
                copy(copy_B, tCsK(_,_,k), tCrK_v(_,_,k));
                copy(sfk_copy, tCsSFK(_,_,k), tCrSFK_v(_,_,k));
                cute::gemm(tiled_mma,
                    make_zip_tensor(tCrQ(_,_,k), tCrSFQ(_,_,k)),
                    make_zip_tensor(tCrK(_,_,k), tCrSFK(_,_,k)),
                    acc_s);
            }

            // ====== Softcap ======
            if constexpr (Has_softcap) {
                for (int i = 0; i < size(acc_s); ++i)
                    acc_s(i) = cutlass::fast_tanh(acc_s(i) * p.softcap_val);
            }

            // ====== Causal mask: set future positions to -inf ======
            if constexpr (Is_causal) {
                // Only the diagonal block (n_start >= m_start) has partially masked positions.
                // All blocks with n_start < m_start are fully below the diagonal.
                if (n_start >= m_start) {
                    Tensor acc_s_rc = make_tensor(acc_s.data(),
                        flash::convert_layout_acc_rowcol(acc_s.layout()));
                    int const warp_m = tid / 32;
                    int const lane_row = (tid % 32) / 4;
                    int const lane_col = (tid % 32) % 4;
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(acc_s_rc); ++mi) {
                        int const row = m_start + warp_m * 16
                            + (mi / 2) * 16 + lane_row + (mi % 2) * 8;
                        #pragma unroll
                        for (int ni = 0; ni < size<1>(acc_s_rc); ni += 2) {
                            int const col = n_start + (ni / 2) * 8 + lane_col * 2;
                            if (col > row) {
                                acc_s_rc(mi, ni)     = -INFINITY;
                                acc_s_rc(mi, ni + 1) = -INFINITY;
                            } else if (col + 1 > row) {
                                acc_s_rc(mi, ni + 1) = -INFINITY;
                            }
                        }
                    }
                }
            }

            // ====== Online softmax ======
            if (n_blk == n_block_max - 1) {
                softmax.template online_softmax<true>(acc_s);
            } else {
                auto sc = softmax.template max_get_scale<false>(acc_s);
                softmax.rescale_o(tOrO, sc);
                softmax.template online_softmax<false>(acc_s);
            }

            // ====== GEMM-II: O += P @ V (block-scaled MMA) ======

            // Step 1: Convert P from FP32 to FP8 and store to SMEM
            {
                Tensor tOrP = make_tensor_like<Element>(acc_s);
                flash::convert_type_out(acc_s, tOrP);

                int const warp_idx_l = tid / 32;
                int const lane_idx_l = tid % 32;
                int const warp_m_l = warp_idx_l;
                int const lane_row_l = lane_idx_l / 4;
                int const lane_col_l = lane_idx_l % 4;

                Tensor tOrP_rc = make_tensor(tOrP.data(),
                    flash::convert_layout_acc_rowcol(tOrP.layout()));

                auto* p_smem = s.smem_p.data();
                #pragma unroll
                for (int mi = 0; mi < size<0>(tOrP_rc); ++mi) {
                    int const row = warp_m_l * 16 + (mi / 2) * 16 + lane_row_l + (mi % 2) * 8;
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrP_rc); ni += 2) {
                        int const col = (ni / 2) * 8 + lane_col_l * 2;
                        int const off = row * kBlockN + col;
                        int const sw  = off ^ (((off >> 7) & 7) << 4);
                        uint8_t b0 = reinterpret_cast<uint8_t const&>(tOrP_rc(mi, ni));
                        uint8_t b1 = reinterpret_cast<uint8_t const&>(tOrP_rc(mi, ni + 1));
                        *reinterpret_cast<uint16_t*>(p_smem + sw) =
                            uint16_t(b0) | (uint16_t(b1) << 8);
                    }
                }
            }

            // Step 2: V transpose + identity SF fill (runs concurrently with P scatter above
            // since smem_p, smem_vt, smem_sfp, smem_sfvt are all in different/aliased-but-safe regions)
            transpose_v_smem(
                s.smem_v.data() + stg * kBlockN * kHeadDim,
                s.smem_vt.data(),
                tid, NumMmaThreads);
            fill_identity_sf(s.smem_sfp.data(), cute::cosize_v<SmemLayoutAtomSFP>, tid, NumMmaThreads);
            fill_identity_sf(s.smem_sfvt.data(), cute::cosize_v<SmemLayoutAtomSFVt>, tid, NumMmaThreads);
            __syncthreads();  // Ensure P, V^T, and SFs are all visible before GEMM-II

            // Step 3: Partition P from SMEM as GEMM-II A-operand (swizzled + LDSM)
            Tensor sP = make_tensor(make_smem_ptr(s.smem_p.data()), SmemLayoutP{});
            Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
            auto sP_flat = make_tensor(make_smem_ptr(s.smem_p.data()),
                make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
            Tensor tCrP = thread_mma.partition_fragment_A(sP_flat);
            auto copy_P_A = make_tiled_copy_A(SmemCopyAtomP{}, tiled_mma);
            auto thr_copy_P = copy_P_A.get_thread_slice(tid);
            Tensor tCsP = thr_copy_P.partition_S(sP_pi);
            Tensor tCrP_v = thr_copy_P.retile_D(tCrP);

            Tensor sSFP = make_tensor(make_smem_ptr(
                reinterpret_cast<ElementSF*>(s.smem_sfp.data())), SmemLayoutAtomSFP{});
            Tensor tCrSFP = sm120_partition_fragment_SFA(sSFP, thread_mma);
            auto sfp_copy = make_tiled_copy_impl(SmemCopyAtomSF{},
                sm120_get_layoutSFA_TV(tiled_mma),
                make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
            auto sfp_thr = sfp_copy.get_thread_slice(tid);
            Tensor tCsSFP = sfp_thr.partition_S(sSFP);
            Tensor tCrSFP_v = sfp_thr.retile_D(tCrSFP);

            // Step 4: Partition V^T from SMEM as GEMM-II B-operand (swizzled + LDSM)
            Tensor sVt = make_tensor(make_smem_ptr(s.smem_vt.data()), SmemLayoutVtSingleStage{});
            Tensor sVt_pi = cute::as_position_independent_swizzle_tensor(sVt);
            auto sVt_flat = make_tensor(make_smem_ptr(s.smem_vt.data()),
                make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));
            Tensor tCrVt = thread_mma.partition_fragment_B(sVt_flat);
            auto copy_Vt_B = make_tiled_copy_B(SmemCopyAtomVt{}, tiled_mma);
            auto thr_copy_Vt = copy_Vt_B.get_thread_slice(tid);
            Tensor tCsVt = thr_copy_Vt.partition_S(sVt_pi);
            Tensor tCrVt_v = thr_copy_Vt.retile_D(tCrVt);

            Tensor sSFVt = make_tensor(make_smem_ptr(
                reinterpret_cast<ElementSF*>(s.smem_sfvt.data())), SmemLayoutAtomSFVt{});
            Tensor tCrSFVt = sm120_partition_fragment_SFB(sSFVt, thread_mma);
            auto sfvt_copy = make_tiled_copy_impl(SmemCopyAtomSF{},
                sm120_get_layoutSFB_TV(tiled_mma),
                make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
            auto sfvt_thr = sfvt_copy.get_thread_slice(tid);
            Tensor tCsSFVt = sfvt_thr.partition_S(sSFVt);
            Tensor tCrSFVt_v = sfvt_thr.retile_D(tCrSFVt);

            // Step 5: Execute GEMM-II with block-scaled MMA
            auto K_MAX_PV = size<2>(tCrP);
            #pragma unroll
            for (int k = 0; k < K_MAX_PV; ++k) {
                copy(copy_P_A, tCsP(_, _, k), tCrP_v(_, _, k));
                copy(copy_Vt_B, tCsVt(_, _, k), tCrVt_v(_, _, k));
                copy(sfp_copy, tCsSFP(_, _, k), tCrSFP_v(_, _, k));
                copy(sfvt_copy, tCsSFVt(_, _, k), tCrSFVt_v(_, _, k));

                cute::gemm(tiled_mma,
                    make_zip_tensor(tCrP(_, _, k), tCrSFP(_, _, k)),
                    make_zip_tensor(tCrVt(_, _, k), tCrSFVt(_, _, k)),
                    tOrO);
            }

            // ------ Sync all warps before releasing stage ------
            __syncthreads();

            // ------ Release this pipeline stage ------
            pipeline.consumer_release(smem_pipe_read);
            ++smem_pipe_read;

            // ------ Issue TMA for next block (overlaps with next iteration) ------
            int const next_n_blk = n_blk - num_stages;
            if (next_n_blk >= 0) {
                pipeline.producer_acquire(smem_pipe_write);
                if (tid == 0) {
                    copy(p.tma_load_K.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                         tKgK(_, next_n_blk), tKsK(_, smem_pipe_write.index()));
                    copy(p.tma_load_V.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                         tVgV(_, next_n_blk), tVsV(_, smem_pipe_write.index()));
                }
                ++smem_pipe_write;
            }
        }

        // ====== Finalize softmax: normalize O and compute LSE ======
        auto scores_scale = softmax.finalize();
        softmax.rescale_o(tOrO, scores_scale);
        for (int mi = 0; mi < size(softmax.row_sum); ++mi) lse_arr[mi] = softmax.row_sum(mi);
    }
};

}  // namespace flash
