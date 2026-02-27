/******************************************************************************
 * SM120 MXFP8 Block-Scaled Flash Attention Forward Mainloop
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

    // SMEM layouts for data - simple row-major (no swizzle, for correctness first)
    using SmemLayoutQ = Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>,
                               Stride<Int<kHeadDim>, _1, Int<kBlockN * kHeadDim>>>;
    using SmemLayoutV = SmemLayoutK;

    // P layout for GEMM-II A-operand [kBlockM, kBlockN]
    using SmemLayoutP = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;

    // Transposed V layout: [kHeadDim, kBlockN] for GEMM-II B-operand
    using SmemLayoutVtSingleStage = Layout<Shape<Int<kHeadDim>, Int<kBlockN>>,
                                           Stride<Int<kBlockN>, _1>>;

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

    // Copy atoms - use UniversalCopy for simple row-major SMEM layouts
    // (SM75_U32x4_LDSM_N requires swizzled ldmatrix-compatible layouts)
    using SmemCopyAtomA = Copy_Atom<UniversalCopy<uint8_t>, uint8_t>;
    using SmemCopyAtomB = Copy_Atom<UniversalCopy<uint8_t>, uint8_t>;
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<uint8_t>, uint8_t>;

    // Shared memory layout — aggressive overlapping to fit within 101 KB limit
    // Phase 1 (GEMM-I): smem_q + smem_k + smem_v + sfq + sfk + sfv
    // Phase 2 (GEMM-II): smem_q→(reused as P), smem_k→(reused as V^T), sfk→(reused as sfp+sfvt)
    // V stays in smem_v during both phases for the current stage
    struct TensorStorage : cute::aligned_struct<128> {
        // Q is loaded once and stays through GEMM-I, then repurposed for P in GEMM-II
        union {
            alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutQ>> smem_q;
            alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutP>> smem_p;
        };
        // K (staged) is consumed in GEMM-I, then repurposed for V^T in GEMM-II
        union {
            alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutK>> smem_k;
            alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutVtSingleStage>> smem_vt;
        };
        // V stays through both phases
        alignas(1024) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutV>> smem_v;
        // Scale factors: SFQ stays, SFK/SFV reused for SFP/SFVt
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
        uint8_t const* ptr_SFV; index_t sfv_batch_stride, sfv_head_stride, sfv_row_stride;
        float softmax_scale, softmax_scale_log2, softcap_val;
        int window_size_left, window_size_right, seqlen_q, seqlen_k;
    };

    static Params to_underlying_arguments(Arguments const& a) {
        return {a.ptr_Q, a.q_batch_stride, a.q_row_stride, a.q_head_stride,
                a.ptr_K, a.k_batch_stride, a.k_row_stride, a.k_head_stride,
                a.ptr_V, a.v_batch_stride, a.v_row_stride, a.v_head_stride,
                a.ptr_SFQ, a.sfq_batch_stride, a.sfq_head_stride, a.sfq_row_stride,
                a.ptr_SFK, a.sfk_batch_stride, a.sfk_head_stride, a.sfk_row_stride,
                a.ptr_SFV, a.sfv_batch_stride, a.sfv_head_stride, a.sfv_row_stride,
                a.softmax_scale, float(a.softmax_scale * M_LOG2E), a.softcap_val,
                a.window_size_left, a.window_size_right, a.seqlen_q, a.seqlen_k};
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

    CUTLASS_DEVICE static void load_sf(uint8_t* dst, uint8_t const* src,
        int rows, int cols, index_t stride, int row_off, int tid, int nthreads) {
        if (src == nullptr) {
            // No scale factors provided — fill with identity (UE8M0 = 127 = 2^0)
            for (int i = tid; i < rows*cols; i += nthreads) { dst[i] = 127; }
            return;
        }
        for (int i = tid; i < rows*cols; i += nthreads) {
            dst[i] = src[(row_off + i/cols)*stride + i%cols];
        }
    }

    /// Fill P identity scale factors (UE8M0 = 127 = 2^0 = 1.0) in SMEM
    CUTLASS_DEVICE static void fill_identity_sf(uint8_t* dst, int count, int tid, int nthreads) {
        for (int i = tid; i < count; i += nthreads) {
            dst[i] = 127;  // UE8M0 encoding of 2^0 = 1.0
        }
    }

    /// Store FP8 tensor from registers to SMEM (simple cooperative store)
    CUTLASS_DEVICE static void store_fp8_to_smem(
        uint8_t* smem_dst, cutlass::float_e4m3_t const* reg_src,
        int count, int tid, int nthreads)
    {
        // Each thread stores its own elements
        for (int i = 0; i < count; ++i) {
            smem_dst[tid * count + i] = reinterpret_cast<uint8_t const*>(reg_src)[i];
        }
    }

    /// Transpose V in SMEM: [kBlockN, kHeadDim] -> [kHeadDim, kBlockN]
    CUTLASS_DEVICE static void transpose_v_smem(
        uint8_t const* src, uint8_t* dst,
        int tid, int nthreads)
    {
        // Simple byte-level transpose
        constexpr int rows = kBlockN;
        constexpr int cols = kHeadDim;
        int total = rows * cols;
        for (int i = tid; i < total; i += nthreads) {
            int r = i / cols;
            int c = i % cols;
            dst[c * rows + r] = src[r * cols + c];
        }
    }

    template <typename FrgTensorO>
    CUTLASS_DEVICE void mha_fwd(
        Params const& p, TensorStorage& s, FrgTensorO& tOrO, float* lse_arr,
        int m_block, int bidb, int bidh, int bidh_kv, int tid)
    {
        int const block_m = kBlockM;
        int const block_n = kBlockN;
        int const m_start = m_block * block_m;
        int const n_block_max = cute::ceil_div(p.seqlen_k, block_n);

        TiledMma tiled_mma;
        auto thread_mma = tiled_mma.get_thread_slice(tid);
        auto tile_shape_mnk = tile_shape(tiled_mma);

        Tensor acc_ref = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
        static constexpr int kAccRows = decltype(size<0>(flash::convert_layout_acc_rowcol(acc_ref.layout())))::value;
        flash::Softmax<kAccRows, 8> softmax(p.softmax_scale_log2);
        clear(tOrO);

        // ====== Load Q + SFQ to SMEM (once per M-block) ======
        load_tile(s.smem_q.data(), reinterpret_cast<uint8_t const*>(p.ptr_Q)
            + bidb*p.q_batch_stride + bidh*p.q_head_stride,
            kBlockM, kHeadDim, p.q_row_stride, m_start, tid, NumMmaThreads);
        load_sf(s.smem_sfq.data(), p.ptr_SFQ
            + bidb*p.sfq_batch_stride + bidh*p.sfq_head_stride,
            kBlockM, kSFCols, p.sfq_row_stride, m_start, tid, NumMmaThreads);
        // Pre-fill P identity scale factors (for GEMM-II A-operand)
        fill_identity_sf(s.smem_sfp.data(), cute::cosize_v<SmemLayoutAtomSFP>, tid, NumMmaThreads);
        __syncthreads();

        // ====== Partition Q + SFQ from SMEM ======
        Tensor sQ = make_tensor(make_smem_ptr(s.smem_q.data()), SmemLayoutQ{});
        Tensor tCrQ = thread_mma.partition_fragment_A(sQ);
        auto copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto thr_copy_A = copy_A.get_thread_slice(tid);
        Tensor tCsQ = thr_copy_A.partition_S((sQ));
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

        // ====== N-block loop ======
        for (int n_blk = n_block_max-1; n_blk >= 0; --n_blk) {
            int const n_start = n_blk * block_n;
            int const stg = n_blk % kStages;

            // ------ Load K + SFK ------
            load_tile(s.smem_k.data() + stg*kBlockN*kHeadDim,
                reinterpret_cast<uint8_t const*>(p.ptr_K) + bidb*p.k_batch_stride + bidh_kv*p.k_head_stride,
                kBlockN, kHeadDim, p.k_row_stride, n_start, tid, NumMmaThreads);
            load_sf(s.smem_sfk.data() + stg*cute::cosize_v<SmemLayoutAtomSFK>,
                p.ptr_SFK + bidb*p.sfk_batch_stride + bidh_kv*p.sfk_head_stride,
                kBlockN, kSFCols, p.sfk_row_stride, n_start, tid, NumMmaThreads);

            // ------ Load V + SFV ------
            load_tile(s.smem_v.data() + stg*kBlockN*kHeadDim,
                reinterpret_cast<uint8_t const*>(p.ptr_V) + bidb*p.v_batch_stride + bidh_kv*p.v_head_stride,
                kBlockN, kHeadDim, p.v_row_stride, n_start, tid, NumMmaThreads);
            // V scales for GEMM-II: [kHeadDim, kBlockN/32] = per-32 seq positions per head dim col
            // Loaded as [kBlockN/32, kHeadDim] then transposed to [kHeadDim, kBlockN/32]
            load_sf(s.smem_sfv.data() + stg*cute::cosize_v<SmemLayoutAtomSFK>,
                p.ptr_SFV + bidb*p.sfv_batch_stride + bidh_kv*p.sfv_head_stride,
                kBlockN, kSFCols, p.sfv_row_stride, n_start, tid, NumMmaThreads);
            __syncthreads();

            // ====== GEMM-I: S = Q @ K^T ======
            using SmemLayoutKStage = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
            Tensor sK_stg = make_tensor(make_smem_ptr(
                s.smem_k.data() + stg*kBlockN*kHeadDim), SmemLayoutKStage{});
            Tensor tCrK = thread_mma.partition_fragment_B(sK_stg);
            auto copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
            auto thr_copy_B = copy_B.get_thread_slice(tid);
            Tensor tCsK = thr_copy_B.partition_S((sK_stg));
            Tensor tCrK_v = thr_copy_B.retile_D(tCrK);

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

            // ====== Online softmax ======
            if (n_blk == n_block_max - 1) {
                softmax.template online_softmax<true>(acc_s);
            } else {
                auto sc = softmax.template max_get_scale<false>(acc_s);
                softmax.rescale_o(tOrO, sc);
                softmax.template online_softmax<false>(acc_s);
            }

            // ====== GEMM-II: O += P @ V (block-scaled MMA) ======
            // P[kBlockM, kBlockN] is the A-operand (attention weights)
            // V^T[kHeadDim, kBlockN] is the B-operand (transposed values)
            // K-dim of GEMM-II = kBlockN, same as GEMM-I since all dims = 128

            // Step 1: Convert P from FP32 to FP8 and store to SMEM
            {
                Tensor tOrP = make_tensor_like<Element>(acc_s);
                flash::convert_type_out(acc_s, tOrP);

                // Scatter P to correct (row, col) positions in SMEM
                int const warp_idx_l = tid / 32;
                int const lane_idx_l = tid % 32;
                int const warp_m_l = warp_idx_l;  // 8x1: all warps along M
                int const lane_row_l = lane_idx_l / 4;
                int const lane_col_l = lane_idx_l % 4;

                Tensor tOrP_rc = make_tensor(tOrP.data(),
                    flash::convert_layout_acc_rowcol(tOrP.layout()));

                auto* p_smem = s.smem_p.data();
                #pragma unroll
                for (int mi = 0; mi < size<0>(tOrP_rc); ++mi) {
                    int const row = warp_m_l * 16 + (mi / 2) * 16 + lane_row_l + (mi % 2) * 8;
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrP_rc); ++ni) {
                        int const col = (ni / 2) * 8 + lane_col_l * 2 + (ni % 2);
                        p_smem[row * kBlockN + col] = reinterpret_cast<uint8_t const&>(tOrP_rc(mi, ni));
                    }
                }
            }

            // Step 2: Transpose V in SMEM: [kBlockN, kHeadDim] -> V^T[kHeadDim, kBlockN]
            // smem_vt aliases smem_k (safe: K consumed by GEMM-I)
            __syncthreads();
            transpose_v_smem(
                s.smem_v.data() + stg * kBlockN * kHeadDim,
                s.smem_vt.data(),
                tid, NumMmaThreads);

            // Step 3: Fill identity scale factors for P and V^T
            // smem_sfp aliases smem_sfk, smem_sfvt aliases smem_sfv (safe: consumed by GEMM-I)
            fill_identity_sf(s.smem_sfp.data(), cute::cosize_v<SmemLayoutAtomSFP>, tid, NumMmaThreads);
            fill_identity_sf(s.smem_sfvt.data(), cute::cosize_v<SmemLayoutAtomSFVt>, tid, NumMmaThreads);
            __syncthreads();

            // Step 4: Partition P from SMEM as GEMM-II A-operand
            Tensor sP = make_tensor(make_smem_ptr(s.smem_p.data()), SmemLayoutP{});
            Tensor tCrP = thread_mma.partition_fragment_A(sP);
            auto copy_P_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
            auto thr_copy_P = copy_P_A.get_thread_slice(tid);
            Tensor tCsP = thr_copy_P.partition_S(sP);
            Tensor tCrP_v = thr_copy_P.retile_D(tCrP);

            // Partition SFP (identity) — reuse SFQ's layout (same dimensions)
            Tensor sSFP = make_tensor(make_smem_ptr(
                reinterpret_cast<ElementSF*>(s.smem_sfp.data())), SmemLayoutAtomSFP{});
            Tensor tCrSFP = sm120_partition_fragment_SFA(sSFP, thread_mma);
            auto sfp_copy = make_tiled_copy_impl(SmemCopyAtomSF{},
                sm120_get_layoutSFA_TV(tiled_mma),
                make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
            auto sfp_thr = sfp_copy.get_thread_slice(tid);
            Tensor tCsSFP = sfp_thr.partition_S(sSFP);
            Tensor tCrSFP_v = sfp_thr.retile_D(tCrSFP);

            // Step 5: Partition V^T from SMEM as GEMM-II B-operand
            Tensor sVt = make_tensor(make_smem_ptr(s.smem_vt.data()), SmemLayoutVtSingleStage{});
            Tensor tCrVt = thread_mma.partition_fragment_B(sVt);
            auto copy_Vt_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
            auto thr_copy_Vt = copy_Vt_B.get_thread_slice(tid);
            Tensor tCsVt = thr_copy_Vt.partition_S(sVt);
            Tensor tCrVt_v = thr_copy_Vt.retile_D(tCrVt);

            // Partition SFVt (identity) — reuse SFK's layout (same dimensions)
            Tensor sSFVt = make_tensor(make_smem_ptr(
                reinterpret_cast<ElementSF*>(s.smem_sfvt.data())), SmemLayoutAtomSFVt{});
            Tensor tCrSFVt = sm120_partition_fragment_SFB(sSFVt, thread_mma);
            auto sfvt_copy = make_tiled_copy_impl(SmemCopyAtomSF{},
                sm120_get_layoutSFB_TV(tiled_mma),
                make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
            auto sfvt_thr = sfvt_copy.get_thread_slice(tid);
            Tensor tCsSFVt = sfvt_thr.partition_S(sSFVt);
            Tensor tCrSFVt_v = sfvt_thr.retile_D(tCrSFVt);

            // Step 6: Execute GEMM-II with block-scaled MMA
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

            __syncthreads();
        }

        // ====== Finalize softmax: normalize O and compute LSE ======
        auto scores_scale = softmax.finalize();
        // Apply final 1/sum normalization to O
        softmax.rescale_o(tOrO, scores_scale);
        // Extract actual LSE values (stored in row_sum after finalize)
        for (int mi = 0; mi < size(softmax.row_sum); ++mi) lse_arr[mi] = softmax.row_sum(mi);
    }
};

}  // namespace flash
