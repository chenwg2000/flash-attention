// SM120 FP8 forward hdim64 stub — kernel does not compile for hdim=64 (tile_to_shape mismatch).
// Provide the symbol so the linker is happy; throw at runtime if actually called.

#include "flash.h"
#include <cutlass/numeric_types.h>
#include <stdexcept>

#ifndef FLASHATTENTION_DISABLE_FP8
#ifndef FLASHATTENTION_DISABLE_HDIM64

template <typename T, int kHeadDim>
void run_mha_fwd_sm120_(Flash_fwd_params &params, cudaStream_t stream);

template <>
void run_mha_fwd_sm120_<cutlass::float_e4m3_t, 64>(Flash_fwd_params &params, cudaStream_t stream) {
    throw std::runtime_error("SM120 FP8 forward with head_dim=64 is not supported");
}

#endif
#endif
