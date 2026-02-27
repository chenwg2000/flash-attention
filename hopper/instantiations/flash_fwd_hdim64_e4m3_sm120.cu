// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// SM120 MXFP8 block-scaled flash attention forward kernel instantiation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template_sm120.h"

#ifndef FLASHATTENTION_DISABLE_FP8
#ifndef FLASHATTENTION_DISABLE_HDIM64
template void run_mha_fwd_sm120_<cutlass::float_e4m3_t, 64>(Flash_fwd_params &params, cudaStream_t stream);
#endif
#endif
