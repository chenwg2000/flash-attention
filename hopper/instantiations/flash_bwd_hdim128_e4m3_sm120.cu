// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// SM120 MXFP8 block-scaled flash attention backward kernel instantiation (Phase 1: dK+dV).

#include "flash_bwd_launch_template_sm120.h"

#ifndef FLASHATTENTION_DISABLE_FP8
#ifndef FLASHATTENTION_DISABLE_HDIM128
template void run_mha_bwd_sm120_<cutlass::float_e4m3_t, 128>(Flash_bwd_params &params, cudaStream_t stream);
#endif
#endif
