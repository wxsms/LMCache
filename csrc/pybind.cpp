#include <pybind11/pybind11.h>
#include "mem_kernels.cuh"
#include "cachegen_kernels.cuh"
#include <torch/torch.h>
#include <iostream>

// vllm-related kernels
#include "vllm_csrc/cache.h"
#include "vllm_csrc/cuda_utils.h"
// #include "vllm_csrc/ops.h"
#include "vllm_csrc/ops_compact.h"
#include "vllm_csrc/core/registration.h"

#include <torch/library.h>

namespace py = pybind11;

PYBIND11_MODULE(lmc_ops, m) {
    m.def("load_and_reshape_flash", &load_and_reshape_flash, "A function that loads the kv cache from paged memory");
    m.def("encode_fast_new", &encode_cuda_new);
    m.def("decode_fast_new", &decode_cuda_new);
    m.def("decode_fast_prefsum", &decode_cuda_prefsum);
    m.def("calculate_cdf", &calculate_cdf);
    
    //m.def("paged_attention_v1", &paged_attention_v1);
    m.def("paged_attention_compact_v1", &paged_attention_compact_v1);
    m.def("rotary_embedding_k", &rotary_embedding_k);
    m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
    m.def("inplace_mem_move", &inplace_mem_move);
    m.def("rotary_embedding_k_fused_paged", &rotary_embedding_k_fused_paged);

    m.def("rotary_embedding_fused_paged_move", &rotary_embedding_fused_paged_move);
    // m.def("paged_attention_v2", &paged_attention_v2);
}