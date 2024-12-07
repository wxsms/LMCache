#pragma once

#include <optional>
#include <torch/library.h>

#include "core/scalar_type.hpp"

// NOTE(Jiayi): starts
void paged_attention_compact_v1(
    torch::Tensor& logits_store,
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const int64_t tp_rank, const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void rotary_embedding_k(
    torch::Tensor& positions,
    torch::Tensor& key, int64_t head_size,
    torch::Tensor& cos_sin_cache, bool is_neox);

void rotary_embedding_k_fused(
    torch::Tensor& old_positions,
    torch::Tensor& new_positions,
    torch::Tensor& key, int64_t head_size,
    torch::Tensor& cos_sin_cache, bool is_neox);

void rotary_embedding_k_fused_paged(
    torch::Tensor& old_positions,
    torch::Tensor& new_positions,
    torch::Tensor& key_cache, 
    torch::Tensor& slot_mapping,
    int64_t head_size,
    torch::Tensor& cos_sin_cache, bool is_neox);
// NOTE(Jiayi): ends

void paged_attention_v1(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const int64_t tp_rank, const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);