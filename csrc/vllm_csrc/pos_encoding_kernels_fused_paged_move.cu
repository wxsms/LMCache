#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace lmc {


template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding_fused_paged_move(
    scalar_t* __restrict__ key_cache,   // [num_blocks, num_heads, head_size/x,
                                        // block_size, x]
    scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const scalar_t* cache_ptr, 
    const int64_t* __restrict__ old_positions, // [num_heads, num_tokens]
    const int64_t* __restrict__ new_positions, // [num_heads, num_tokens]
    const int64_t* src_slot_mapping,  // [num_heads, num_tokens]
    const int64_t* dst_slot_mapping,  // [num_heads, num_tokens]
    const int num_tokens,
    const int head_size,
    const int num_kv_heads, const int rot_dim, const int token_idx,
    const int block_size, const int x) {
  const int embed_dim = rot_dim / 2;


  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    const int64_t src_slot_idx = src_slot_mapping[head_idx * num_tokens + token_idx];
    const int64_t old_pos = old_positions[head_idx * num_tokens + token_idx];
    const int64_t new_pos = new_positions[head_idx * num_tokens + token_idx];
    
    const scalar_t* old_cos_ptr = cache_ptr + old_pos * rot_dim;
    const scalar_t* old_sin_ptr = cache_ptr + old_pos * rot_dim + embed_dim;

    const scalar_t* new_cos_ptr = cache_ptr + new_pos * rot_dim;
    const scalar_t* new_sin_ptr = cache_ptr + new_pos * rot_dim + embed_dim;

    const int64_t src_block_idx = src_slot_idx / block_size;
    const int64_t src_block_offset = src_slot_idx % block_size;


    // positional encoding starts
    int x_index, y_index;
    scalar_t old_cos, old_sin;
    scalar_t new_cos, new_sin;
    if (IS_NEOX) {
        // GPT-NeoX style rotary embedding.
        x_index = rot_offset;
        y_index = embed_dim + rot_offset;
        old_cos = VLLM_LDG(old_cos_ptr + x_index);
        old_sin = VLLM_LDG(old_sin_ptr + x_index);

        new_cos = VLLM_LDG(new_cos_ptr + x_index);
        new_sin = VLLM_LDG(new_sin_ptr + x_index);
    } else {
        // GPT-J style rotary embedding.
        x_index = 2 * rot_offset;
        y_index = 2 * rot_offset + 1;
        old_cos = VLLM_LDG(old_cos_ptr + x_index / 2);
        old_sin = VLLM_LDG(old_sin_ptr + x_index / 2);

        new_cos = VLLM_LDG(new_cos_ptr + x_index / 2);
        new_sin = VLLM_LDG(new_sin_ptr + x_index / 2);
    }

    const int x_x_idx = x_index / x;
    const int x_x_offset = x_index % x;
    const int y_x_idx = y_index / x;
    const int y_x_offset = y_index % x;

    const int64_t src_x_key_idx = 
        src_block_idx * num_kv_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_x_idx * block_size * x +
        src_block_offset * x + x_x_offset;
    
    const int64_t src_y_key_idx = 
        src_block_idx * num_kv_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + y_x_idx * block_size * x +
        src_block_offset * x + y_x_offset;

    const scalar_t x = key_cache[src_x_key_idx];
    const scalar_t y = key_cache[src_y_key_idx];
    
    const scalar_t x_reverse = x * old_cos + y * old_sin;
    const scalar_t y_reverse = y * old_cos - x * old_sin;

    const int64_t dst_slot_idx = dst_slot_mapping[head_idx * num_tokens + token_idx];
    const int64_t dst_block_idx = dst_slot_idx / block_size;
    const int64_t dst_block_offset = dst_slot_idx % block_size;

    const int64_t dst_x_key_idx = 
        dst_block_idx * num_kv_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_x_idx * block_size * x +
        dst_block_offset * x + x_x_offset;
    
    const int64_t dst_y_key_idx = 
        dst_block_idx * num_kv_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + y_x_idx * block_size * x +
        dst_block_offset * x + y_x_offset;
    
    key_cache[dst_x_key_idx] = x_reverse * new_cos - y_reverse * new_sin;
    key_cache[dst_y_key_idx] = y_reverse * new_cos + x_reverse * new_sin;

    // Inplace move value cache
    const int64_t src_x_value_idx =
        src_block_idx * num_kv_heads * head_size * block_size +
        head_idx * head_size * block_size + x_index * block_size +
        src_block_offset;
    
    const int64_t src_y_value_idx =
        src_block_idx * num_kv_heads * head_size * block_size +
        head_idx * head_size * block_size + y_index * block_size +
        src_block_offset;
    
    const int64_t dst_x_value_idx =
        dst_block_idx * num_kv_heads * head_size * block_size +
        head_idx * head_size * block_size + x_index * block_size +
        dst_block_offset;
    const int64_t dst_y_value_idx =
        dst_block_idx * num_kv_heads * head_size * block_size +
        head_idx * head_size * block_size + y_index * block_size +
        dst_block_offset;
    scalar_t dst_x_value = value_cache[src_x_value_idx];
    scalar_t dst_y_value = value_cache[src_y_value_idx];
    value_cache[dst_x_value_idx] = dst_x_value;
    value_cache[dst_y_value_idx] = dst_y_value;
    // positional encoding ends
  }

}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel_fused_paged_move(
    const int64_t* __restrict__ old_positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    
    const int64_t* __restrict__ new_positions,  // [batch_size, seq_len] or
                                            // [num_tokens]

    scalar_t* __restrict__ key_cache,  // [num_blocks, num_heads, head_size/x,
                                        // block_size, x]
    scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const int64_t* __restrict__ src_slot_mapping,  // [num_heads, num_tokens]
    const int64_t* __restrict__ dst_slot_mapping,  // [num_heads, num_tokens]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int num_tokens,
    const int num_kv_heads, const int head_size,
    const int block_size, const int x) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  //int64_t old_pos = old_positions[token_idx];
  //int64_t new_pos = new_positions[token_idx];

  //const scalar_t* old_cache_ptr = cos_sin_cache + old_pos * rot_dim;
  //const scalar_t* new_cache_ptr = cos_sin_cache + new_pos * rot_dim;

  apply_rotary_embedding_fused_paged_move<scalar_t, IS_NEOX>(
      key_cache, 
      value_cache,
      cos_sin_cache,
      old_positions, new_positions,
      src_slot_mapping,
      dst_slot_mapping,
      num_tokens,
      head_size, num_kv_heads, rot_dim,
      token_idx, 
      block_size, x);
}


}  // namespace lmc

void rotary_embedding_fused_paged_move(
    torch::Tensor& old_positions,  // [num_heads, num_tokens]
    torch::Tensor& new_positions,  // [num_heads, num_tokens]
    torch::Tensor& key_cache,    // [num_blocks, num_heads, head_size/x,
                                // block_size, x]
    torch::Tensor& value_cache,    // [num_blocks, num_heads, head_size,
                                 // block_size]
    torch::Tensor& src_slot_mapping,  // [num_heads, num_tokens]
    torch::Tensor& dst_slot_mapping,  // [num_heads, num_tokens]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  int64_t num_tokens = src_slot_mapping.size(1);
  int rot_dim = cos_sin_cache.size(1);
  int num_kv_heads = key_cache.size(1);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_kv_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_FLOATING_TYPES(key_cache.scalar_type(), "rotary_embedding_fused_paged_move", [&] {
    if (is_neox) {
      lmc::rotary_embedding_kernel_fused_paged_move<scalar_t, true><<<grid, block, 0, stream>>>(
          old_positions.data_ptr<int64_t>(),
          new_positions.data_ptr<int64_t>(),
          key_cache.data_ptr<scalar_t>(), 
          value_cache.data_ptr<scalar_t>(),
          src_slot_mapping.data_ptr<int64_t>(),
          dst_slot_mapping.data_ptr<int64_t>(),
          cos_sin_cache.data_ptr<scalar_t>(), rot_dim,
          num_tokens,
          num_kv_heads, head_size,
          block_size, x);
    } else {
      lmc::rotary_embedding_kernel_fused_paged_move<scalar_t, false>
          <<<grid, block, 0, stream>>>(
            old_positions.data_ptr<int64_t>(),
            new_positions.data_ptr<int64_t>(),
            key_cache.data_ptr<scalar_t>(), 
            value_cache.data_ptr<scalar_t>(),
            src_slot_mapping.data_ptr<int64_t>(),
            dst_slot_mapping.data_ptr<int64_t>(),
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim,
            num_tokens,
            num_kv_heads, head_size,
            block_size, x);
    }
  });
}

