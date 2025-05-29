# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn

# FIXME(Jiayi): A few things need to be tested/supported:
# PP, Multimodal

class LMCLlamaModel(nn.Module):
    
    def __init__(
        self, 
        vllm_model,
        blender,
    ):
        self.vllm_model = vllm_model
        
        self.attn_layers = 
        
        self.blender = 
    
    def compute_layer(
        self,
        input_ids: torch.Tensor,
    ):
        hidden_states = self.vllm_model.get_input_embeddings(input_ids)
        residual = None
        
        for idx, layer in enumerate(
            self.vllm_model.layers[
                self.vllm_model.start_layer:self.vllm_model.end_layer]):
            # TODO(Jiayi) The last layer doesn't have to be computed
            #hidden_states, residual = layer(positions, hidden_states, residual)
            
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(
                    hidden_states, residual)
            #hidden_states = self.self_attn(positions=positions,
            #                            hidden_states=hidden_states)
            
            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [layer.self_attn.q_size, 
                 layer.self_attn.kv_size, 
                 layer.self_attn.kv_size], 
                dim=-1)
            
            # NOTE: do rope somewhere else
            #q, k = self.rotary_emb(positions, q, k)
            k, v, attn_metadata = self.blender.process_kv(k, v)
            
            # TODO: Fix this, make this our customized attention
            attn_output = self.attn_layers[idx].forward_contiguous(
                q, k, v, self.output, attn_metadata)
            
            output, _ = layer.self_attn.o_proj(attn_output)

            # Fully Connected
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual)
            hidden_states = layer.mlp(hidden_states)
            
            yield