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

from lmcache.v1.compute.blend.metadata import LMCBlendMetadata, LMCBlendCommonMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm

class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self, 
        cache_engine, 
        gpu_connector,
        vllm_model,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector
        
        self.layerwise_model = infer_model_from_vllm(vllm_model)
        self.num_layers = 
        
        # TODO (Jiayi): make this less hard-coded
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=[1],
            recomp_ratios=[0.15],
            thresholds=None,
        )
        
        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None, 
            attn_mask=None,
        )
        

    def process_kv(self, k, v, layer_id: int):
        old_k, old_v = self.gpu_connector.get_kv(layer_id)
        if layer_id in self.common_metadata.check_layers:
            diff_k = torch.sum((k.to(torch.float32)-\
                            old_k.to(torch.float32))**2,
                               dim=[1,2])
            total_len = diff_k.shape[0]
            topk_num = int(total_len*self.common_metadata.recomp_ratio)
            
            top_indices = torch.topk(diff_k, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)
            
            k, v = k[top_indices], v[top_indices]
            
            self.metadata.imp_indices = top_indices
        
        if self.metadata.imp_indices:
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
        else:
            old_k = k
            old_v = v
        
        return old_k, old_v

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might 
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self, 
        tokens: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """
        
        # TODO(Jiayi): store is currently not included in this function
        
        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(
            tokens, mask, **kwargs)
        
        next(layerwise_retriever)
        yield
        
        for i range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield
        
        self.metadata.clean()
        yield
        
    def blend(
        self, 
        tokens: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """
        layerwise_blender = self.blend_layer(
            tokens, mask, **kwargs)
        
        for i in range(self.num_layers+2):
            next(layerwise_blender)
        