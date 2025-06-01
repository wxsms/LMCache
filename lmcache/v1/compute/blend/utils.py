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
from typing import Dict

from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.v1.cache_engine import CacheEngine
from lmcache.v1.gpu_connector import GPUConnectorInterface

class LMCacheBlenderBuilder:
    _blenders: Dict[str, LMCBlender] = {}

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        cache_engine: CacheEngine,
        gpu_connector: GPUConnectorInterface,
    ):
        """
        Get or create a blender for the given instance_id.
        """
        
        if instance_id not in cls._blenders:
            logger.info(f"Creating blender for {instance_id}")
            vllm_model = VLLMModelTracker.get(instance_id)
            blender = LMCBlender(
                cache_engine=cache_engine,
                gpu_connector=gpu_connector,
                vllm_model=vllm_model,
            )
            cls._blenders[instance_id] = blender
        else:
            logger.info(f"Blender for {instance_id} already exists, returning the original one.")
        return cls._blenders[instance_id]
    
    @classmethod
    def get(
        cls,
        instance_id: str,
    ) -> nn.Module:
        """
        Get the blender by instance_id.
        """
        if instance_id not in cls._blenders:
            raise ValueError(f"Blender for {instance_id} not found.")
        return cls._blenders[instance_id]