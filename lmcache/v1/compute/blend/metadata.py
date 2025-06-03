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

# Standard
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import torch


@dataclass
class LMCBlendCommonMetadata:
    """
    Metadata for blending operations in LMCache.
    This class holds the necessary information for blending computations.
    """

    check_layers: List[int]
    recomp_ratios: Optional[List[float]] = None
    thresholds: Optional[List[float]] = None


@dataclass
class LMCBlendMetadata:
    """
    Metadata for blending operations in LMCache.
    This class holds the necessary information for blending computations.
    """

    imp_indices: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None

    def clean(self):
        self.imp_indices = None
        self.attn_mask = None
