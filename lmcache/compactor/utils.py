from typing import Tuple, List, Dict, Callable
from dataclasses import dataclass
import torch



@dataclass
class CompactorMetadata:
    num_gpu_blocks: int
    rotary_emb: Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
                             Tuple[torch.Tensor, torch.Tensor]]

# TODO(Jiayi): The following assumption needs to be more flexible
# Current assumption: 
# Across layers: same number, different tokens
# Across heads: same number, same tokens

@dataclass
class CompactorInput:
    # map from old slot mapping to new slot mapping
    #kv_mmaps: List[Tuple[List[int], List[int]]]
    
    # dst memory across all heads and layers
    # Since number of tokens are uniform, we can reuse block tables
    # across all layers
    
    # {seq_idx: List[int]}
    dst_slot_mappings: Dict[int, List[torch.Tensor]]
    end_seq_ids: List[int]
    
    def reset(self):
        self.dst_slot_mappings = {}
        self.end_seq_ids = []


# NOTE(Jiayi): a potential optimization is to only send the
# number of compacted tokens back to scheduler
@dataclass
class CompactorOutput:
    compacted_indices_dict: Dict[int, List[torch.Tensor]]

