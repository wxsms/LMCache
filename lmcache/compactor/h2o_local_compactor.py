import abc
from typing import Tuple, List, Dict
import torch
import queue

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.backends.utils import compute_slot_mapping
from vllm import _custom_ops as ops

from lmcache.compactor.base_local_compactor import BaseLocalCompactor
from lmcache.compactor.utils import CompactorOutput
from lmcache.logging import init_logger


logger = init_logger(__name__)

class H2OCompactor(BaseLocalCompactor):
    """
    H2O compactor
    """
    
    def __init__(self, compactor_metadata):
        super().__init__(compactor_metadata)
        
        self.min_window_size = 256
        self.max_window_size = 512
    
    def decide_compact(
        self,
        seq_len) -> bool:
        return seq_len >= self.max_window_size     
    
    def update_imp_scores(
        self,
        seq_id,
        idx,
        chunked_attetnion_weights):
        """
        Simply add the attention_weight to the existing imp_scores
        """
        
        for layer_idx in range(self.num_layers):
            attn_weight = chunked_attetnion_weights[layer_idx][idx]
            seq_len = attn_weight.shape[1]
            self.imp_scores[seq_id][layer_idx,:,:seq_len] += \
                attn_weight
        
    def adjust_positional_encoding(
        self,
        old_positions,
        new_positions,
        old_keys: torch.Tensor,
    ):
        """
        Not clearly mentioned in the paper. But seems to have better quality
        with `adjusting_positional_encoding`.
        """
        new_keys = self.reverse_rotary_emb(
            old_positions,
            new_positions,
            old_keys,
            is_reverse=False,
            is_fuse=True,
        )
        return new_keys
    
    def compute_indices(
        self,
        seq_id,
        seq_len,
    ):
        """
        compute indices for schedulers
        compact imp_scores
        """
        compacted_indices = []
        imp_score = self.imp_scores[seq_id]
        for layer_idx in range(self.num_layers):
            # sum of all heads
            imp_score_gqa = imp_score[layer_idx].view(
                -1, int(num_heads/num_kv_heads), imp_score[layer_idx].size(1))
            sum_scores_layer = torch.sum(imp_score_gqa, dim=1)
            
            imp_indices_layer = torch.topk(
                imp_score[layer_idx], k=self.min_window_size).indices
            imp_indices_layer = torch.sort(imp_indices_layer).values
            
            # TODO(Jiayi): check the following correctness
            # TODO(Jiayi): need to repeat imp_indices_layer for all heads
            # compact imp_scores
            imp_score[layer_idx,: , :self.min_window_size] = \
                torch.gather(imp_score[layer_idx], 
                             dim=-1,
                             index=imp_indices_layer)
            imp_score[layer_idx, : , self.min_window_size:] = 0
            
            # imp_indices_layer = imp_indices_layer.tolist()
            compacted_indices.append(imp_indices_layer)

        #import pdb
        #pdb.set_trace()
        return compacted_indices