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


class SinkCompactor(BaseLocalCompactor):
    """
    SteamingLLM-like compactor
    Always retain the first 4 tokens (attention sinks)
    """
    def __init__(self, compactor_metadata):
        super().__init__(compactor_metadata)
        
        self.min_window_size = 512
        self.max_window_size = 1024
        self.num_sink = 4
        
        
    
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
        No `imp_scores` for AttentionSink
        Do nothing
        """
        pass
    
    # TODO (Jiayi): please fuse the positional encoding
    def adjust_positional_encoding(
        self,
        old_positions,
        new_positions,
        old_keys: torch.Tensor,
    ):
        """
        reverse and recover the positional encoding
        """
        
        new_keys = self.reverse_rotary_emb(
            old_positions,
            new_positions,
            old_keys,
            is_reverse=False,
            is_fuse=True,
        )
        return new_keys
    
    
    def compute_indices(self, seq_id, seq_len):
        """
        
        """
        num_last = self.min_window_size - self.num_sink
        
        sink_indices = [i for i in range(self.num_sink)]
        last_indices = [i for i in range(seq_len - num_last,
                                         seq_len)]
        compacted_indices = [torch.tensor([sink_indices + last_indices] * self.num_kv_heads,
                                          device=self.device)
            for i in range(self.num_layers)]
        
        return compacted_indices