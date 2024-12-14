from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
import torch

from vllm.core.interfaces import AllocStatus,
from vllm.sequence import Sequence, SequenceGroup

@dataclass
class GranularBlockTable:
    # 3-d tensor with shape [layer, head, num_blocks]
    # Set compacted blocks to -1
    # A caveat of this design is that the memory usage of the block table
    # is not copacted
    physical_block_ids: torch.Tensor
    n_blocks: int
    
    def __len__(self):
        return self.n_blocks

class GranularBlockManager:
    
    # TODO(Jiayi): change the signiture here
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        
        # TODO(Jiayi): Dummy varaibles
        self.watermark = 0.01
        self.enable_caching = False
        
        # TODO(Jiayi): remove hardcodes
        self.num_kv_heads = 8
        self.num_layers = 32
        self.block_sliding_window = None
        
        # TODO: pass in real device
        self.device = "cuda"
        self.block_size = block_size
        self.replica_constant = self.num_kv_heads * self.num_layers
        self.num_total_blocks = num_gpu_blocks * self.replica_constant
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        
        self.num_free_blocks = self.num_blocks
        # on-device metadata
        self.physical_blocks_status = torch.tensor(
            [1] * self.num_blocks, device=self.device) # 0-1 tensor
        self.block_tables: Dict[int, GranularBlockTable] = {}
    
    def _get_seq_num_required_blocks(self, seq: Optional[Sequence]) -> int:
        return 0 if seq is None else seq.n_blocks # TODO: rewrite n_blocks
    
    # TODO(Jiayi): Considering batched interface (e.g., batched_allocate)
    # NOTE: the update of num_available_blocks should be done in the
    # `can_xxx` functions for `batched_xxx` functions
    
    # FUNCTIONAL FUNCTIONS
    def _allocate(self, n_blocks: int) -> torch.Tensor:
        prefsum_blocks: torch.tensor = \
            torch.cumsum(self.physical_blocks_status, dim=0)
        mask = torch.logical_and(
            prefsum_blocks <= n_blocks, 
            self.physical_blocks_status == 1)
        physical_block_ids = torch.nonzero(mask)
        
        self.physical_blocks_status[physical_block_ids] = 0
        return physical_block_ids
    
    
    def allocate(self, seq_group: SequenceGroup):
        wait_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        seq = wait_seqs[0]
        
        # TODO
        assert is_encoder_decoder is False, "Not implemented"
        assert seq_group.num_seqs() == 1, "Not implemented"
        
        n_blocks = self._get_seq_num_required_blocks(seq)
        physical_block_ids = self._allocate(n_blocks)
        
        # TODO: need to optimize the following loop
        temp_idx = 0
        physical_block_ids.reshape
        granular_block_table = GranularBlockTable(
                                physical_block_ids.reshape(self.num_layers, 
                                                           self.num_kv_heads, -1),
                                n_blocks)
        self.block_tables[seq.seq_id] = granular_block_table
        
    def batched_allocate(self, seq_group: "SequenceGroup"):
        pass
    
    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        self_num_required_blocks = self._get_seq_num_required_blocks(
            seq_group.get_seqs(status=SequenceStatus.WAITING)[0])
        
        # TODO
        cross_num_required_blocks = 0
        
        num_required_blocks = self_num_required_blocks + \
                              cross_num_required_blocks
        assert self.block_sliding_window is None, "Not implemented"
        
        if (self.num_total_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if self.num_free_blocks - num_required_blocks >= self.watermark_blocks:
            self.num_free_blocks -= num_required_blocks
            return AllocStatus.OK
        else:
            return AllocStatus.LATER
        
    def batched_can_allocate(self):
        pass
    
    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int = 0, #unused
    ) -> None:
        n_blocks = seq.n_blocks
        granular_block_table = self.block_tables[seq.seq_id]
        if len(granular_block_table) < n_blocks:
            assert len(granular_block_table) == n_blocks - 1
            sliced_granular_block_table = self._allocate(self.replica_constant)
            
            granular_block_table.physical_block_ids = torch.cat([
                    granular_block_table.physical_block_ids,
                    sliced_granular_block_table],
                    dim=-1)
            granular_block_table.n_blocks += self.replica_constant
        return []

    def batched_can_append_slots(self):
        pass
    
    def can_append_slots(
        self,
        seq_group: SequenceGroup,
        num_lookahead_slots: int = 0,
    ) -> bool:
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs * self.replica_constant <= self.num_free_blocks
    
    def compact(
        self, 
        seq: Sequence, 
        mask: torch.Tensor, # [num_layers, num_heads, num_blocks]
    ):
        granular_block_table = self.block_tables[seq.seq_id]
        physical_block_ids = granular_block_table.physical_block_ids
        self.physical_blocks_status[physical_block_ids.flatten()[mask]] = 1
        physical_block_ids[mask] = -1
        
        num_blocks_to_free = torch.sum(mask)
        self.num_free_blocks += num_blocks_to_free
        granular_block_table.n_blocks -= num_blocks_to_free
        
    
    def batched_compact(self):
        pass
    
    def free(self, seq: Sequence):
        granular_block_table = self.block_tables[seq.seq_id]
        physical_block_ids = granular_block_table.physical_block_ids
        
        # TODO(Jiayi): fuse the following two operations
        mask = physical_block_ids.flatten() != -1
        self.physical_blocks_status[physical_block_ids.flatten()[mask]] = 0
        del self.block_tables[seq.seq_id]
        
        num_blocks_to_free = torch.sum(mask)
        self.num_free_blocks += num_blocks_to_free

    
    def get_block_table(self, seq: Sequence) -> torch.Tensor:
        return self.block_tables[seq.seq_id].physical_block_ids
    
    
    # DUMMY FUNCTIONS
    def mark_blocks_as_computed(
        self, 
        seq_group: SequenceGroup,
        token_chunk_size: int
    ):
        pass
    
    def swap_in(self):
        pass
    
    def can_swap_in(self):
        pass
    
    def swap_out(self):
        pass
    
    def can_swap_out(self):
        pass
    
    def access_all_blocks_in_sequence(self):
        pass
    
    # Other Functions
    # NotImplementedError
    
    
        
    