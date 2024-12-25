from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
import torch

from vllm.core.interfaces import AllocStatus
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

@dataclass
class GranularBlockTable:
    # 1-d flattened tensor with shape [layer, head, max_num_blocks]
    # Set compacted blocks to -1
    # A caveat of this design is that the memory usage of the block table
    # is not compacted
    physical_block_ids: torch.Tensor
    n_blocks_layer_head: torch.Tensor # [layer, head]
    n_blocks: int
    

class GranularBlockManager:
    
    # TODO(Jiayi): change the signiture here
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int, #TODO: unused
        
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
        self.max_block_per_seq = XXX #TODO
        
        # TODO: pass in real device
        self.device = "cuda"
        self.block_size = block_size
        self.replica_constant = self.num_kv_heads * self.num_layers
        self.num_total_blocks = num_gpu_blocks * self.replica_constant
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        
        self.num_free_blocks = self.num_total_blocks
        # on-device metadata
        self.physical_blocks_status = torch.tensor(
            [1] * self.num_total_blocks, device=self.device) # 0-1 tensor
        self.block_tables: Dict[int, GranularBlockTable] = {}
    
    def _get_seq_num_required_blocks(self, seq: Optional[Sequence]) -> int:
        return 0 if seq is None else torch.sum(seq.n_blocks)
    
    # TODO(Jiayi): Considering batched interface (e.g., batched_allocate)
    # NOTE: the update of num_available_blocks should be done in the
    # `can_xxx` functions for `batched_xxx` functions
    
    # FUNCTIONAL FUNCTIONS
    def _allocate(self, n_blocks: int) -> torch.Tensor:
        
        assert self.num_free_blocks > n_blocks
        
        self.num_free_blocks -= n_blocks
        prefsum_blocks: torch.tensor = \
            torch.cumsum(self.physical_blocks_status, dim=0)
        mask = torch.logical_and(
            prefsum_blocks <= n_blocks, 
            self.physical_blocks_status == 1)
        physical_block_ids = torch.nonzero(mask).flatten()
        self.physical_blocks_status[physical_block_ids] = 0
        return physical_block_ids
    
    
    def allocate(self, seq_group: SequenceGroup):
        """
        Initialize block tables before prefill stage
        """
        wait_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        seq = wait_seqs[0]
        
        # TODO
        assert is_encoder_decoder is False, "Not implemented"
        assert seq_group.num_seqs() == 1, "Not implemented"
        
        n_blocks = self._get_seq_num_required_blocks(seq)
        physical_block_ids = self._allocate(n_blocks)
        
        # TODO: need to optimize the following memory copy
        
        physical_block_ids = physical_block_ids.reshape(
                        self.num_layers, self.num_kv_heads, -1)
        
        n_blocks_per_layer_head = physical_block_ids.shape[-1]
        
        full_block_table = torch.full(
            (self.num_layers, self.num_kv_heads, self.max_block_per_seq),
            -1,
            device=self.device,
        )
        
        full_block_table[:, :, :n_blocks_per_layer_head] = physical_block_ids
        self.block_tables[seq.seq_id] = GranularBlockTable(
            physical_block_ids=full_block_table,
            n_blocks_layer_head=torch.tensor(
                [n_blocks_per_layer_head] * self.num_layers * self.num_kv_heads,
                device=self.device),
            n_blocks = n_blocks
        )
        
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
        
        if (self.num_total_blocks - num_required_blocks < \
                self.watermark_blocks):
            return AllocStatus.NEVER
        if self.num_free_blocks - num_required_blocks >= self.watermark_blocks:
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
        n_blocks_layer_head = seq.n_blocks # [num_layer, num_head]
        granular_block_table = self.block_tables[seq.seq_id]
        mask = (n_blocks_layer_head > granular_block_table.n_blocks_layer_head)
        new_required_blocks = torch.sum(mask)
        if new_required_blocks > 0:
            new_physical_ids = self._allocate(new_required_blocks)
            layer_head_indices = torch.nonzero(mask) # [num_new_blocks, 2]
            block_indices = granular_block_table.n_blocks_layer_head[
                        layer_head_indices[:, 0], 
                        layer_head_indices[:, 1]]
            
            layer_head_block_indices = torch.cat([
                    layer_head_indices,
                    block_indices],
                dim=-1)
            block_indices += 1
            
            granular_block_table.physical_block_ids[
                    layer_head_block_indices[:, 0],
                    layer_head_block_indices[:, 1],
                    layer_head_block_indices[:, 2]
                ] = new_physical_ids
            granular_block_table.n_blocks += new_required_blocks
        return []

    def batched_can_append_slots(self):
        pass
    
    # NOTE(Jiayi): There's some reservation in the following function
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
    ):
        """
        Compact the block table of a sequence.
        First free all the blocks and then allocte the new blocks.
        """
        
        self._free(seq)
        granular_block_table = self.block_tables[seq.seq_id]
        physical_block_ids = granular_block_table.physical_block_ids
        
        # TODO(Jiayi): Optimize the following code with a kernel
        physical_block_ids._fill(-1)
        
        n_blocks_layer_head = seq.data.n_blocks_layer_head
        # Create a selection mask
        indices = torch.arange(
                self.max_block_per_seq,
                device=self.device,
            ).view(1, 1, -1).expand(
                self.num_layers, self.num_kv_heads, -1)
        mask = indices < n_blocks_layer_head.unsqueeze(-1)
        num_required_blocks = torch.sum(mask)
        physical_block_ids[mask] = self._allocate(num_required_blocks)
        granular_block_table.n_blocks_layer_head[:] =\
            n_blocks_layer_head
        granular_block_table.n_blocks = num_required_blocks
            
        
    
    def batched_compact(self):
        pass
    
    def _free(self, seq: Sequence):
        granular_block_table = self.block_tables[seq.seq_id]
        physical_block_ids = granular_block_table.physical_block_ids
        
        # TODO(Jiayi): fuse the following two operations
        mask = physical_block_ids.flatten() != -1
        self.physical_blocks_status[physical_block_ids.flatten()[mask]] = 0
        
        self.num_free_blocks += granular_block_table.n_blocks
    
    def free(self, seq: Sequence):
        self._free(seq)
        
        del self.block_tables[seq.seq_id]
        

    
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
    
    
        
    