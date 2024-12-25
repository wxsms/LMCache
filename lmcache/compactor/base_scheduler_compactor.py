import abc
from typing import Tuple, List, Dict
import torch
from array import array

#from vllm.attention.backends.utils import compute_slot_mapping
from vllm.sequence import SequenceGroup
from lmcache.logging import init_logger
from lmcache-compactor.utils import compute_n_tokens_layer_head

logger = init_logger(__name__)

VLLM_TOKEN_ID_ARRAY_TYPE = "l"

# FIXME(Jiayi): this LocalCompactor design need to be 
# compatible with PP/TP some how
class BaseSchedulerCompactor:
    """
    Interface for scheduler compactor
    """
    
    # TODO: `dst_block_tables` is unnecessary as it already exists in input
    # metadata
    @classmethod
    def compact_blocks(
        cls,
        block_manager,
        compacted_indices_dict,,
        dst_block_tables: Dict[int, torch.Tensor],
        seq_group: SequenceGroup):
        """
        Perform slot/metadata compaction in scheduler.
        Update dst_slot_mapping
        
        """
        
        #start_event = torch.cuda.Event(enable_timing=True)
        #end_event = torch.cuda.Event(enable_timing=True)
        #start_event.record()
        
        for seq in seq_group.get_seqs():
            seq_id = seq.seq_id
            # Check whether the current seq_id needs to be compacted
            if seq_id not in compacted_indices_dict:
                continue
            
            n_tokens_layer_head = compute_n_tokens_layer_head(compacted_indices_dict[seq_id])
            
            # Update sequence here
            seq.data.n_tokens_layer_head = n_tokens_layer_head
            
            # Perform free and allocate
            block_manager.compact_blocks(seq)
            
            dst_block_tables[seq_id] = block_manager.block_tables[seq_id]
            
            
            logger.debug(f"[Compactor] base_scheduler_compactor taking effect! seq_id: {seq_id}")
            