import pytest

from lmcache.compactor import GranularBlockManager
import torch
import pdb


def test_internal_alloc():
    block_manager = GranularBlockManager(
                        block_size=16,
                        num_gpu_blocks=100, # 100 * replication_constant
                        num_cpu_blocks=0,
                        watermark=0.01,
                    )
    block_table = block_manager._allocate(16)
    assert len(block_table) == 16
    assert block_manager.num_free_blocks == \
        100*block_manager.replica_constant - 16

def test_alloc():
    block_manager = GranularBlockManager(
                        block_size=16,
                        num_gpu_blocks=100,
                        num_cpu_blocks=0,
                        watermark=0.01,
                    )
    null_seq_group = torch.load("compaction/seq_group.pt")
    import pdb
    pdb.set_trace()
    
    # TODO(Jiayi): add a sequence generator
    block_table = block_manager._allocate(16)
    assert len(block_table) == 16
    assert block_manager.num_free_blocks == 84