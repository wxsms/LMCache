import abc
from typing import Tuple, List, Dict
import torch
import queue

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.backends.utils import compute_slot_mapping
from vllm import _custom_ops as ops

from lmcache.compactor.utils import CompactorOutput
from lmcache.logging import init_logger
from lmcache_vllm.utils.positional_encoding import get_reverse_rope
from lmcache_vllm.utils.rotary_embedding import get_rope

logger = init_logger(__name__)

# FIXME(Jiayi): this LocalCompactor design need to be 
# compatible with PP/TP some how
class BaseLocalCompactor(metaclass=abc.ABCMeta):
    """
    Interface for local compactor
    """
    
    def __init__(self, compactor_metadata):
        # NOTE(Jiayi): keeping src_slot_mappings in local compactor
        # minimizes communication overhead between scheduler and worker
         
        #{seq_idx: num_layers * slot_mapping}
        self.src_slot_mappings = {}
        # track old and new positions for position recovery
        self.positions_tracker = {}
        
        # tensor: num_layer, num_head, window_limit
        #{seq_idx: imp_scores}
        # imp_scores should be initialized as the seq_id enters
        self.imp_scores = {}
        
        
        # TODO: remove this hardcode
        self.num_layers = 32
        self.num_heads = 32
        self.num_kv_heads = 8
        self.head_size = 128
        self.device = "cuda"
        self.vllm_block_size = 16
        
        # TODO(Jiayi): None-rope models might be configured differently
        self.rotary_emb = compactor_metadata.rotary_emb
        self.reverse_rotary_emb = get_reverse_rope(
            self.head_size,
            rotary_dim=self.head_size,
            max_position=self.rotary_emb.max_position_embeddings,
            base=self.rotary_emb.base,
            is_neox_style=self.rotary_emb.is_neox_style,
        )
        
        
        # TODO: better done in initialization phase
        # The following memory allocation may explode the memory if
        # `gpu_memory_utilization` is set too high in vllm
        # TODO: Also, please make it more flexible
        max_num_tokens = 100000
        
        
        # The logits buffer need to be preallocated
        # to be compatible with cuda graph
        # TODO: queue looks weird here. This queue exists only because
        # layer_idx is not available in attention module
        self.logits_buffer_queue = queue.Queue()
        for i in range(self.num_layers):
            self.logits_buffer_queue.put(
                torch.empty((self.num_heads, max_num_tokens),
                        device=self.device,
                        dtype=torch.float32)
                )
    
    
    @abc.abstractmethod
    def update_imp_scores(
        self,
        seq_id,
        idx,
        chunked_attetnion_weights):
        """
        update importance scores
        """
        
        raise NotImplementedError
    
    @abc.abstractmethod
    def compute_indices(
        self,
        seq_id,
        seq_len,
    ):
        """
        compute indices for schedulers
        compact imp_scores
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def decide_compact(
        self,
        seq_len,
    ) -> bool:
        """
        Decide whether to perform compaction
        """
        raise NotImplementedError
        
    def allocate_imp_scores(
        self,
        model_input,
    ):
        seq_group_metadata_list = model_input.seq_group_metadata_list
        idx = 0
        seq_lens = model_input.attn_metadata.seq_lens
        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            seq_ids = model_input.request_ids_to_seq_ids[request_id]
            for seq_id in seq_ids:
                if seq_id in self.imp_scores:
                    idx += 1
                    continue
                
                # FIXME(Jiayi): this incurs memory overhead if the input is too long
                # FIXME(Jiayi): this part requires fixing
                # Qizheng fix: increment buffer_size by 1 to avoid overflow
                buffer_size = max((seq_lens[idx] // self.max_window_size + 1) * \
                    self.max_window_size,
                    self.max_window_size) + 1
                
                imp_scores_temp = torch.zeros(
                    (self.num_layers, self.num_heads, buffer_size),
                    device=self.device,
                    dtype=torch.float32)
                self.imp_scores[seq_id] = imp_scores_temp
                
                idx += 1

    # FIXME(Jiayi): Extremely slow now
    # Improved a lot, requires futher optimization
    # 1. batching sequence (no kernels needed) (DONE)
    # 2. batching head 
    # 3. Let pos encoding be `inpace`, operating on paged memory directly
    # 4. Let compaction be `inplace`
    # 5. need to minimize transfer somehow (e.g., how to free a token in the middle
    # with only one block's free/allocate) -> is `block_size=1` the only way?
    def compact_memory(
        self,
        model_input_subset,
        kv_caches,
        dst_slot_mappings,
        preempt_seq_ids):
        """
        Make real memory movement here
        """
        
        attn_layers = model_input_subset.attn_layers
        start_layer = model_input_subset.start_layer
        end_layer = model_input_subset.end_layer
        
        # Qizheng: throw away things in preempt_seq_ids
        for preempt_seq_id in preempt_seq_ids:
            # import pdb; pdb.set_trace()
            self.src_slot_mappings.pop(preempt_seq_id, None)
            self.positions_tracker.pop(preempt_seq_id, None)
            self.imp_scores.pop(preempt_seq_id, None)
        
        if len(dst_slot_mappings) == 0:
            return
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()
        
        dst_slot_mapping_batched = []
        for seq_id, dst_slot_mapping in dst_slot_mappings.items():
            dst_slot_mapping = torch.tensor(dst_slot_mapping, 
                                            device=kv_caches[0][0].device)
            dst_slot_mapping_batched.append(dst_slot_mapping)
        dst_slot_mapping_batched = torch.cat(dst_slot_mapping_batched)
        
        for layer_idx in range(self.num_layers):
            
            src_slot_mapping_batched = []
            old_positions_batched = []
            new_positions_batched = []
            
            for seq_id, src_slot_mapping in self.src_slot_mappings.items():
                if seq_id not in dst_slot_mappings:
                    continue
                src_slot_mapping = torch.tensor(src_slot_mapping[layer_idx],
                                                device=kv_caches[0][0].device)
                src_slot_mapping_batched.append(src_slot_mapping)
                old_positions = torch.tensor(self.positions_tracker[seq_id][0][layer_idx],
                                             device=kv_caches[0][0].device)
                old_positions_batched.append(old_positions)
                new_positions = torch.tensor(self.positions_tracker[seq_id][1][layer_idx],
                                             device=kv_caches[0][0].device)
                new_positions_batched.append(new_positions)

            src_slot_mapping_batched = torch.cat(src_slot_mapping_batched)
            old_positions_batched = torch.cat(old_positions_batched)
            new_positions_batched = torch.cat(new_positions_batched)
            
            kv_cache = kv_caches[layer_idx]
            attn_layer = attn_layers[layer_idx]
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
        
            # perm & reshape K
            key_cache_temp = key_cache.permute(0,3,1,2,4)
            
            # TODO(Jiayi): tensor is copied here. Please avoid this
            key_cache_temp = key_cache_temp.reshape(
                            -1, self.num_kv_heads, self.head_size)
            key_cache_temp = key_cache_temp[src_slot_mapping_batched]
            
            #import pdb
            #pdb.set_trace()
            # adjust pos encoding of k
            self.adjust_positional_encoding(
                old_positions_batched,
                new_positions_batched,
                key_cache_temp,
            )
            
            # perm & reshape V
            value_cache_temp = value_cache.permute(0,3,1,2)
            value_cache_temp = value_cache_temp.reshape(
                            -1, self.num_kv_heads, self.head_size)
            value_cache_temp = value_cache_temp[src_slot_mapping_batched]
            
            assert len(src_slot_mapping_batched) == len(dst_slot_mapping_batched)
            misaligned_indices = torch.where(
                src_slot_mapping_batched != dst_slot_mapping_batched)[0]
            
            if len(misaligned_indices) == 0:
                continue
            
            # reshape_and_cache_flash is only used for flash attention
            ops.reshape_and_cache(
                key_cache_temp[misaligned_indices],
                value_cache_temp[misaligned_indices],
                key_cache,
                value_cache,
                dst_slot_mapping_batched[misaligned_indices],
                attn_layer.attn.kv_cache_dtype,
                attn_layer.attn._k_scale,
                attn_layer.attn._v_scale,
            )
        
        # pop src_slot_mapping to reduce memory usage
        for seq_id in dst_slot_mappings.keys():
            self.src_slot_mappings.pop(seq_id, None)
            self.positions_tracker.pop(seq_id, None)
        
        # end_event.record()
        # torch.cuda.synchronize()
        # run_time = start_event.elapsed_time(end_event)
        # print(f"memory compaction time, {len(dst_slot_mappings)} seqs: {run_time}")

    def clean_request_states(
        self,
        end_seq_ids,
    ):
        if end_seq_ids is None:
            return
        for end_seq_id in end_seq_ids:
            self.src_slot_mappings.pop(end_seq_id, None)
            self.positions_tracker.pop(end_seq_id, None)
            self.imp_scores.pop(end_seq_id, None)
        
    
    def post_model_update(
        self,
        kv_caches,
        model_input):
        """
        1. update imp_scores
        2. Conditionally compute indices for schedulers
        3. Conditionally update src_slot_mapping
        """
        
        # skip profile run
        is_profile_run = (kv_caches is None) or (kv_caches[0] is None)
        if is_profile_run:
            return
        
        seq_group_metadata_list = model_input.seq_group_metadata_list
        attn_meta = model_input.attn_metadata
        prefill_meta = attn_meta.prefill_metadata
        
        seq_lens = attn_meta.seq_lens
        sum_seq_len = sum(seq_lens)
        
        chunked_attetnion_weights = None
        
        # FIXME(Jiayi): we are skipping prefill for now
        is_all_prefill_run = ((attn_meta.num_prefills == len(seq_lens))\
            and prefill_meta is not None)
        
        if is_all_prefill_run:
            null_compactor_output = CompactorOutput(compacted_indices_dict={})
            return null_compactor_output
           
        chunked_attetnion_weights = []
        for i in range(self.num_layers):
            buffer = self.logits_buffer_queue.get()
            chunked_buffer = torch.split(
                buffer[:, :sum_seq_len], 
                seq_lens, dim=1)
            chunked_attetnion_weights.append(chunked_buffer)
            self.logits_buffer_queue.put(buffer)
        
        #start_event = torch.cuda.Event(enable_timing=True)
        #end_event = torch.cuda.Event(enable_timing=True)
        #start_event.record()
        
        compacted_indices_dict = {}
        idx = 0
        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            seq_ids = model_input.request_ids_to_seq_ids[request_id]
            for seq_id in seq_ids:
                # if 117 in seq_ids:
                #     import pdb; pdb.set_trace()
                if chunked_attetnion_weights is not None:
                    self.update_imp_scores(
                        seq_id,
                        idx,
                        chunked_attetnion_weights,
                    )
                seq_data = seq_group_metadata.seq_data[seq_id]
                seq_len = seq_data.get_len()
                
                # Qizheng: update position tracker
                if seq_id not in self.positions_tracker.keys():
                    range_list = [i for i in range(seq_len)]
                    self.positions_tracker[seq_id] = ([range_list] * self.num_layers, [])
                else:
                    range_list = [i for i in range(seq_len)]
                    self.positions_tracker[seq_id][0][:] = [range_list] * self.num_layers

                # Decide whether to perform compaction
                if not self.decide_compact(seq_len):
                    idx += 1
                    continue

                # # Qizheng: update position tracker
                # if seq_id not in self.positions_tracker.keys():
                #     range_list = [i for i in range(seq_len)]
                #     self.positions_tracker[seq_id] = ([range_list] * self.num_layers, [])
                # else:
                #     range_list = [i for i in range(seq_len)]
                #     self.positions_tracker[seq_id][0][:] = [range_list] * self.num_layers
                
                compacted_indices = self.compute_indices(seq_id, seq_len)
                # logger.debug(f"[Compactor] local_compactor taking effect! seq_id: {seq_id}")
                # logger.debug(f"[Compactor] seq_len at layer 0: {seq_len}"
                #              f"-> {len(compacted_indices[0])}")
                compacted_indices_dict[seq_id] = compacted_indices

                # update src_slot_mappings
                slot_mapping = []
                compute_slot_mapping(False, slot_mapping, seq_id, seq_len, 
                    0, 0, self.vllm_block_size, seq_group_metadata.block_tables)
                
                # FIXME(Jiayi): Please move this part inside the real compactor
                if seq_id not in self.positions_tracker:
                    old_positions = [[i for i in range(seq_len)] for j in range(len(compacted_indices))]
                else:
                    # Qizheng fix: old_positions should be first entry in positions_tracker
                    old_positions = self.positions_tracker[seq_id][0]
                
                updated_old_positions = []
                new_positions = []
                for layer_idx, compacted_indices_layer in enumerate(compacted_indices):
                    updated_old_positions.append(
                        [old_positions[layer_idx][i] for i in compacted_indices_layer])
                    
                    new_positions.append(
                        [i for i in range(len(compacted_indices_layer))])
                
                self.positions_tracker[seq_id] = (updated_old_positions, new_positions)
                
                
                # FIXME(Jiayi): Please use tensor operations if possible
                compacted_slot_mapping = []
                
                for compacted_indices_layer in compacted_indices:
                    compacted_slot_mapping.append(
                        [slot_mapping[i] for i in compacted_indices_layer])
                
                self.src_slot_mappings[seq_id] = compacted_slot_mapping
                idx += 1
                
        compactor_output = CompactorOutput(
            compacted_indices_dict=compacted_indices_dict,)
        
        #end_event.record()
        #torch.cuda.synchronize()
        #run_time = start_event.elapsed_time(end_event)
        #print(f"post model update time, {len(seq_group_metadata_list)} seqs: {run_time}")
        
        return compactor_output
    

