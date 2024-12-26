from typing import List
import torch

PAD_SLOT_ID = -1



# TODO(Jiayi): `seq_len` should be a tensor with shape [num_layer, num_head]
# `start_idx` should also be a tensor with shape [num_layer, num_head]
# `slot_mapping` should be replication_constant * range_start-range_end

# Jiayi Modification starts
def new_compute_slot_mapping(is_profile_run: bool, old_slot_mapping: torch.Tensor,
                         seq_id: int, seq_len: int, context_len: int,
                         start_idx: int, block_size: int,
                         block_tables: Dict[int, torch.Tensor]) -> torch.Tensor:
# Jiayi Modification ends
    """
    Compute slot mapping.
    """
    if is_profile_run:
        # During memory profiling, the block tables are not
        # initialized yet. In this case, we just use a dummy
        # slot mapping.
        # In embeddings, the block tables are {seq_id: None}.
        slot_mapping.extend([PAD_SLOT_ID] * seq_len)
        return

    # Mask the [0, start_idx) tokens of the prompt with
    # PAD_SLOT_ID, where start_idx is max(0, seq_len -
    # sliding_window). For example, if the prompt len is 10,
    # sliding window is 8, and block size is 4, the first two
    # tokens are masked and the slot mapping will be
    # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
    padding_mask_len = max(0, start_idx - context_len)
    slot_mapping.extend([PAD_SLOT_ID] * padding_mask_len)

    range_start = max(start_idx, context_len)
    range_end = seq_len
    numel = range_end - range_start
    block_table = block_tables[seq_id]

    # numpy implementation will be faster than python if we have
    # many elements, otherwise it will be slower.
    #if numel < _COMPUTE_SLOT_MAPPING_NUMPY_NUMEL:
    #    _compute_slot_mapping_python(slot_mapping, block_table, range_start,
    #                                 range_end, block_size)
    #else:
    #    _compute_slot_mapping_numpy(slot_mapping, block_table, range_start,
    #                                range_end, block_size)
    
    num_layers, num_kv_heads, max_block_per_seq = block_table.shape
    block_size = 16
    indices = torch.arange(
            max_block_per_seq * block_size,
            device=block_table.device,
        ).view(1, 1, -1).expand(
            num_layers, num_kv_heads, -1)
    mask = (indices >= range_start.unsqueeze(-1)) & (indices < range_end.unsqueeze(-1))
    indices = indices[mask]
    indices = indices.view(num_layers, num_kv_heads, -1)
    
    block_offsets = indices % block_size
    indices //= block_size
    
    seq_slot_mapping = torch.gather(block_table, dim=2, index=indices)
    
    # TODO(Jiayi): Please optimize away this  cat with preallocation
    new_slot_mapping = torch.cat([old_slot_mapping, seq_slot_mapping], dim=2)
    
    return new_slot_mapping



def _new_add_seq_group(
        self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
        chunked_prefill_enabled: bool):
    is_prompt = inter_data.is_prompt
    block_tables = inter_data.block_tables
    computed_block_nums = inter_data.computed_block_nums

    for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
            curr_sliding_window_block) in zip(
                inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                inter_data.orig_seq_lens, inter_data.seq_lens,
                inter_data.query_lens, inter_data.context_lens,
                inter_data.curr_sliding_window_blocks):
        self.context_lens.append(context_len)
        if is_prompt:
            self.num_prefills += 1
            self.num_prefill_tokens += token_len
            self.prefill_seq_lens.append(seq_len)
        else:
            assert query_len == 1, (
                "seq_len: {}, context_len: {}, query_len: {}".format(
                    seq_len, context_len, query_len))
            self.num_decode_tokens += query_len
            self.curr_seq_lens.append(curr_seq_len)

        # Compute block table.
        # TODO(sang): Combine chunked prefill and prefix caching by
        # only allowing multiple of block_size chunk size.
        # NOTE: This only works for oooooooxxx style attention.
        block_table = []
        if inter_data.prefix_cache_hit:
            block_table = computed_block_nums
        elif ((chunked_prefill_enabled or not is_prompt)
                and block_tables is not None):
            block_table = block_tables[seq_id][-curr_sliding_window_block:]
        self.block_tables.append(block_table)

        # Jiayi Modification starts
        # Compute slot mapping.
        is_profile_run = is_block_tables_empty(block_tables)
        
        #start_idx = compute_slot_mapping_start_idx(
        #    is_prompt, query_len, context_len, self.sliding_window,
        #    self.use_v2_block_manager)
        
        new_slot_mapping = new_compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                seq_len, context_len, start_idx,
                                self.block_size, inter_data.block_tables)
        self.slot_mapping = new_slot_mapping 
        # Jiayi Modification ends

def new_build(self, seq_lens: List[int], query_lens: List[int],
            cuda_graph_pad_size: int, batch_size: int):
    """Build attention metadata with on-device tensors.

    Args:
        seq_lens: The maybe padded sequence lengths of the input sequences.
        query_lens: The query lengths of the input sequences.
        cuda_graph_pad_size: The padding size for cuda graph.
                                -1 if cuda graph is not used.
        batch_size: The maybe padded batch size.
    """
    for inter_data in self.input_builder.inter_data_list:
        self._add_seq_group(inter_data,
                            self.input_builder.chunked_prefill_enabled)

    device = self.runner.device
    use_captured_graph = cuda_graph_pad_size != -1

    max_query_len = max(query_lens)
    max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
    max_decode_seq_len = max(self.curr_seq_lens, default=0)
    num_decode_tokens = self.num_decode_tokens

    if use_captured_graph:
        self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
        self.block_tables.extend([] * cuda_graph_pad_size)
        num_decode_tokens = batch_size

        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        input_block_tables = self.runner.graph_block_tables[:batch_size]
        for i, block_table in enumerate(self.block_tables):
            if block_table:
                input_block_tables[i, :len(block_table)] = block_table
        block_tables = torch.from_numpy(input_block_tables).to(
            device, non_blocking=True)
    else:
        block_tables = make_tensor_with_pad(
            self.block_tables,
            pad=0,
            dtype=torch.int,
            device=device,
        )
    assert max_query_len > 0, "query_lens: {}".format(query_lens)

    assert device is not None
    context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                            device, self.runner.pin_memory)
    seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                        self.runner.pin_memory)
    query_lens_tensor = async_tensor_h2d(query_lens, torch.long, device,
                                            self.runner.pin_memory)
    slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                            device, self.runner.pin_memory)
    
    query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
    seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device=device)
    torch.cumsum(seq_lens_tensor,
                    dim=0,
                    dtype=seq_start_loc.dtype,
                    out=seq_start_loc[1:])
    torch.cumsum(query_lens_tensor,
                    dim=0,
                    dtype=query_start_loc.dtype,
                    out=query_start_loc[1:])

    return self._metadata_cls(  # type: ignore
        num_prefills=self.num_prefills,
        slot_mapping=slot_mapping_tensor,
        num_prefill_tokens=self.num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        seq_lens=seq_lens,
        seq_lens_tensor=seq_lens_tensor,
        max_query_len=max_query_len,
        max_prefill_seq_len=max_prefill_seq_len,
        max_decode_seq_len=max_decode_seq_len,
        query_start_loc=query_start_loc,
        seq_start_loc=seq_start_loc,
        context_lens_tensor=context_lens_tensor,
        block_tables=block_tables,
        use_cuda_graph=use_captured_graph,
    )