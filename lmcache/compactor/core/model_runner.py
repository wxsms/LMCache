
def new_build(self) -> ModelInputForGPU:
    """Finalize the builder intermediate data and
    create on-device tensors.
    """
    # Combine and flatten intermediate data.
    input_tokens = []
    for inter_data in self.inter_data_list:
        for cur_input_tokens in inter_data.input_tokens:
            input_tokens.extend(cur_input_tokens)

    if not input_tokens:
        # This may happen when all prefill requests hit
        # prefix caching and there is no decode request.
        return self.model_input_cls()

    mrope_input_positions: Optional[List[List[int]]] = None
    if any(inter_data.mrope_input_positions is not None
            for inter_data in self.inter_data_list):
        mrope_input_positions = [[] for _ in range(3)]
        for idx in range(3):
            for inter_data in self.inter_data_list:
                msections = inter_data.mrope_input_positions
                if msections is None:
                    for _seq_input_positions in inter_data.input_positions:
                        mrope_input_positions[idx].extend(
                            _seq_input_positions)
                else:
                    for _seq_mrope_input_positions in msections:
                        mrope_input_positions[idx].extend(
                            _seq_mrope_input_positions[idx])
        input_positions = None
    else:
        input_positions = []
        for inter_data in self.inter_data_list:
            for cur_input_positions in inter_data.input_positions:
                input_positions.extend(cur_input_positions)

    seq_lens = []
    max_decode_seq_len = 0
    for inter_data in self.inter_data_list:
        seq_lens.extend(inter_data.seq_lens)
        if not inter_data.is_prompt:
            max_decode_seq_len = max(max_decode_seq_len,
                                        max(inter_data.seq_lens))
    query_lens = []
    for inter_data in self.inter_data_list:
        query_lens.extend(inter_data.query_lens)

    # Mapping from request IDs to sequence IDs. Used for Jamba models
    # that manages the cache by itself.
    request_ids_to_seq_ids = {
        data.request_id: data.seq_ids
        for data in self.inter_data_list
    }

    batch_size = len(input_tokens)
    use_captured_graph = self._use_captured_graph(batch_size,
                                                    max_decode_seq_len)

    # If cuda graph can be used, pad tensors accordingly.
    # See `capture_model` API for more details.
    # vLLM uses cuda graph only for decoding requests.
    cuda_graph_pad_size = -1
    if use_captured_graph:
        graph_batch_size = _get_graph_batch_size(batch_size)
        assert graph_batch_size >= batch_size
        cuda_graph_pad_size = graph_batch_size - batch_size
        batch_size = graph_batch_size

    # Tokens and positions.
    if cuda_graph_pad_size:
        input_tokens.extend(itertools.repeat(0, cuda_graph_pad_size))
    assert self.runner.device is not None
    input_tokens_tensor = async_tensor_h2d(input_tokens, torch.long,
                                            self.runner.device,
                                            self.runner.pin_memory)
    if mrope_input_positions is not None:
        for idx in range(3):
            mrope_input_positions[idx].extend(
                itertools.repeat(0, cuda_graph_pad_size))
        input_positions_tensor = async_tensor_h2d(mrope_input_positions,
                                                    torch.long,
                                                    self.runner.device,
                                                    self.runner.pin_memory)
    else:
        input_positions.extend(itertools.repeat(0, cuda_graph_pad_size))
        input_positions_tensor = async_tensor_h2d(input_positions,
                                                    torch.long,
                                                    self.runner.device,
                                                    self.runner.pin_memory)
    # Sequence and query lengths.
    if cuda_graph_pad_size:
        seq_lens.extend(itertools.repeat(1, cuda_graph_pad_size))

    # Attention metadata.
    attn_metadata = self.attn_metadata_builder.build(
        seq_lens, query_lens, cuda_graph_pad_size, batch_size)

    # LoRA data.
    lora_requests = set()
    lora_mapping = None
    if self.enable_lora:
        lora_requests = set(r for data in self.inter_data_list
                            for r in data.lora_requests)
        lora_index_mapping = flatten_2d_lists([
            flatten_2d_lists(inter_data.lora_index_mapping)
            for inter_data in self.inter_data_list
        ])
        if cuda_graph_pad_size:
            lora_index_mapping.extend(
                itertools.repeat(0, cuda_graph_pad_size))
        lora_prompt_mapping = flatten_2d_lists([
            flatten_2d_lists(inter_data.lora_prompt_mapping)
            for inter_data in self.inter_data_list
        ])

        lora_mapping = LoRAMapping(
            **dict(index_mapping=lora_index_mapping,
                    prompt_mapping=lora_prompt_mapping,
                    is_prefill=not self.decode_only))

    # Prompt adapter data.
    prompt_adapter_requests: Set[PromptAdapterRequest] = set()
    prompt_adapter_mapping = None
    if self.enable_prompt_adapter:
        prompt_adapter_requests = set(
            data.prompt_adapter_request for data in self.inter_data_list
            if data.prompt_adapter_request is not None)
        prompt_adapter_index_mapping = flatten_2d_lists([
            inter_data.prompt_adapter_index_mapping
            for inter_data in self.inter_data_list
        ])
        if cuda_graph_pad_size:
            prompt_adapter_index_mapping.extend(
                itertools.repeat(0, cuda_graph_pad_size))
        prompt_adapter_prompt_mapping = flatten_2d_lists([
            inter_data.prompt_adapter_prompt_mapping
            for inter_data in self.inter_data_list
        ])
        prompt_adapter_mapping = PromptAdapterMapping(
            prompt_adapter_index_mapping,
            prompt_adapter_prompt_mapping,
        )

    # Multi-modal data.
    multi_modal_inputs_list = [
        data.multi_modal_inputs for data in self.inter_data_list
        if data.multi_modal_inputs is not None
    ]
    multi_modal_kwargs = MultiModalInputs.batch(multi_modal_inputs_list)

    return self.model_input_cls(
        input_tokens=input_tokens_tensor,
        input_positions=input_positions_tensor,
        attn_metadata=attn_metadata,
        seq_lens=seq_lens,
        query_lens=query_lens,
        lora_mapping=lora_mapping,
        lora_requests=lora_requests,
        multi_modal_kwargs=multi_modal_kwargs,
        request_ids_to_seq_ids=request_ids_to_seq_ids,
        finished_requests_ids=self.finished_requests_ids,
        prompt_adapter_mapping=prompt_adapter_mapping,
        prompt_adapter_requests=prompt_adapter_requests)