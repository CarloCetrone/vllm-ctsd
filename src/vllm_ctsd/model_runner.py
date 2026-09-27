import logging
import torch
from typing import Dict

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.sampler import SamplerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ctsd.config import CTSDConfig
from vllm_ctsd.tree_attention import CTSDTreeAttentionMetadata
from vllm_ctsd.tree_bias import build_tree_attn_bias
from vllm_ctsd.tree_search import ctsd_select, expand_level, root_topk
from vllm_ctsd.tree_state import (
    CTSDRequestState,
    create_initial_state,
    expand_frontier_to_leaves,
    steady_state_select_and_reroot,
)

logger = logging.getLogger("vllm_ctsd")


class CTSDGPUModelRunner(GPUModelRunner):
    """
    Model runner for Cautious Tree Search Decoding (CTSD).
    Subclasses the vLLM V1 GPUModelRunner to evaluate tree candidates on GPU.
    Supports persistent interior state across decoding steps.
    """

    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        self._ctsd_config = CTSDConfig.from_env()
        self._ctsd_enabled = (
            self._ctsd_config.enabled and self._ctsd_config.breadth >= 2
        )
        self._ctsd_states: Dict[str, CTSDRequestState] = {}

        if self._ctsd_enabled:
            logger.info(
                "Initializing CTSDGPUModelRunner: breadth=%d, depth=%d",
                self._ctsd_config.breadth,
                self._ctsd_config.depth,
            )
            topology = self._ctsd_config.build_topology()
            self._ctsd_topology = topology
            self._ctsd_tree_size = self._ctsd_config.tree_size
            # Pre-compute and slice tree attention bias to [tree_size, tree_size].
            # Index 0 is root, which resides in prefix (before tree_start_pos)
            # and is attended to causally with 0.0 bias in the kernel.
            self._ctsd_tree_attn_bias = build_tree_attn_bias(
                topology, device, torch.float32
            )[1:, 1:].contiguous()

            # Pre-cache tensors on GPU
            self._path_indices = torch.tensor(
                topology["path_indices"], dtype=torch.int64, device=device
            )
            self._path_lengths = torch.tensor(
                topology["path_lengths"], dtype=torch.float32, device=device
            )
            self._path_first_token_idx = torch.tensor(
                topology["path_first_token_idx"],
                dtype=torch.int64,
                device=device,
            )
            self._level_counts = topology["level_counts"]
            self._cu_level_counts = topology["cu_level_counts"]

            # Force eager execution (no CUDA graphs for tree search)
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        if self._ctsd_enabled:
            self._init_ctsd_attention_layers()

    def _init_ctsd_attention_layers(self):
        try:
            from vllm_ctsd.tree_attention import (
                CTSDTreeAttentionImpl,
                CTSDTreeAttentionMetadata,
            )
            from vllm.v1.attention.backend import AttentionType

            static_fc = getattr(self.compilation_config, "static_forward_context", {})
            initialized_count = 0
            for name, layer in static_fc.items():
                if hasattr(layer, "impl") and not hasattr(layer, "_ctsd_tree_impl"):
                    impl = layer.impl
                    tree_impl = CTSDTreeAttentionImpl(
                        num_heads=impl.num_heads,
                        head_size=impl.head_size,
                        scale=impl.scale,
                        num_kv_heads=impl.num_kv_heads,
                        alibi_slopes=getattr(impl, "alibi_slopes", None),
                        sliding_window=getattr(layer, "sliding_window", None),
                        kv_cache_dtype=impl.kv_cache_dtype,
                        logits_soft_cap=getattr(impl, "logits_soft_cap", None),
                        attn_type=getattr(impl, "attn_type", AttentionType.DECODER),
                        kv_sharing_target_layer_name=getattr(impl, "kv_sharing_target_layer_name", None),
                    )
                    tree_impl.sliding_window = getattr(impl, "sliding_window", (-1, -1))
                    layer._ctsd_tree_impl = tree_impl

                    orig_fwd = impl.forward

                    def make_inst_forward(orig, tr_impl):
                        def inst_forward(attn_layer, query, key, value, kv_cache, attn_metadata, **kwargs):
                            if isinstance(attn_metadata, CTSDTreeAttentionMetadata):
                                return tr_impl.forward(attn_layer, query, key, value, kv_cache, attn_metadata, **kwargs)
                            return orig(attn_layer, query, key, value, kv_cache, attn_metadata, **kwargs)
                        return inst_forward

                    impl.forward = make_inst_forward(orig_fwd, tree_impl)
                    initialized_count += 1
            if initialized_count > 0:
                logger.info("CTSDGPUModelRunner: attached tree attention to %d layers", initialized_count)
        except Exception:
            logger.exception("CTSDGPUModelRunner: error pre-initializing tree attention layers")

    def _get_block_table_tensor(self) -> torch.Tensor:
        """Extract the 1D device tensor of block IDs for the active request."""
        bt = self.input_batch.block_table[0]
        if hasattr(bt, "get_device_tensor"):
            return bt.get_device_tensor(1)[0]
        elif hasattr(bt, "block_table") and hasattr(bt.block_table, "gpu"):
            return bt.block_table.gpu[0]
        return bt[0]

    def sample_tokens(self, grammar_output):
        if (
            self.execute_model_state is None
            or not self._ctsd_enabled
            or len(self.input_batch.req_ids) != 1
        ):
            return super().sample_tokens(grammar_output)

        saved_state = self.execute_model_state
        try:
            return self._ctsd_sample_tokens(grammar_output)
        except Exception:
            logger.warning(
                "CTSD step failed; falling back to standard decode",
                exc_info=True,
            )
            self.execute_model_state = saved_state
            return super().sample_tokens(grammar_output)

    def _ctsd_sample_tokens(self, grammar_output):
        # 1. Unpack ephemeral state
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        self.execute_model_state = None

        root_hidden = sample_hidden_states  # [1, H]
        req_id = self.input_batch.req_ids[0]

        # Clean up any state for inactive requests
        active_req_ids = set(self.input_batch.req_ids)
        for dead_id in list(self._ctsd_states.keys()):
            if dead_id not in active_req_ids:
                self._ctsd_states.pop(dead_id, None)

        # 2. Evaluate CTSD tree forward and select winning branch
        committed = self._ctsd_step(req_id, root_hidden, scheduler_output, slot_mappings)

        # 3. Construct SamplerOutput with committed token
        sampler_output = SamplerOutput(
            sampled_token_ids=torch.tensor(
                [[committed]], device=self.device, dtype=torch.int32
            ),
            logprobs_tensors=None,
        )

        # 4. Bookkeeping synchronization
        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        # Clean up state on request completion
        if req_id in invalid_req_indices:
            self._ctsd_states.pop(req_id, None)

        # 5. Build ModelRunnerOutput
        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=None,
            ec_connector_output=ec_connector_output
            if self.supports_mm_inputs
            else None,
            num_nans_in_logits=num_nans_in_logits,
            cudagraph_stats=cudagraph_stats,
        )

        if getattr(self, "use_async_scheduling", False):
            try:
                from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

                async_output = AsyncGPUModelRunnerOutput(
                    model_runner_output=output,
                    sampled_token_ids=sampler_output.sampled_token_ids,
                    logprobs_tensors=sampler_output.logprobs_tensors,
                    invalid_req_indices=invalid_req_indices,
                    async_output_copy_stream=self.async_output_copy_stream,
                    vocab_size=self.input_batch.vocab_size,
                )
                self.input_batch.set_async_sampled_token_ids(
                    async_output.sampled_token_ids_cpu,
                    async_output.async_copy_ready_event,
                )
                return async_output
            except Exception:
                logger.warning(
                    "Failed to create AsyncGPUModelRunnerOutput, returning synchronous output",
                    exc_info=True,
                )

        return output

    def _ctsd_step(self, req_id, root_hidden, scheduler_output, slot_mappings):
        """Full tree forward pass for each decode step."""
        root_logits = self.model.compute_logits(root_hidden).squeeze(0)
        tree_logits_per_level, tree_token_ids, tree_start_pos = self._ctsd_forward_tree(
            root_hidden, scheduler_output, slot_mappings
        )
        committed, winning_path_idx, _ = ctsd_select(
            tree_logits_per_level,
            tree_token_ids,
            self._path_indices,
            self._path_lengths,
            self._path_first_token_idx,
            root_logits=root_logits,
        )
        return committed

    def _ctsd_cold_start(self, req_id, root_hidden, scheduler_output, slot_mappings):
        """Full tree forward pass for initial cold-start decode step."""
        root_logits = self.model.compute_logits(root_hidden).squeeze(0)
        tree_logits_per_level, tree_token_ids, tree_start_pos = self._ctsd_forward_tree(
            root_hidden, scheduler_output, slot_mappings
        )
        committed, winning_path_idx, _ = ctsd_select(
            tree_logits_per_level,
            tree_token_ids,
            self._path_indices,
            self._path_lengths,
            self._path_first_token_idx,
            root_logits=root_logits,
        )

        all_node_logits = torch.cat([root_logits.unsqueeze(0), tree_logits_per_level], dim=0)
        all_tree_token_ids = torch.cat(
            [torch.tensor([-1], dtype=torch.int64, device=tree_token_ids.device), tree_token_ids]
        )

        _, state = create_initial_state(
            req_id=req_id,
            breadth=self._ctsd_config.breadth,
            depth=self._ctsd_config.depth,
            tree_start_pos=tree_start_pos,
            root_logits=root_logits,
            tree_token_ids=all_tree_token_ids,
            all_node_logits=all_node_logits,
            winning_path_idx=winning_path_idx,
            paths=self._ctsd_topology["paths"],
            cu_level_counts=self._ctsd_topology["cu_level_counts"],
        )
        return committed, state

    def _ctsd_steady_state(self, state, scheduler_output, slot_mappings):
        """Steady-state decode step forwarding only the B^D new leaves."""
        breadth = self._ctsd_config.breadth
        depth = self._ctsd_config.depth
        device = self.device
        num_leaves = breadth ** depth

        # 1. Expand frontier to new leaves
        leaf_token_ids, leaf_parent_lps = expand_frontier_to_leaves(
            state, self._ctsd_config.temperature
        )

        # 2. Compute positions and slot mapping for the new leaves
        block_table_tensor = self._get_block_table_tensor()
        block_size = self.attn_groups[0][0].kv_cache_spec.block_size
        interior_size = self._ctsd_tree_size - num_leaves  # B + B^2 + ... + B^(D-1)

        # RoPE positions for leaves (depth D from root)
        leaf_pos = torch.full(
            (num_leaves,),
            state.tree_start_pos + depth - 1,
            dtype=torch.int64,
            device=device,
        )
        # Sequential KV cache positions for leaves so each leaf gets its own slot
        leaf_seq_pos = state.tree_start_pos + interior_size + torch.arange(
            num_leaves, dtype=torch.int64, device=device
        )
        b_nums = leaf_seq_pos // block_size
        b_ids = block_table_tensor.gather(dim=0, index=b_nums)
        slot_mapping = (b_ids * block_size + leaf_seq_pos % block_size).to(torch.int64)

        # 3. Build CTSDTreeAttentionMetadata
        row_start = interior_size
        row_end = self._ctsd_tree_size
        col_end = self._ctsd_tree_size
        tree_attn_bias = self._ctsd_tree_attn_bias[row_start:row_end, 0:col_end].contiguous()
        assert tree_attn_bias.shape == (num_leaves, self._ctsd_tree_size)

        seq_len_now = state.tree_start_pos + self._ctsd_tree_size

        meta = CTSDTreeAttentionMetadata(
            num_actual_tokens=num_leaves,
            max_query_len=num_leaves,
            query_start_loc=torch.tensor([0, num_leaves], dtype=torch.int32, device=device),
            max_seq_len=seq_len_now,
            seq_lens=torch.tensor([seq_len_now], dtype=torch.int32, device=device),
            block_table=block_table_tensor.unsqueeze(0),
            slot_mapping=slot_mapping,
            num_prefill_tokens=0,
            num_decode_tokens=num_leaves,
            num_prefills=0,
            num_decodes=1,
            tree_attn_bias=tree_attn_bias,
            tree_start_pos=state.tree_start_pos,
            tree_size=self._ctsd_tree_size,
        )

        per_layer_meta = {
            layer_name: meta for layer_name in self.attn_groups[0][0].layer_names
        }
        layer_slot_mapping = {
            layer_name: slot_mapping for layer_name in self.attn_groups[0][0].layer_names
        }

        # 4. Model forward pass over the new leaves only
        with set_forward_context(
            per_layer_meta,
            self.vllm_config,
            num_tokens=num_leaves,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=layer_slot_mapping,
        ):
            hidden = self._model_forward(
                input_ids=leaf_token_ids.to(torch.int32),
                positions=leaf_pos,
            )

        leaf_logits = self.model.compute_logits(hidden)

        # 5. Score paths, commit winning first token, and re-root
        committed, new_state = steady_state_select_and_reroot(
            state=state,
            leaf_token_ids=leaf_token_ids,
            leaf_logits=leaf_logits,
            leaf_parent_logprobs=leaf_parent_lps,
        )
        return committed, new_state

    def _ctsd_forward_tree(self, root_hidden, scheduler_output, slot_mappings):
        root_logits = self.model.compute_logits(root_hidden).squeeze(0)
        breadth = self._ctsd_config.breadth
        depth = self._ctsd_config.depth
        device = self.device

        req_id = self.input_batch.req_ids[0]
        num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 1)
        num_computed = int(self.input_batch.num_computed_tokens_cpu[0])
        # Absolute RoPE position of the first level-1 tree node.
        tree_start_pos = num_computed + num_sched

        block_table_tensor = self._get_block_table_tensor()
        block_size = self.attn_groups[0][0].kv_cache_spec.block_size

        all_level_logits = []   # list of [B^k, V]
        all_level_tokens = []   # list of [B^k] int64

        parent_logits = root_logits.unsqueeze(0)  # [1, V]
        level_offset = 0  # number of nodes in all levels before the current one

        for k in range(1, depth + 1):
            num_children_per_parent = breadth
            # Top-k for every parent at the previous level
            top_ids, _ = expand_level(parent_logits, num_children_per_parent)  # [num_parents, B]
            level_tokens = top_ids.reshape(-1)  # [num_parents * B] = [B^k]

            num_nodes_this_level = level_tokens.shape[0]
            # Absolute RoPE positions: all level-k nodes sit at depth k (RoPE position tree_start_pos + k - 1)
            level_positions = torch.full(
                (num_nodes_this_level,), tree_start_pos + k - 1,
                dtype=torch.int64, device=device,
            )

            # Sequential KV cache sequence positions so each node gets its own slot
            level_seq_pos = tree_start_pos + level_offset + torch.arange(
                num_nodes_this_level, dtype=torch.int64, device=device
            )
            block_numbers = level_seq_pos // block_size
            block_ids = block_table_tensor.gather(dim=0, index=block_numbers)
            slot_mapping = (block_ids * block_size + level_seq_pos % block_size).to(torch.int64)

            # tree_attn_bias for the Triton kernel: full [tree_size, tree_size] matrix
            tree_attn_bias = self._ctsd_tree_attn_bias

            # seq_lens for the kernel: prefix + all tree nodes written so far
            tree_nodes_written = level_offset + num_nodes_this_level
            prefix_len = tree_start_pos
            seq_len_now = prefix_len + tree_nodes_written

            meta = CTSDTreeAttentionMetadata(
                num_actual_tokens=num_nodes_this_level,
                max_query_len=num_nodes_this_level,
                query_start_loc=torch.tensor([0, num_nodes_this_level], dtype=torch.int32, device=device),
                max_seq_len=seq_len_now,
                seq_lens=torch.tensor([seq_len_now], dtype=torch.int32, device=device),
                block_table=block_table_tensor.unsqueeze(0),
                slot_mapping=slot_mapping,
                num_prefill_tokens=0,
                num_decode_tokens=num_nodes_this_level,
                num_prefills=0,
                num_decodes=1,
                tree_attn_bias=tree_attn_bias,
                tree_start_pos=tree_start_pos,
                tree_size=self._ctsd_tree_size,
            )

            per_layer_meta = {name: meta for name in self.attn_groups[0][0].layer_names}
            layer_slot_mapping = {
                name: slot_mapping for name in self.attn_groups[0][0].layer_names
            }

            with set_forward_context(
                per_layer_meta,
                self.vllm_config,
                num_tokens=num_nodes_this_level,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=layer_slot_mapping,
            ):
                hidden = self._model_forward(
                    input_ids=level_tokens.to(torch.int32),
                    positions=level_positions,
                )

            level_logits = self.model.compute_logits(hidden)  # [B^k, V]
            all_level_logits.append(level_logits)
            all_level_tokens.append(level_tokens)
            parent_logits = level_logits
            level_offset += num_nodes_this_level

        # Concatenate to BFS order: level 1, then level 2, ...
        tree_token_ids = torch.cat(all_level_tokens, dim=0)      # [tree_size]
        tree_logits_per_level = torch.cat(all_level_logits, dim=0)  # [tree_size, V]
        return tree_logits_per_level, tree_token_ids, tree_start_pos
