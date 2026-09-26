import logging
import torch
from typing import Dict

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.v1.outputs import AsyncGPUModelRunnerOutput, ModelRunnerOutput
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
            self._ctsd_tree_attn_bias = build_tree_attn_bias(
                topology, device, torch.float32
            )

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

    def sample_tokens(self, grammar_output):
        if (
            self.execute_model_state is None
            or not self._ctsd_enabled
            or len(self.input_batch.req_ids) != 1
        ):
            return super().sample_tokens(grammar_output)

        try:
            return self._ctsd_sample_tokens(grammar_output)
        except Exception:
            logger.warning(
                "CTSD step failed; falling back to standard decode",
                exc_info=True,
            )
            if self.execute_model_state is not None:
                return super().sample_tokens(grammar_output)
            raise

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

        # 2. Check if request has existing interior state
        if req_id not in self._ctsd_states:
            # Cold-start: full tree forward pass
            committed, state = self._ctsd_cold_start(
                req_id, root_hidden, scheduler_output, slot_mappings
            )
            self._ctsd_states[req_id] = state
        else:
            # Steady-state: expand frontier into B^D new leaves, forward only new leaves
            state = self._ctsd_states[req_id]
            committed, new_state = self._ctsd_steady_state(
                state, scheduler_output, slot_mappings
            )
            self._ctsd_states[req_id] = new_state

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

        if not self.use_async_scheduling:
            return output

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

    def _ctsd_cold_start(self, req_id, root_hidden, scheduler_output, slot_mappings):
        """Full tree forward pass for initial cold-start decode step."""
        tree_logits, tree_token_ids, tree_slot_mapping = self._ctsd_forward_tree(
            root_hidden, scheduler_output, slot_mappings
        )
        committed, winning_path_idx, _ = ctsd_select(
            tree_logits,
            tree_token_ids,
            self._path_indices,
            self._path_lengths,
            self._path_first_token_idx,
        )

        num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 1)
        num_computed = int(self.input_batch.num_computed_tokens_cpu[0])
        tree_start_pos = num_computed + num_sched

        root_logits = tree_logits[0]
        _, state = create_initial_state(
            req_id=req_id,
            breadth=self._ctsd_config.breadth,
            depth=self._ctsd_config.depth,
            tree_start_pos=tree_start_pos,
            root_logits=root_logits,
            tree_token_ids=tree_token_ids,
            all_node_logits=tree_logits,
            winning_path_idx=winning_path_idx,
            paths=self._ctsd_topology["paths"],
            cu_level_counts=self._ctsd_topology["cu_level_counts"],
        )
        return committed, state

    def _ctsd_steady_state(self, state, scheduler_output, slot_mappings):
        """Steady-state decode step forwarding only the B^D new leaves."""
        breadth = self._ctsd_config.breadth
        depth = self._ctsd_config.depth
        num_leaves = breadth ** depth

        # 1. Expand frontier to new leaves
        leaf_token_ids, leaf_parent_lps = expand_frontier_to_leaves(
            state, self._ctsd_config.temperature
        )

        # 2. Compute positions and slot mapping for the new leaves
        block_table = self.input_batch.block_table[0]
        block_size = self.attn_groups[0][0].kv_cache_spec.block_size

        leaf_pos = torch.full(
            (num_leaves,),
            state.tree_start_pos + depth - 1,
            dtype=torch.int64,
            device=self.device,
        )
        b_nums = leaf_pos // block_size
        b_ids = block_table.gather(dim=0, index=b_nums)
        slot_mapping = (b_ids * block_size + leaf_pos % block_size).to(torch.int32)

        # 3. Build CTSDTreeAttentionMetadata
        meta = CTSDTreeAttentionMetadata(
            num_actual_tokens=num_leaves,
            max_query_len=num_leaves,
            query_start_loc=torch.tensor([0, num_leaves], dtype=torch.int32, device=self.device),
            max_seq_len=int(scheduler_output.seq_lens[0]) + depth,
            seq_lens=torch.tensor([int(scheduler_output.seq_lens[0]) + depth], dtype=torch.int32, device=self.device),
            block_table=self.input_batch.block_table[0].unsqueeze(0),
            slot_mapping=slot_mapping,
            num_prefill_tokens=0,
            num_decode_tokens=num_leaves,
            num_prefills=0,
            num_decodes=1,
            tree_attn_bias=self._ctsd_tree_attn_bias[1:, 1:],
            tree_start_pos=state.tree_start_pos,
            tree_size=self._ctsd_tree_size,
        )

        per_layer_meta = {
            layer_name: meta for layer_name in self.attn_groups[0][0].layer_names
        }

        # 4. Model forward pass over the new leaves only
        with set_forward_context(
            per_layer_meta,
            self.vllm_config,
            num_tokens=num_leaves,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=slot_mapping,
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
        """Orchestrate GPU forward pass for CTSD candidate tree during cold start."""
        root_logits = self.model.compute_logits(root_hidden).squeeze(0)
        vocab_size = root_logits.shape[-1]
        tree_size = self._ctsd_tree_size
        breadth = self._ctsd_config.breadth
        depth = self._ctsd_config.depth

        tree_token_ids = torch.empty(
            tree_size + 1, dtype=torch.int64, device=self.device
        )
        tree_token_ids[0] = -1

        tree_logits = torch.zeros(
            tree_size + 1, vocab_size, dtype=torch.float32, device=self.device
        )
        tree_logits[0] = root_logits

        req_id = self.input_batch.req_ids[0]
        num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 1)
        num_computed = int(self.input_batch.num_computed_tokens_cpu[0])
        current_prefix_len = num_computed + num_sched

        block_table = self.input_batch.block_table[0]
        block_size = self.attn_groups[0][0].kv_cache_spec.block_size

        # Level 1 expansion
        top_ids, _ = root_topk(root_logits, breadth)
        tree_token_ids[1 : 1 + breadth] = top_ids
        tree_logits[1 : 1 + breadth] = root_logits.unsqueeze(0).expand(
            breadth, -1
        )

        level_starts = [1] + [
            1 + self._cu_level_counts[i]
            for i in range(len(self._cu_level_counts) - 1)
        ]

        slot_mapping_last = None

        if depth > 1:
            curr_parent_logits = None
            for lvl in range(1, depth):
                start_idx = level_starts[lvl - 1]
                cnt = self._level_counts[lvl - 1]
                end_idx = start_idx + cnt

                lvl_tokens = tree_token_ids[start_idx:end_idx]
                lvl_pos = torch.full(
                    (cnt,),
                    current_prefix_len + lvl - 1,
                    dtype=torch.int64,
                    device=self.device,
                )

                b_nums = lvl_pos // block_size
                b_ids = block_table.gather(dim=0, index=b_nums)
                lvl_slot = (b_ids * block_size + lvl_pos % block_size).to(
                    torch.int32
                )
                slot_mapping_last = lvl_slot

                # Forward pass for this level to get hidden states
                hidden = self._model_forward(
                    input_ids=lvl_tokens.to(torch.int32),
                    positions=lvl_pos,
                )
                curr_parent_logits = self.model.compute_logits(hidden)

                # Next level child tokens
                next_start = level_starts[lvl]
                next_top_ids, _ = expand_level(curr_parent_logits, breadth)
                next_cnt = cnt * breadth
                tree_token_ids[next_start : next_start + next_cnt] = (
                    next_top_ids.view(-1)
                )

                # Populate tree_logits for children with their parent distribution
                for p_idx in range(cnt):
                    c_start = next_start + p_idx * breadth
                    tree_logits[c_start : c_start + breadth] = (
                        curr_parent_logits[p_idx].unsqueeze(0).expand(
                            breadth, -1
                        )
                    )

        if slot_mapping_last is None:
            pos = torch.full(
                (breadth,),
                current_prefix_len,
                dtype=torch.int64,
                device=self.device,
            )
            b_nums = pos // block_size
            b_ids = block_table.gather(dim=0, index=b_nums)
            slot_mapping_last = (b_ids * block_size + pos % block_size).to(
                torch.int32
            )

        return tree_logits, tree_token_ids, slot_mapping_last
