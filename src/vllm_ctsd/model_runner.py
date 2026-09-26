import logging
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
from vllm.v1.outputs import AsyncGPUModelRunnerOutput, ModelRunnerOutput
from vllm.v1.sample.sampler import SamplerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ctsd.config import CTSDConfig
from vllm_ctsd.tree_bias import build_tree_attn_bias
from vllm_ctsd.tree_search import ctsd_select, expand_level, root_topk

logger = logging.getLogger("vllm_ctsd")


class CTSDGPUModelRunner(GPUModelRunner):
    """
    Model runner for Cautious Tree Search Decoding (CTSD).
    Subclasses the vLLM V1 GPUModelRunner to evaluate tree candidates on GPU.
    """

    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        self._ctsd_config = CTSDConfig.from_env()
        self._ctsd_enabled = (
            self._ctsd_config.enabled and self._ctsd_config.breadth >= 2
        )

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

        # 2. Forward pass over the CTSD tree
        tree_logits, tree_token_ids, tree_slot_mapping = self._ctsd_forward_tree(
            root_hidden, scheduler_output, slot_mappings
        )

        # 3. Select winning path and commit its first token
        committed, _, _ = ctsd_select(
            tree_logits,
            tree_token_ids,
            self._path_indices,
            self._path_lengths,
            self._path_first_token_idx,
        )

        # 4. Construct SamplerOutput with committed token
        sampler_output = SamplerOutput(
            sampled_token_ids=torch.tensor(
                [[committed]], device=self.device, dtype=torch.int32
            ),
            logprobs_tensors=None,
        )

        # 5. Bookkeeping synchronization
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

        # 6. Build ModelRunnerOutput
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

    def _ctsd_forward_tree(self, root_hidden, scheduler_output, slot_mappings):
        """
        Orchestrate GPU forward pass for CTSD candidate tree.
        """
        # 1. Compute root logits
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
