from dataclasses import dataclass
import torch

try:
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.tree_attn import (
        TreeAttentionImpl,
        TreeAttentionMetadata,
    )
    from vllm_ctsd.kernels.triton_unified_attention_ctsd import (
        unified_attention as unified_attention_ctsd,
    )
except ImportError:
    @dataclass
    class TreeAttentionMetadata:
        num_actual_tokens: int = 0
        max_query_len: int = 0
        query_start_loc: torch.Tensor | None = None
        max_seq_len: int = 0
        seq_lens: torch.Tensor | None = None
        block_table: torch.Tensor | None = None
        slot_mapping: torch.Tensor | None = None
        num_prefill_tokens: int = 0
        num_decode_tokens: int = 0
        num_prefills: int = 0
        num_decodes: int = 0
        tree_attn_bias: torch.Tensor | None = None
        _cached_prefill_metadata: object = None
        _cached_decode_metadata: object = None

        @property
        def prefill_metadata(self):
            return None

        @property
        def decode_metadata(self):
            return None

    TreeAttentionImpl = object  # type: ignore
    unified_attention_ctsd = None  # type: ignore
    ops = None  # type: ignore


@dataclass
class CTSDTreeAttentionMetadata(TreeAttentionMetadata):
    tree_start_pos: int = -1
    tree_size: int = 0

    # Fields expected by FlashAttention / Triton attention backends
    use_cascade: bool = False
    common_prefix_len: int = 0
    cu_prefix_query_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0
    causal: bool = True

    @property
    def decode_metadata(self):
        meta = super().decode_metadata
        if meta is not None:
            setattr(meta, "tree_start_pos", self.tree_start_pos)
            setattr(meta, "tree_size", self.tree_size)
            setattr(meta, "use_cascade", False)
        return meta


class CTSDTreeAttentionImpl(TreeAttentionImpl):
    """
    Subclass of TreeAttentionImpl forwarding tree_start_pos and tree_size
    to the vendored triton unified attention kernel.
    """

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TreeAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for TreeAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # Cache the input KVs.
        key_cache, value_cache = kv_cache.unbind(0)
        k_scale = getattr(layer, "_k_scale", None)
        v_scale = getattr(layer, "_v_scale", None)

        if self.kv_sharing_target_layer_name is None:
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, key.shape[1])
        k_descale = k_scale.expand(descale_shape) if k_scale is not None else None
        v_descale = v_scale.expand(descale_shape) if v_scale is not None else None

        if prefill_meta := attn_metadata.prefill_metadata:
            unified_attention_ctsd(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                seqused_k=prefill_meta.seq_lens,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=prefill_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,
                k_descale=k_descale,
                v_descale=v_descale,
            )

        if decode_meta := attn_metadata.decode_metadata:
            tree_start_pos = getattr(attn_metadata, "tree_start_pos", -1)
            tree_size = getattr(attn_metadata, "tree_size", 0)
            unified_attention_ctsd(
                q=query[:num_decode_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_decode_tokens],
                cu_seqlens_q=decode_meta.query_start_loc,
                max_seqlen_q=decode_meta.max_query_len,
                seqused_k=decode_meta.seq_lens,
                max_seqlen_k=decode_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                qq_bias=decode_meta.tree_attn_bias,
                window_size=self.sliding_window,
                block_table=decode_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,
                k_descale=k_descale,
                v_descale=v_descale,
                tree_start_pos=tree_start_pos,
                tree_size=tree_size,
            )
        return output
