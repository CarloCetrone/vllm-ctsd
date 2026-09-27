import pytest
import torch
from unittest.mock import MagicMock

from vllm_ctsd.tree_attention import CTSDTreeAttentionMetadata, CTSDTreeAttentionImpl


def test_tree_attention_metadata_attributes():
    meta = CTSDTreeAttentionMetadata(
        num_actual_tokens=4,
        max_query_len=4,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        max_seq_len=10,
        seq_lens=torch.tensor([10], dtype=torch.int32),
        block_table=torch.zeros((1, 16), dtype=torch.int32),
        slot_mapping=torch.zeros(4, dtype=torch.int64),
        num_prefill_tokens=0,
        num_decode_tokens=4,
        num_prefills=0,
        num_decodes=1,
        tree_start_pos=6,
        tree_size=4,
    )
    # Ensure cascade and common backend attributes exist and don't raise AttributeError
    assert hasattr(meta, "use_cascade")
    assert meta.use_cascade is False
    assert hasattr(meta, "causal")
    assert meta.causal is True
    assert meta.tree_start_pos == 6
    assert meta.tree_size == 4


def test_attention_impl_forward_routing():
    class DummyImpl:
        def __init__(self):
            self.num_heads = 8
            self.head_size = 64
            self.scale = 0.125
            self.num_kv_heads = 8
            self.alibi_slopes = None
            self.kv_cache_dtype = "auto"
            self.logits_soft_cap = None
            self.attn_type = "decoder"
            self.kv_sharing_target_layer_name = None
            self.sliding_window = (-1, -1)
            self.called_with = None

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, **kwargs):
            self.called_with = "original"
            return "original_output"

    impl = DummyImpl()
    layer = MagicMock()
    layer.sliding_window = None

    # Wrap forward as done in plugin / model runner
    orig_fwd = impl.forward
    tree_mock = MagicMock()
    tree_mock.forward.return_value = "tree_output"
    layer._ctsd_tree_impl = tree_mock

    def wrapped(self_impl, attn_layer, query, key, value, kv_cache, attn_metadata, **kwargs):
        if isinstance(attn_metadata, CTSDTreeAttentionMetadata):
            return attn_layer._ctsd_tree_impl.forward(
                attn_layer, query, key, value, kv_cache, attn_metadata, **kwargs
            )
        return orig_fwd(attn_layer, query, key, value, kv_cache, attn_metadata, **kwargs)

    meta_tree = CTSDTreeAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        max_seq_len=5,
        seq_lens=torch.tensor([5], dtype=torch.int32),
        block_table=torch.zeros((1, 8), dtype=torch.int32),
        slot_mapping=torch.zeros(2, dtype=torch.int64),
    )

    # 1. Routing with CTSDTreeAttentionMetadata
    out = wrapped(impl, layer, None, None, None, None, meta_tree)
    assert out == "tree_output"
    tree_mock.forward.assert_called_once()

    # 2. Routing with standard metadata
    standard_meta = MagicMock()
    out2 = wrapped(impl, layer, None, None, None, None, standard_meta)
    assert out2 == "original_output"
    assert impl.called_with == "original"


def test_ctsd_block_table_scratch_isolation():
    """
    Verify that _get_ctsd_block_table isolates tree nodes into scratch blocks
    and never leaves unallocated blocks pointing to physical Block 0.
    """
    from vllm_ctsd.model_runner import CTSDGPUModelRunner

    runner = object.__new__(CTSDGPUModelRunner)
    runner._ctsd_tree_size = 14  # breadth=2, depth=3 -> 14 nodes
    block_size = 16

    # Mock attn_groups for block_size
    mock_spec = MagicMock()
    mock_spec.block_size = block_size
    mock_group = MagicMock()
    mock_group.kv_cache_spec = mock_spec
    runner.attn_groups = [[mock_group]]

    # Mock kv_caches: 100 total cache blocks
    mock_cache = torch.zeros((2, 100, block_size, 2, 64))
    runner.kv_caches = [mock_cache]

    # Prompt: 25 tokens. 2 blocks allocated by vLLM (physical block 5 and 8).
    # Unallocated block indices are 0.
    raw_block_table = torch.tensor([5, 8, 0, 0, 0, 0], dtype=torch.int32)
    runner._get_block_table_tensor = MagicMock(return_value=raw_block_table)

    tree_start_pos = 26
    # Prefix spans tokens 0..25 -> num_prefix_blocks = 2
    # Tree spans up to 26 + 14 = 40 tokens -> num_total_blocks = 3
    # Block index 2 must be remapped to scratch block (100 - 1 = 99)
    ctsd_bt = runner._get_ctsd_block_table(tree_start_pos)

    assert ctsd_bt[0].item() == 5  # Prefix block 0 preserved
    assert ctsd_bt[1].item() == 8  # Prefix block 1 preserved
    assert ctsd_bt[2].item() == 99  # Tree candidate block mapped to scratch, NOT 0
    assert ctsd_bt[2].item() != 0  # CRITICAL: must never overwrite prompt block 0

