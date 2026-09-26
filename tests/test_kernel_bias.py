import numpy as np
import pytest
import torch
from vllm_ctsd.config import CTSDConfig
from vllm_ctsd.tree_bias import build_tree_attn_bias


def compute_reference_attention_bias(
    seq_len: int,
    query_start_abs: int,
    num_queries: int,
    tree_start_pos: int,
    tree_size: int,
    tree_attn_bias: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pure-Python / NumPy reference implementation of the CTSD modified attention bias logic.

    Rules:
      When tree_start_pos == -1 or tree_size == 0:
        Preserve standard attention behavior (0.0 for keys <= query, -inf for future).
      When tree_start_pos >= 0 and tree_size > 0:
        1. Keys before tree_start_pos (prefix tokens) have bias 0.0 (attended to causally).
        2. Keys in [tree_start_pos, tree_start_pos + tree_size) are tree-region keys:
           row_idx = query_abs_pos - tree_start_pos
           col_idx = key_abs_pos - tree_start_pos
           bias = tree_attn_bias[row_idx, col_idx]
        3. Keys ahead of query_abs_pos are masked with -inf (causal masking).
    """
    bias = np.full((num_queries, seq_len), -np.inf, dtype=np.float32)

    for q in range(num_queries):
        query_abs_pos = query_start_abs + q

        for k in range(seq_len):
            key_abs_pos = k

            # Causal mask: key must not be in the future
            if key_abs_pos > query_abs_pos:
                continue

            # Fallback mode (CTSD disabled / standard decode)
            if tree_start_pos < 0 or tree_size == 0:
                bias[q, k] = 0.0
                continue

            # Case 1: Keys in the prefix before the tree region
            if key_abs_pos < tree_start_pos:
                bias[q, k] = 0.0
                continue

            # Case 2: Keys in the tree region [tree_start_pos, tree_start_pos + tree_size)
            if tree_start_pos <= key_abs_pos < (tree_start_pos + tree_size):
                row_idx = query_abs_pos - tree_start_pos
                col_idx = key_abs_pos - tree_start_pos
                if tree_attn_bias is not None and 0 <= row_idx < tree_size and 0 <= col_idx < tree_size:
                    bias[q, k] = tree_attn_bias[row_idx, col_idx]
                else:
                    bias[q, k] = -np.inf
                continue

            # Case 3: In-query keys outside tree region (if any)
            if key_abs_pos >= query_start_abs:
                key_rel_pos = key_abs_pos - query_start_abs
                if tree_attn_bias is not None and 0 <= q < tree_attn_bias.shape[0] and 0 <= key_rel_pos < tree_attn_bias.shape[1]:
                    bias[q, k] = tree_attn_bias[q, key_rel_pos]
                else:
                    bias[q, k] = 0.0

    return bias


def test_reference_kernel_bias_persistent_interior():
    """
    Test against dense attention computation for a (B=2, D=2) tree with a persistent interior.
    Verify that Leaf 3 does not attend to Leaf 1's interior ancestor.
    """
    # 1. Build topology and tree attention bias for (B=2, D=2)
    # tree_size = 2 + 4 = 6 non-root nodes
    # Node 0, 1: depth 1 interior nodes
    # Node 2, 3: children of Node 0 (Leaf 0, Leaf 1)
    # Node 4, 5: children of Node 1 (Leaf 2, Leaf 3)
    cfg = CTSDConfig(enabled=True, breadth=2, depth=2, temperature=1.0, seed=0)
    topo = cfg.build_topology()
    full_bias_t = build_tree_attn_bias(topo, device=torch.device("cpu"), dtype=torch.float32)
    # Exclude root at index 0: [tree_size, tree_size]
    tree_bias_np = full_bias_t[1:, 1:].numpy()

    # 2. Setup sequence layout
    # Prefix length = 4 (positions 0, 1, 2, 3)
    # Tree starts at position 4
    # Interior nodes occupy positions 4, 5 (Node 0, Node 1)
    # New leaves are forwarded as queries at positions 6, 7, 8, 9 (Nodes 2, 3, 4, 5)
    prefix_len = 4
    tree_start_pos = 4
    tree_size = 6
    num_queries = 4       # The 4 new leaves
    query_start_abs = 6   # Queries start at position 6
    seq_len = 10          # Total sequence length: 4 prefix + 2 interior + 4 leaves

    bias = compute_reference_attention_bias(
        seq_len=seq_len,
        query_start_abs=query_start_abs,
        num_queries=num_queries,
        tree_start_pos=tree_start_pos,
        tree_size=tree_size,
        tree_attn_bias=tree_bias_np,
    )

    # Key indices:
    # 0..3: prefix tokens
    # 4: Interior Node 0 (ancestor of Leaf 0 & Leaf 1)
    # 5: Interior Node 1 (ancestor of Leaf 2 & Leaf 3)
    # 6: Leaf 0 (Node 2)
    # 7: Leaf 1 (Node 3)
    # 8: Leaf 2 (Node 4)
    # 9: Leaf 3 (Node 5)
    interior_node_0_pos = tree_start_pos + 0  # pos 4
    interior_node_1_pos = tree_start_pos + 1  # pos 5
    leaf_3_q_idx = 3                          # 4th query (Node 5)
    leaf_1_q_idx = 1                          # 2nd query (Node 3)

    # Verification 1: Leaf 3 MUST NOT attend to Leaf 1's interior ancestor (Node 0)
    assert np.isneginf(bias[leaf_3_q_idx, interior_node_0_pos]), (
        f"Expected -inf bias from Leaf 3 to Node 0, got {bias[leaf_3_q_idx, interior_node_0_pos]}"
    )

    # Verification 2: Leaf 3 MUST attend to its own interior ancestor (Node 1)
    assert bias[leaf_3_q_idx, interior_node_1_pos] == 0.0, (
        f"Expected 0.0 bias from Leaf 3 to its ancestor Node 1, got {bias[leaf_3_q_idx, interior_node_1_pos]}"
    )

    # Verification 3: Leaf 3 MUST attend to all prefix tokens
    for p in range(prefix_len):
        assert bias[leaf_3_q_idx, p] == 0.0, (
            f"Expected 0.0 bias from Leaf 3 to prefix token {p}, got {bias[leaf_3_q_idx, p]}"
        )

    # Verification 4: Leaf 3 only attends to itself among the leaves
    for l_pos in range(query_start_abs, seq_len):
        if l_pos == query_start_abs + leaf_3_q_idx:
            assert bias[leaf_3_q_idx, l_pos] == 0.0
        else:
            assert np.isneginf(bias[leaf_3_q_idx, l_pos])

    # 3. Dense attention computation
    # Run full dense QK^T attention and verify softmax probabilities
    d_k = 16
    np.random.seed(42)
    Q = np.random.randn(num_queries, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)

    scores = (Q @ K.T) / np.sqrt(d_k) + bias
    # Softmax along key dimension
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Verify probability distribution:
    # Leaf 3's attention weight on Leaf 1's ancestor MUST be EXACTLY 0.0
    assert attn_weights[leaf_3_q_idx, interior_node_0_pos] == 0.0
    # Leaf 3's attention weight on its own ancestor MUST be > 0.0
    assert attn_weights[leaf_3_q_idx, interior_node_1_pos] > 0.0
    # Leaf 3's attention weight on prefix tokens MUST be > 0.0
    for p in range(prefix_len):
        assert attn_weights[leaf_3_q_idx, p] > 0.0


def test_kernel_bias_tree_size_zero_fallback():
    """
    Verify that when tree_size == 0 or tree_start_pos == -1,
    the bias logic behaves identically to standard attention.
    """
    tree_size = 0
    tree_start_pos = -1
    seq_len = 6
    num_queries = 2
    query_start_abs = 4

    bias = compute_reference_attention_bias(
        seq_len=seq_len,
        query_start_abs=query_start_abs,
        num_queries=num_queries,
        tree_start_pos=tree_start_pos,
        tree_size=tree_size,
    )

    # For standard causal attention, all keys <= query_abs_pos should be 0.0
    for q in range(num_queries):
        q_abs = query_start_abs + q
        for k in range(seq_len):
            if k <= q_abs:
                assert bias[q, k] == 0.0
            else:
                assert np.isneginf(bias[q, k])
