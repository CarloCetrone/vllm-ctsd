import torch
import pytest
from vllm_ctsd.config import CTSDConfig
from vllm_ctsd.tree_state import (
    CTSDRequestState,
    create_initial_state,
    expand_frontier_to_leaves,
    steady_state_select_and_reroot,
)


def test_tree_state_machine_cold_start_and_reroot_2_2():
    """
    Test CPU state machine for B=2, D=2:
      1. Cold-start expansion, commit, and state creation.
      2. Steady-state expand, select, prune, and re-root.
    """
    breadth = 2
    depth = 2
    cfg = CTSDConfig(enabled=True, breadth=breadth, depth=depth, temperature=1.0, seed=0)
    topo = cfg.build_topology()
    vocab_size = 50

    # 1. Simulate cold start
    # tree_size = 2 + 4 = 6
    torch.manual_seed(100)
    root_logits = torch.randn(vocab_size)
    tree_token_ids = torch.tensor([-1, 10, 20, 11, 12, 21, 22], dtype=torch.int64)
    all_node_logits = torch.randn(7, vocab_size)

    # Let's say path 0 wins (0, 0): root -> node 1 (tok 10) -> node 3 (tok 101)
    winning_path_idx = 0
    tree_start_pos = 10

    committed_1, state = create_initial_state(
        req_id="req-1",
        breadth=breadth,
        depth=depth,
        tree_start_pos=tree_start_pos,
        root_logits=root_logits,
        tree_token_ids=tree_token_ids,
        all_node_logits=all_node_logits,
        winning_path_idx=winning_path_idx,
        paths=topo["paths"],
        cu_level_counts=topo["cu_level_counts"],
    )

    # First token committed should be 10 (node 1)
    assert committed_1 == 10
    assert state.tree_start_pos == tree_start_pos + 1  # 11
    # For D=2, num_frontier = B^(D-1) = 2^1 = 2
    assert len(state.frontier_tokens) == 2
    assert state.frontier_logits.shape == (2, vocab_size)

    # 2. Steady state: expand frontier to leaves
    leaf_ids, leaf_lps = expand_frontier_to_leaves(state)
    # B^D = 2^2 = 4 new leaves
    assert len(leaf_ids) == 4
    assert len(leaf_lps) == 4

    # 3. Simulate forward pass logits for the 4 leaves
    leaf_logits = torch.randn(4, vocab_size)
    # Make leaf 1 have extremely high probability to force it to win
    leaf_logits[1, :] = 10.0

    committed_2, state_2 = steady_state_select_and_reroot(
        state=state,
        leaf_token_ids=leaf_ids,
        leaf_logits=leaf_logits,
        leaf_parent_logprobs=leaf_lps,
    )

    # Tree start pos must advance by 1
    assert state_2.tree_start_pos == state.tree_start_pos + 1
    # Frontier size must remain B^(D-1) = 2
    assert len(state_2.frontier_tokens) == 2
    assert state_2.frontier_logits.shape == (2, vocab_size)


def test_tree_state_multi_step_loop():
    """
    Test continuous multi-step decoding on CPU across 10 consecutive steps:
      - Cold start
      - 10 steady-state steps
      - Verify state invariants (tree_start_pos advancement, frontier size consistency).
    """
    breadth = 3
    depth = 2
    cfg = CTSDConfig(enabled=True, breadth=breadth, depth=depth, temperature=1.0, seed=0)
    topo = cfg.build_topology()
    vocab_size = 100

    tree_size = cfg.tree_size  # 3 + 9 = 12
    root_logits = torch.randn(vocab_size)
    tree_token_ids = torch.arange(tree_size + 1, dtype=torch.int64)
    tree_token_ids[0] = -1
    all_node_logits = torch.randn(tree_size + 1, vocab_size)

    tree_start_pos = 100
    committed, state = create_initial_state(
        req_id="req-loop",
        breadth=breadth,
        depth=depth,
        tree_start_pos=tree_start_pos,
        root_logits=root_logits,
        tree_token_ids=tree_token_ids,
        all_node_logits=all_node_logits,
        winning_path_idx=0,
        paths=topo["paths"],
        cu_level_counts=topo["cu_level_counts"],
    )

    assert committed == int(tree_token_ids[1].item())
    expected_pos = tree_start_pos + 1
    assert state.tree_start_pos == expected_pos

    num_steady_steps = 10
    expected_frontier_size = breadth ** (depth - 1)  # 3^1 = 3
    expected_num_leaves = breadth ** depth           # 3^2 = 9

    for step in range(num_steady_steps):
        leaf_ids, leaf_lps = expand_frontier_to_leaves(state)
        assert len(leaf_ids) == expected_num_leaves

        leaf_logits = torch.randn(expected_num_leaves, vocab_size)
        committed_step, state = steady_state_select_and_reroot(
            state=state,
            leaf_token_ids=leaf_ids,
            leaf_logits=leaf_logits,
            leaf_parent_logprobs=leaf_lps,
        )

        expected_pos += 1
        assert state.tree_start_pos == expected_pos
        assert len(state.frontier_tokens) == expected_frontier_size
        assert state.frontier_logits.shape == (expected_frontier_size, vocab_size)


def test_tree_state_depth_1():
    """
    Test boundary case D=1 on CPU.
    """
    breadth = 2
    depth = 1
    cfg = CTSDConfig(enabled=True, breadth=breadth, depth=depth, temperature=1.0, seed=0)
    topo = cfg.build_topology()
    vocab_size = 30

    root_logits = torch.randn(vocab_size)
    tree_token_ids = torch.tensor([-1, 42, 43], dtype=torch.int64)
    all_node_logits = torch.randn(3, vocab_size)

    committed, state = create_initial_state(
        req_id="req-d1",
        breadth=breadth,
        depth=depth,
        tree_start_pos=5,
        root_logits=root_logits,
        tree_token_ids=tree_token_ids,
        all_node_logits=all_node_logits,
        winning_path_idx=1,
        paths=topo["paths"],
        cu_level_counts=topo["cu_level_counts"],
    )

    assert committed == 43
    assert state.tree_start_pos == 6
    assert len(state.frontier_tokens) == 1

    leaf_ids, leaf_lps = expand_frontier_to_leaves(state)
    assert len(leaf_ids) == 2

    leaf_logits = torch.randn(2, vocab_size)
    committed_2, state_2 = steady_state_select_and_reroot(
        state=state,
        leaf_token_ids=leaf_ids,
        leaf_logits=leaf_logits,
        leaf_parent_logprobs=leaf_lps,
    )
    assert state_2.tree_start_pos == 7
    assert len(state_2.frontier_tokens) == 1
