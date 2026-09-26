import torch
from vllm_ctsd.config import CTSDConfig
from vllm_ctsd.tree_bias import build_tree_attn_bias


def test_tree_bias_2_1():
    cfg = CTSDConfig(enabled=True, breadth=2, depth=1, temperature=1.0, seed=0)
    topology = cfg.build_topology()
    device = torch.device("cpu")
    bias = build_tree_attn_bias(topology, device, torch.float32)

    assert bias.shape == (3, 3)

    # Everyone attends to root (col 0 is 0.0)
    assert torch.all(bias[:, 0] == 0.0)

    # Diagonal is 0.0 (self-attention)
    for i in range(3):
        assert bias[i, i] == 0.0

    # Row 0: root only attends to itself, others are -inf
    assert bias[0, 1] == -float("inf")
    assert bias[0, 2] == -float("inf")

    # Row 1: attends to root (0) and self (1), not sibling (2)
    assert bias[1, 0] == 0.0
    assert bias[1, 1] == 0.0
    assert bias[1, 2] == -float("inf")

    # Row 2: attends to root (0) and self (2), not sibling (1)
    assert bias[2, 0] == 0.0
    assert bias[2, 1] == -float("inf")
    assert bias[2, 2] == 0.0


def test_tree_bias_2_2():
    cfg = CTSDConfig(enabled=True, breadth=2, depth=2, temperature=1.0, seed=0)
    topology = cfg.build_topology()
    device = torch.device("cpu")
    bias = build_tree_attn_bias(topology, device, torch.float32)

    # tree_size is 6, so total size is 7
    assert bias.shape == (7, 7)

    # Everyone attends to root
    assert torch.all(bias[:, 0] == 0.0)

    # Self-attention
    for i in range(7):
        assert bias[i, i] == 0.0

    # Node 4 (child of node 1, which is child of root 0)
    # Attends to 0 (root), 1 (parent), and 4 (self)
    assert bias[4, 0] == 0.0
    assert bias[4, 1] == 0.0
    assert bias[4, 4] == 0.0
    # Does not attend to node 2, 3, 5, 6
    assert bias[4, 2] == -float("inf")
    assert bias[4, 3] == -float("inf")
    assert bias[4, 5] == -float("inf")
    assert bias[4, 6] == -float("inf")
