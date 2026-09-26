import torch


def build_tree_attn_bias(
    topology: dict,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build tree attention bias tensor of shape [tree_size + 1, tree_size + 1].

    Entry (i, j) is:
      - 0.0 if j == 0 (root: everyone attends to root)
      - 0.0 if i == j (self-attention)
      - 0.0 if j is an ancestor of i
      - -inf otherwise
    """
    parent_idx = topology["parent_idx"]
    tree_size = len(parent_idx)
    total_size = tree_size + 1

    bias = torch.full(
        (total_size, total_size), -float("inf"), device=device, dtype=dtype
    )

    # 1. Root attention: everyone attends to root (col 0)
    bias[:, 0] = 0.0

    # 2. Self-attention: diagonal is zero
    for i in range(total_size):
        bias[i, i] = 0.0

    # 3. Ancestor attention: follow parent pointers up to root
    for i in range(1, total_size):
        curr = i
        while curr > 0:
            p = parent_idx[curr - 1]
            bias[i, p] = 0.0
            curr = p

    return bias
