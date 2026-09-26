import torch


def root_topk(
    root_logits: torch.Tensor, breadth: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given root_logits of shape [vocab_size] or [1, vocab_size],
    return top-k candidate token ids and logprobs.
    """
    logprobs = torch.log_softmax(root_logits.float(), dim=-1)
    top_lp, top_ids = torch.topk(logprobs, breadth, dim=-1)
    return top_ids.squeeze(0) if top_ids.dim() > 1 else top_ids, (
        top_lp.squeeze(0) if top_lp.dim() > 1 else top_lp
    )


def expand_level(
    parent_logits: torch.Tensor, breadth: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given parent_logits of shape [num_parents, vocab_size],
    return top-k candidate token ids and logprobs per parent:
      ids: [num_parents, breadth]
      logprobs: [num_parents, breadth]
    """
    logprobs = torch.log_softmax(parent_logits.float(), dim=-1)
    top_lp, top_ids = torch.topk(logprobs, breadth, dim=-1)
    return top_ids, top_lp


def ctsd_select(
    tree_logits: torch.Tensor,
    tree_token_ids: torch.Tensor,
    path_indices_tensor: torch.Tensor,
    path_lengths_tensor: torch.Tensor,
    path_first_token_idx_tensor: torch.Tensor,
) -> tuple[int, int, torch.Tensor]:
    """
    Computes average perplexity for each root-to-leaf path and selects the winning branch.

    Args:
      tree_logits: [tree_size + 1, vocab_size] float32. Row i contains the logits
                   distribution that generated node i (row 0 is root or dummy).
      tree_token_ids: [tree_size + 1] int64. Entry 0 is root placeholder (-1),
                      entries 1..tree_size are token ids for non-root nodes.
      path_indices_tensor: [num_paths, max_path_len] int64 with node indices, or -1 for padding.
      path_lengths_tensor: [num_paths] float32.
      path_first_token_idx_tensor: [num_paths] int64.

    Returns:
      committed_token: (int) the token id at position 1 of the best path.
      best_path_idx: (int) index of the best path.
      avg_logprobs: [num_paths] float32.
    """
    log_probs = torch.log_softmax(tree_logits.float(), dim=-1)

    # Gather log-prob of each token under its generating distribution
    clamped_ids = tree_token_ids.clamp(min=0).unsqueeze(1)
    node_lp = log_probs.gather(1, clamped_ids).squeeze(1)

    # Gather along paths, masking out root (index 0) and padding (< 0)
    valid_mask = path_indices_tensor > 0
    safe_indices = path_indices_tensor.clamp(min=0)
    gathered_lp = node_lp[safe_indices]
    path_lp = torch.where(valid_mask, gathered_lp, torch.zeros_like(gathered_lp))

    # Divide by number of generated tokens along the path
    valid_counts = valid_mask.sum(dim=-1).float().clamp(min=1.0)
    avg_logprobs = path_lp.sum(dim=-1) / valid_counts

    best_path_idx = torch.argmax(avg_logprobs)
    first_idx = path_first_token_idx_tensor[best_path_idx]

    # The single .item() allowed at final commit
    committed_token = int(tree_token_ids[first_idx].item())
    best_path_int = int(best_path_idx.item())

    return committed_token, best_path_int, avg_logprobs
