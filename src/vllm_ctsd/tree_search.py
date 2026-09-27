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
    *args,
    root_logits: torch.Tensor | None = None,
    **kwargs,
) -> tuple[int, int, torch.Tensor]:
    """
    Computes average perplexity for each root-to-leaf path and selects the winning branch.

    ctsd_select takes root_logits separately and computes level-1 logprobs from it,
    then level-k logprobs from tree_logits_per_level[parent_idx].

    Supports:
      ctsd_select(root_logits, tree_logits_per_level, tree_token_ids, paths, lengths, first_tok)
      ctsd_select(tree_logits_per_level, tree_token_ids, paths, lengths, first_tok, root_logits=root_logits)
    """
    if len(args) == 6:
        if args[0].ndim == 1 or (args[0].ndim == 2 and args[0].shape[0] == 1):
            r_logits = args[0]
            t_logits = args[1]
            t_toks = args[2]
            paths = args[3]
            lengths = args[4]
            first_tok = args[5]
        elif root_logits is not None:
            r_logits = root_logits
            t_logits = args[0]
            t_toks = args[1]
            paths = args[2]
            lengths = args[3]
            first_tok = args[4]
        else:
            r_logits = args[0][0]
            t_logits = args[0][1:]
            t_toks = args[1]
            paths = args[2]
            lengths = args[3]
            first_tok = args[4]
    elif len(args) == 5:
        if root_logits is not None:
            r_logits = root_logits
            t_logits = args[0]
            t_toks = args[1]
            paths = args[2]
            lengths = args[3]
            first_tok = args[4]
        else:
            r_logits = args[0][0]
            t_logits = args[0][1:]
            t_toks = args[1]
            paths = args[2]
            lengths = args[3]
            first_tok = args[4]
    else:
        raise ValueError(f"Unexpected number of arguments to ctsd_select: {len(args)}")

    full_logits = torch.cat([r_logits.reshape(1, -1), t_logits], dim=0)
    if t_toks.shape[0] == t_logits.shape[0]:
        full_token_ids = torch.cat(
            [torch.tensor([-1], dtype=torch.int64, device=t_toks.device), t_toks]
        )
    else:
        full_token_ids = t_toks

    log_probs = torch.log_softmax(full_logits.float(), dim=-1)

    num_paths, path_len = paths.shape
    path_lps = torch.zeros(num_paths, device=full_logits.device, dtype=torch.float32)

    for step in range(1, path_len):
        parent_nodes = paths[:, step - 1]
        child_nodes = paths[:, step]
        valid = (child_nodes > 0) & (parent_nodes >= 0)

        safe_parent = parent_nodes.clamp(min=0)
        safe_child = child_nodes.clamp(min=0)
        child_toks = full_token_ids[safe_child].clamp(min=0)

        # logprobs of child tokens under parent distribution
        step_lp = log_probs[safe_parent].gather(1, child_toks.unsqueeze(1)).squeeze(1)
        path_lps += torch.where(valid, step_lp, torch.zeros_like(step_lp))

    valid_counts = (paths > 0).sum(dim=-1).float().clamp(min=1.0)
    avg_logprobs = path_lps / valid_counts

    best_path_idx = torch.argmax(avg_logprobs)
    first_idx = first_tok[best_path_idx]

    committed_token = int(full_token_ids[first_idx].item())
    best_path_int = int(best_path_idx.item())

    import math
    import os
    import sys

    debug = any(
        os.environ.get(k, "0").lower() in ("1", "true", "yes")
        for k in ("CAUTIOUS_DEBUG", "VLLM_CTSD_DEBUG", "VLLM_CAUTIOUS_DEBUG")
    )
    if debug:
        step_num = kwargs.get("step_num")
        step_str = f"Step {step_num:3d}" if step_num is not None else "Step"
        greedy_root = int(torch.argmax(full_logits[0]).item())
        probs = torch.softmax(full_logits[0].float(), dim=-1)
        greedy_prob = float(probs[greedy_root].item())
        committed_prob = float(probs[committed_token].item())

        winning_nodes = paths[best_path_int].tolist()
        winning_tokens = [int(full_token_ids[n].item()) for n in winning_nodes if n > 0]
        best_lp = float(avg_logprobs[best_path_int].item())
        best_ppl = float(math.exp(-best_lp))
        match = (committed_token == greedy_root)

        log_lines = [
            f"[CTSD {step_str}] Committed: {committed_token:6d} (p={committed_prob:.3f}) | "
            f"Greedy: {greedy_root:6d} (p={greedy_prob:.3f}) | "
            f"Winning Path {best_path_int:2d}: {winning_tokens} (PPL: {best_ppl:.4f}) | "
            f"Match: {match}\n"
        ]

        top_k = min(num_paths, 3)
        sorted_indices = torch.argsort(avg_logprobs, descending=True)[:top_k]
        for p_idx in sorted_indices:
            p = int(p_idx.item())
            p_nodes = paths[p].tolist()
            p_toks = [int(full_token_ids[n].item()) for n in p_nodes if n > 0]
            p_lp = float(avg_logprobs[p].item())
            p_ppl = float(math.exp(-p_lp))
            mark = " <-- WINNER" if p == best_path_int else ""
            log_lines.append(f"    Path {p:2d}: tokens={p_toks}, PPL={p_ppl:.4f}, avg_lp={p_lp:.4f}{mark}\n")

        full_log = "".join(log_lines)

        log_file = (
            os.environ.get("CAUTIOUS_LOG_FILE")
            or os.environ.get("VLLM_CTSD_LOG_FILE")
            or ("/tmp/cautious_debug.log" if os.name != "nt" else "cautious_debug.log")
        )
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(full_log)
                f.flush()
        except Exception:
            pass

        try:
            sys.stdout.write(full_log)
            sys.stdout.flush()
            sys.stderr.write(full_log)
            sys.stderr.flush()
        except Exception:
            pass

    return committed_token, best_path_int, avg_logprobs
