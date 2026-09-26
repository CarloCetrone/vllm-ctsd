from dataclasses import dataclass, field
import torch
from typing import Optional, List, Tuple


@dataclass
class CTSDRequestState:
    """
    Tracks state of a single CTSD generation request across decoding steps
    for persistent-interior tree search decoding.
    """
    req_id: str
    tree_start_pos: int
    breadth: int
    depth: int
    # Frontier nodes at relative depth (depth - 1), count = breadth ** (depth - 1)
    frontier_tokens: torch.Tensor               # [num_frontier] int64
    frontier_logits: torch.Tensor               # [num_frontier, vocab_size] float32
    frontier_accum_lp: torch.Tensor             # [num_frontier] float32
    # For each frontier node, sequence of tokens and logprobs from depth 1 under committed root
    frontier_token_paths: List[List[int]]       # [num_frontier] paths of length (depth - 1)
    frontier_lp_paths: List[List[float]]        # [num_frontier] logprobs of length (depth - 1)
    interior_tokens: List[int] = field(default_factory=list)


def create_initial_state(
    req_id: str,
    breadth: int,
    depth: int,
    tree_start_pos: int,
    root_logits: torch.Tensor,
    tree_token_ids: torch.Tensor,
    all_node_logits: torch.Tensor,
    winning_path_idx: int,
    paths: List[Tuple[int, ...]],
    cu_level_counts: List[int],
) -> Tuple[int, Optional[CTSDRequestState]]:
    """
    Cold-start step:
    Given full tree forward pass results, commits winning first token and
    constructs the persistent interior CTSDRequestState.
    """
    winning_path = paths[winning_path_idx]
    first_child_choice = winning_path[0]
    committed_token = int(tree_token_ids[1 + first_child_choice].item())

    if depth == 1:
        committed_node_idx = 1 + first_child_choice
        frontier_logits = all_node_logits[committed_node_idx].unsqueeze(0)
        state = CTSDRequestState(
            req_id=req_id,
            tree_start_pos=tree_start_pos + 1,
            breadth=breadth,
            depth=depth,
            frontier_tokens=torch.tensor([committed_token], dtype=torch.int64, device=tree_token_ids.device),
            frontier_logits=frontier_logits,
            frontier_accum_lp=torch.zeros(1, dtype=torch.float32, device=tree_token_ids.device),
            frontier_token_paths=[[committed_token]],
            frontier_lp_paths=[[0.0]],
            interior_tokens=[],
        )
        return committed_token, state

    # Identify surviving paths that start with first_child_choice
    num_frontier = breadth ** (depth - 1)
    surviving_frontier_tokens = []
    surviving_frontier_logits = []
    surviving_frontier_accum_lp = []
    surviving_token_paths = []
    surviving_lp_paths = []

    # In BFS topology, node 1 + first_child_choice is the committed root c*
    c_star_node = 1 + first_child_choice
    
    # We want all nodes at relative depth (depth - 1) under c_star_node
    # For D=2, relative depth 1 = depth 2 in full tree (children of c_star)
    def get_frontier_nodes(parent_node: int, current_rel_depth: int, target_rel_depth: int, cur_toks: List[int], cur_lps: List[float]):
        if current_rel_depth == target_rel_depth:
            tok = int(tree_token_ids[parent_node].item())
            surviving_frontier_tokens.append(tok)
            surviving_frontier_logits.append(all_node_logits[parent_node])
            surviving_frontier_accum_lp.append(sum(cur_lps))
            surviving_token_paths.append(cur_toks)
            surviving_lp_paths.append(cur_lps)
            return

        # Children of parent_node in BFS:
        # Find which level parent_node belongs to
        # Full tree level of parent_node is 1 + current_rel_depth - 1 = current_rel_depth
        lvl = current_rel_depth
        level_start = 1 + cu_level_counts[lvl - 1]
        prev_level_start = 1 if lvl == 1 else 1 + cu_level_counts[lvl - 2]
        parent_offset = parent_node - prev_level_start

        for b in range(breadth):
            child_node = level_start + parent_offset * breadth + b
            child_tok = int(tree_token_ids[child_node].item())
            child_lp = torch.log_softmax(all_node_logits[parent_node].float(), dim=-1)[child_tok].item()
            get_frontier_nodes(
                child_node,
                current_rel_depth + 1,
                target_rel_depth,
                cur_toks + [child_tok],
                cur_lps + [child_lp],
            )

    get_frontier_nodes(c_star_node, 0, depth - 1, [], [])

    frontier_logits_tensor = torch.stack(surviving_frontier_logits[:num_frontier], dim=0)
    frontier_tokens_tensor = torch.tensor(
        surviving_frontier_tokens[:num_frontier], dtype=torch.int64, device=tree_token_ids.device
    )
    frontier_accum_lp_tensor = torch.tensor(
        surviving_frontier_accum_lp[:num_frontier], dtype=torch.float32, device=tree_token_ids.device
    )

    state = CTSDRequestState(
        req_id=req_id,
        tree_start_pos=tree_start_pos + 1,
        breadth=breadth,
        depth=depth,
        frontier_tokens=frontier_tokens_tensor,
        frontier_logits=frontier_logits_tensor,
        frontier_accum_lp=frontier_accum_lp_tensor,
        frontier_token_paths=surviving_token_paths[:num_frontier],
        frontier_lp_paths=surviving_lp_paths[:num_frontier],
        interior_tokens=[committed_token],
    )
    return committed_token, state


def expand_frontier_to_leaves(
    state: CTSDRequestState,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expands frontier nodes into B^D new leaves.
    Returns:
      leaf_token_ids: [num_leaves] int64
      leaf_parent_logprobs: [num_leaves] float32
    """
    breadth = state.breadth
    num_frontier = len(state.frontier_tokens)
    num_leaves = num_frontier * breadth

    logits = state.frontier_logits
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    top_lp, top_ids = torch.topk(logprobs, breadth, dim=-1)

    leaf_token_ids = top_ids.reshape(num_leaves).to(torch.int64)
    leaf_parent_logprobs = top_lp.reshape(num_leaves).to(torch.float32)

    return leaf_token_ids, leaf_parent_logprobs


def steady_state_select_and_reroot(
    state: CTSDRequestState,
    leaf_token_ids: torch.Tensor,
    leaf_logits: torch.Tensor,
    leaf_parent_logprobs: torch.Tensor,
) -> Tuple[int, CTSDRequestState]:
    """
    Scores all paths through the new leaves, commits the winning first token,
    prunes unchosen branches, and constructs the CTSDRequestState for the next step.
    """
    breadth = state.breadth
    depth = state.depth
    num_frontier = len(state.frontier_tokens)
    num_leaves = num_frontier * breadth

    # Compute path log-probs:
    # Each leaf i has parent index i // breadth in frontier
    parent_indices = torch.arange(num_leaves, device=leaf_token_ids.device) // breadth
    accum_parent_lp = state.frontier_accum_lp[parent_indices]
    path_total_lp = accum_parent_lp + leaf_parent_logprobs
    path_avg_lp = path_total_lp / float(depth)

    best_leaf_idx = int(torch.argmax(path_avg_lp).item())

    if depth == 1:
        committed_token = int(leaf_token_ids[best_leaf_idx].item())
        new_state = CTSDRequestState(
            req_id=state.req_id,
            tree_start_pos=state.tree_start_pos + 1,
            breadth=breadth,
            depth=depth,
            frontier_tokens=torch.tensor([committed_token], dtype=torch.int64, device=leaf_token_ids.device),
            frontier_logits=leaf_logits[best_leaf_idx].unsqueeze(0),
            frontier_accum_lp=torch.zeros(1, dtype=torch.float32, device=leaf_token_ids.device),
            frontier_token_paths=[[committed_token]],
            frontier_lp_paths=[[0.0]],
            interior_tokens=[],
        )
        return committed_token, new_state

    # When depth >= 2:
    best_frontier_idx = best_leaf_idx // breadth
    winning_token_seq = state.frontier_token_paths[best_frontier_idx]
    committed_token = winning_token_seq[0]

    # Find all leaves that share winning_token_seq[0]
    surviving_leaf_indices = [
        l_idx for l_idx in range(num_leaves)
        if state.frontier_token_paths[l_idx // breadth][0] == committed_token
    ]

    new_num_frontier = breadth ** (depth - 1)
    new_leaf_indices = surviving_leaf_indices[:new_num_frontier]

    new_frontier_tokens = leaf_token_ids[new_leaf_indices]
    new_frontier_logits = leaf_logits[new_leaf_indices]

    # Update path sequences and logprobs: drop committed first token, append new leaf
    new_token_paths = []
    new_lp_paths = []
    new_accum_lps = []

    for l_idx in new_leaf_indices:
        p_idx = l_idx // breadth
        parent_seq = state.frontier_token_paths[p_idx]
        parent_lps = state.frontier_lp_paths[p_idx]

        leaf_tok = int(leaf_token_ids[l_idx].item())
        leaf_lp = float(leaf_parent_logprobs[l_idx].item())

        updated_seq = parent_seq[1:] + [leaf_tok]
        updated_lps = parent_lps[1:] + [leaf_lp]

        new_token_paths.append(updated_seq)
        new_lp_paths.append(updated_lps)
        new_accum_lps.append(sum(updated_lps))

    new_frontier_accum_lp = torch.tensor(
        new_accum_lps, dtype=torch.float32, device=leaf_token_ids.device
    )

    new_state = CTSDRequestState(
        req_id=state.req_id,
        tree_start_pos=state.tree_start_pos + 1,
        breadth=breadth,
        depth=depth,
        frontier_tokens=new_frontier_tokens,
        frontier_logits=new_frontier_logits,
        frontier_accum_lp=new_frontier_accum_lp,
        frontier_token_paths=new_token_paths,
        frontier_lp_paths=new_lp_paths,
        interior_tokens=[committed_token],
    )
    return committed_token, new_state
