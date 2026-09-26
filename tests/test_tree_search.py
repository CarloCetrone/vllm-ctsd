import torch
from vllm_ctsd.config import CTSDConfig
from vllm_ctsd.tree_search import ctsd_select, expand_level, root_topk


def test_root_topk():
    logits = torch.tensor([1.0, 5.0, 3.0, 0.5, 4.0])
    ids, lps = root_topk(logits, breadth=2)
    assert ids.tolist() == [1, 4]  # indices with values 5.0 and 4.0
    assert lps.shape == (2,)


def test_expand_level():
    logits = torch.tensor(
        [
            [1.0, 5.0, 2.0],
            [4.0, 0.0, 3.0],
        ]
    )
    ids, lps = expand_level(logits, breadth=2)
    assert ids.shape == (2, 2)
    assert ids[0].tolist() == [1, 2]
    assert ids[1].tolist() == [0, 2]


def test_ctsd_select_deterministic():
    cfg = CTSDConfig(enabled=True, breadth=2, depth=2, temperature=1.0, seed=0)
    top = cfg.build_topology()

    paths_tensor = torch.tensor(top["path_indices"], dtype=torch.int64)
    lengths_tensor = torch.tensor(top["path_lengths"], dtype=torch.float32)
    first_tok_tensor = torch.tensor(
        top["path_first_token_idx"], dtype=torch.int64
    )

    V = 10
    tree_size = 6
    # Root logits: level-1 logprobs from root
    root_logits = torch.zeros(V)
    root_logits[5] = 10.0  # High probability for token 5 (node 1)

    # tree_logits_per_level: level-k logprobs from parent_idx
    # Node 1 is at index 0 of tree_logits_per_level. It predicts its children (nodes 3 and 4).
    tree_logits_per_level = torch.zeros(tree_size, V)
    tree_logits_per_level[0, 2] = 10.0  # Node 1 gives high probability to token 2 (node 4)

    token_ids = torch.tensor([5, 8, 3, 2, 7, 4], dtype=torch.int64)

    # Path 1: root -> node 1 (token 5) -> node 4 (token 2)
    committed_token, best_path_idx, avg_logprobs = ctsd_select(
        root_logits,
        tree_logits_per_level,
        token_ids,
        paths_tensor,
        lengths_tensor,
        first_tok_tensor,
    )

    assert best_path_idx == 1
    assert committed_token == 5
    assert avg_logprobs[1] > avg_logprobs[0]
    assert avg_logprobs[1] > avg_logprobs[2]
    assert avg_logprobs[1] > avg_logprobs[3]

