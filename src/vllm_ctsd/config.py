from dataclasses import dataclass
import itertools
import os


@dataclass
class CTSDConfig:
    enabled: bool
    breadth: int
    depth: int
    temperature: float
    seed: int

    def __post_init__(self) -> None:
        if self.breadth < 2:
            raise ValueError(f"breadth must be >= 2, got {self.breadth}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

    @classmethod
    def from_env(cls) -> "CTSDConfig":
        raw_enable = os.environ.get("VLLM_CTSD_ENABLE", "0").strip().lower()
        enabled = raw_enable in ("1", "true", "yes", "on")

        raw_breadth = int(os.environ.get("VLLM_CTSD_BREADTH", "4"))
        raw_depth = int(os.environ.get("VLLM_CTSD_DEPTH", "3"))
        raw_temp = float(os.environ.get("VLLM_CTSD_TEMPERATURE", "1.0"))
        raw_seed = int(os.environ.get("VLLM_CTSD_SEED", "0"))

        return cls(
            enabled=enabled,
            breadth=raw_breadth,
            depth=raw_depth,
            temperature=raw_temp,
            seed=raw_seed,
        )

    @property
    def tree_size(self) -> int:
        """Total number of non-root nodes across all levels in the tree."""
        return sum(self.breadth**i for i in range(1, self.depth + 1))

    def build_topology(self) -> dict:
        """Build tree topology metadata returned as plain Python structures."""
        level_counts = [self.breadth**i for i in range(1, self.depth + 1)]
        cu_level_counts = list(itertools.accumulate(level_counts))

        node_depths: list[int] = []
        parent_idx: list[int] = []

        # 1-based level start indices for non-root nodes (where root is node 0)
        level_starts = [1] + [
            1 + cu_level_counts[i] for i in range(len(cu_level_counts) - 1)
        ]

        for lvl in range(self.depth):
            cnt = level_counts[lvl]
            node_depths.extend([lvl + 1] * cnt)
            if lvl == 0:
                parent_idx.extend([0] * cnt)
            else:
                p_start = level_starts[lvl - 1]
                for c in range(cnt):
                    parent_idx.append(p_start + (c // self.breadth))

        paths = list(itertools.product(range(self.breadth), repeat=self.depth))
        path_indices: list[list[int]] = []
        path_first_token_idx: list[int] = []

        for path in paths:
            seq = [0]
            cur_node = 0
            for lvl, choice in enumerate(path):
                l_start = level_starts[lvl]
                if lvl == 0:
                    cur_node = l_start + choice
                else:
                    prev_lvl_start = level_starts[lvl - 1]
                    parent_offset = cur_node - prev_lvl_start
                    cur_node = l_start + (parent_offset * self.breadth) + choice
                seq.append(cur_node)

            path_indices.append(seq)
            path_first_token_idx.append(seq[1])

        path_lengths = [len(seq) for seq in path_indices]

        return {
            "paths": paths,
            "level_counts": level_counts,
            "cu_level_counts": cu_level_counts,
            "node_depths": node_depths,
            "parent_idx": parent_idx,
            "path_indices": path_indices,
            "path_lengths": path_lengths,
            "path_first_token_idx": path_first_token_idx,
            "tree_size": self.tree_size,
        }
