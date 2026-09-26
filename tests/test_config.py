import os
import pytest
from vllm_ctsd.config import CTSDConfig


def test_config_validation():
    # Valid
    cfg = CTSDConfig(enabled=True, breadth=2, depth=2, temperature=1.0, seed=42)
    assert cfg.breadth == 2
    assert cfg.depth == 2

    # Invalid breadth < 2
    with pytest.raises(ValueError, match="breadth must be >= 2"):
        CTSDConfig(enabled=True, breadth=1, depth=2, temperature=1.0, seed=42)

    # Invalid depth < 1
    with pytest.raises(ValueError, match="depth must be >= 1"):
        CTSDConfig(enabled=True, breadth=2, depth=0, temperature=1.0, seed=42)

    # Invalid temperature <= 0
    with pytest.raises(ValueError, match="temperature must be > 0"):
        CTSDConfig(enabled=True, breadth=2, depth=2, temperature=0.0, seed=42)


def test_tree_size():
    cfg2_2 = CTSDConfig(enabled=True, breadth=2, depth=2, temperature=1.0, seed=0)
    # 2^1 + 2^2 = 2 + 4 = 6
    assert cfg2_2.tree_size == 6

    cfg4_3 = CTSDConfig(enabled=True, breadth=4, depth=3, temperature=1.0, seed=0)
    # 4^1 + 4^2 + 4^3 = 4 + 16 + 64 = 84
    assert cfg4_3.tree_size == 84


def test_from_env_defaults(monkeypatch):
    for key in [
        "VLLM_CTSD_ENABLE",
        "VLLM_CTSD_BREADTH",
        "VLLM_CTSD_DEPTH",
        "VLLM_CTSD_TEMPERATURE",
        "VLLM_CTSD_SEED",
    ]:
        monkeypatch.delenv(key, raising=False)

    cfg = CTSDConfig.from_env()
    assert not cfg.enabled
    assert cfg.breadth == 4
    assert cfg.depth == 3
    assert cfg.temperature == 1.0
    assert cfg.seed == 0


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("VLLM_CTSD_ENABLE", "1")
    monkeypatch.setenv("VLLM_CTSD_BREADTH", "2")
    monkeypatch.setenv("VLLM_CTSD_DEPTH", "2")
    monkeypatch.setenv("VLLM_CTSD_TEMPERATURE", "0.7")
    monkeypatch.setenv("VLLM_CTSD_SEED", "123")

    cfg = CTSDConfig.from_env()
    assert cfg.enabled
    assert cfg.breadth == 2
    assert cfg.depth == 2
    assert cfg.temperature == 0.7
    assert cfg.seed == 123


def test_build_topology_2_2():
    cfg = CTSDConfig(enabled=True, breadth=2, depth=2, temperature=1.0, seed=0)
    top = cfg.build_topology()

    assert top["tree_size"] == 6
    assert top["paths"] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert top["level_counts"] == [2, 4]
    assert top["cu_level_counts"] == [2, 6]
    assert top["node_depths"] == [1, 1, 2, 2, 2, 2]
    assert top["parent_idx"] == [0, 0, 1, 1, 2, 2]
    assert top["path_indices"] == [[0, 1, 3], [0, 1, 4], [0, 2, 5], [0, 2, 6]]
    assert top["path_lengths"] == [3, 3, 3, 3]
    assert top["path_first_token_idx"] == [1, 1, 2, 2]
