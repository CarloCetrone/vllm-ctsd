import pytest

vllm = pytest.importorskip("vllm")


def test_plugin_imports():
    import vllm_ctsd.plugin
    import vllm_ctsd.model_runner
    import vllm_ctsd.tree_attention
    import vllm_ctsd.tree_state


def test_no_hallucinated_outputs():
    from vllm.v1.outputs import ModelRunnerOutput
    assert ModelRunnerOutput is not None
