# vLLM Symbol Audit (vLLM 0.17.1)

This audit documents every external vLLM symbol imported or referenced by `vllm-ctsd`, verified against the verified installation/source of **vLLM 0.17.1**.

| Symbol | Import Statement | vLLM 0.17.1 Source Location | Status |
|---|---|---|---|
| `ModelRunnerOutput` | `from vllm.v1.outputs import ModelRunnerOutput` | `vllm/v1/outputs.py:149` | Verified |
| `AttentionConfig` | `from vllm.config import AttentionConfig` | `vllm/config/attention.py:13`, exported in `vllm/config/__init__.py:8` | Verified |
| `CUDAGraphMode` | `from vllm.config import CUDAGraphMode` | `vllm/config/compilation.py:33`, exported in `vllm/config/__init__.py:12` | Verified |
| `AttentionBackendEnum` | `from vllm.v1.attention.backends.registry import AttentionBackendEnum` | `vllm/v1/attention/backends/registry.py:34` | Verified |
| `TreeAttentionMetadata` | `from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata` | `vllm/v1/attention/backends/tree_attn.py:73` | Verified |
| `TreeAttentionImpl` | `from vllm.v1.attention.backends.tree_attn import TreeAttentionImpl` | `vllm/v1/attention/backends/tree_attn.py:289` | Verified |
| `TreeAttentionBackend` | `from vllm.v1.attention.backends.tree_attn import TreeAttentionBackend` | `vllm/v1/attention/backends/tree_attn.py:31` | Verified |
| `SamplerOutput` | `from vllm.v1.sample.sampler import SamplerOutput` | `vllm/v1/sample/sampler.py:27` | Verified |
| `GPUModelRunner` | `from vllm.v1.worker.gpu_model_runner import GPUModelRunner` | `vllm/v1/worker/gpu_model_runner.py:355` | Verified |
| `Scheduler` | `from vllm.v1.core.sched.scheduler import Scheduler` | `vllm/v1/core/sched/scheduler.py:75` | Verified |
| `set_forward_context` | `from vllm.forward_context import set_forward_context` | `vllm/forward_context.py:46` | Verified |
| `get_attention_context` | `from vllm.forward_context import get_attention_context` | `vllm/forward_context.py:56` | Verified |
| `direct_register_custom_op` | `from vllm.utils.torch_utils import direct_register_custom_op` | `vllm/utils/torch_utils.py:792` | Verified |
| `_custom_ops` | `from vllm import _custom_ops as ops` | `vllm/__init__.py:46` | Verified |
| `init_logger` | `from vllm.logger import init_logger` | `vllm/logger.py:40` | Verified |
| `EngineArgs` | `import vllm.engine.arg_utils as au` | `vllm/engine/arg_utils.py:168` | Verified |
| `gpu_worker` | `import vllm.v1.worker.gpu_worker as gw` | `vllm/v1/worker/gpu_worker.py:65` | Verified |

## Verification Commands for Colab / Linux Environment

```bash
python -c "from vllm.v1.outputs import ModelRunnerOutput; print('OK:', ModelRunnerOutput)"
python -c "from vllm.config import AttentionConfig, CUDAGraphMode; print('OK:', AttentionConfig, CUDAGraphMode)"
python -c "from vllm.v1.attention.backends.registry import AttentionBackendEnum; print('OK:', AttentionBackendEnum.TREE_ATTN)"
python -c "from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata, TreeAttentionImpl, TreeAttentionBackend; print('OK:', TreeAttentionMetadata, TreeAttentionImpl, TreeAttentionBackend)"
python -c "from vllm.v1.sample.sampler import SamplerOutput; print('OK:', SamplerOutput)"
python -c "from vllm.v1.worker.gpu_model_runner import GPUModelRunner; print('OK:', GPUModelRunner)"
python -c "from vllm.v1.core.sched.scheduler import Scheduler; print('OK:', Scheduler)"
python -c "from vllm.forward_context import set_forward_context, get_attention_context; print('OK:', set_forward_context, get_attention_context)"
python -c "from vllm.utils.torch_utils import direct_register_custom_op; print('OK:', direct_register_custom_op)"
python -c "import vllm.engine.arg_utils as au; print('OK:', au.EngineArgs)"
```

## Note on Removed Non-Existent Symbols

- **`AsyncGPUModelRunnerOutput`**: Completely removed from all plugin files. CTSD does not support async scheduling; `output` (`ModelRunnerOutput`) is returned directly and unconditionally.
