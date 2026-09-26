# vllm-ctsd

**Cautious Tree Search Decoding (CTSD)** is a decoding plugin for vLLM that replaces standard greedy or single-token sampling with an on-GPU candidate tree search. At each decoding step, CTSD samples `BREADTH` candidate tokens from the current node's distribution, expands each branch up to `DEPTH` levels, and runs a batched forward pass over the entire tree that shares the KV cache of the common prefix. It computes the average perplexity (normalized negative log-likelihood) for every root-to-leaf path on GPU, commits the first token of the lowest-perplexity path, and prunes all remaining branches to continue generation from the newly committed root.

## Installation

Install directly from GitHub via pip:

```bash
pip install git+https://github.com/USER/vllm-ctsd.git
```

For editable local development:

```bash
git clone https://github.com/USER/vllm-ctsd.git
cd vllm-ctsd
pip install -e .
```

## Requirements

- **vLLM**: `vllm==0.17.1`
- **PyTorch**: `>= 2.0`
- **Prefix Caching**: Must be enabled in vLLM (`enable_prefix_caching=True`, which is the default in vLLM 0.17.1).

## Usage

Set `VLLM_CTSD_ENABLE=1` and any desired configuration environment variables before importing or starting vLLM:

```python
import os
os.environ["VLLM_CTSD_ENABLE"] = "1"
os.environ["VLLM_CTSD_BREADTH"] = "2"
os.environ["VLLM_CTSD_DEPTH"] = "2"

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    enable_prefix_caching=True,
)

params = SamplingParams(max_tokens=64)
outputs = llm.generate(["Explain quantum computing in one sentence."], params)
print(outputs[0].outputs[0].text)
```

## Configuration Reference

| Environment Variable | Description | Default | Valid Range |
|---|---|---|---|
| `VLLM_CTSD_ENABLE` | Activates CTSD tree decoding plugin | `0` (disabled) | `0`, `1`, `true`, `false` |
| `VLLM_CTSD_BREADTH` | Number of child branches sampled per frontier node | `4` | `>= 2` |
| `VLLM_CTSD_DEPTH` | Maximum tree search lookahead depth (excluding root) | `3` | `>= 1` |
| `VLLM_CTSD_TEMPERATURE` | Softmax temperature for tree candidate sampling | `1.0` | `> 0.0` |
| `VLLM_CTSD_SEED` | RNG seed for stochastic selection | `0` | Any integer |

## Architecture & How It Works

1. **Plugin Bootstrap**: The package registers an entry point under `vllm.general_plugins`. At vLLM initialization, `register()` is executed across all engine and worker processes.
2. **GPU Model Runner Interception**: Replaces `GPUModelRunner` with `CTSDGPUModelRunner`, which overrides `sample_tokens` to perform tree expansion, evaluation, and path scoring on GPU.
3. **KV Cache Lookahead Reservation**: Monkey-patches `Scheduler.__init__` to set `self.num_lookahead_tokens = tree_size + 1`, reserving slots in the paged KV cache for intermediate tree nodes.
4. **Pruning & Re-rooting**: When the winning branch's first token is committed, vLLM's built-in scheduler rollback (`request.num_computed_tokens -= num_rejected`) automatically frees/overwrites the unchosen tree branches in subsequent steps without memory movement.
