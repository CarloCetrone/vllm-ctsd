"""Run in Colab after `pip install git+...`."""
import os
import time

os.environ["VLLM_CTSD_ENABLE"] = "1"
os.environ["VLLM_CTSD_BREADTH"] = "2"
os.environ["VLLM_CTSD_DEPTH"] = "2"

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.85,
)

sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=30)
prompt = "What is 2+2? Answer briefly."

t0 = time.time()
out = llm.generate([prompt], sp)[0].outputs[0].text
dt = time.time() - t0

print("OUTPUT:", out)
print(f"TIME: {dt:.2f}s")
assert out.strip(), "empty output"
print("SMOKE TEST PASSED")
