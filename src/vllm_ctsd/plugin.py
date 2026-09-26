import logging
import os
import vllm_ctsd
from vllm_ctsd.config import CTSDConfig

logger = logging.getLogger("vllm_ctsd")


def register():
    if vllm_ctsd._REGISTERED:
        return
    cfg = CTSDConfig.from_env()
    if not cfg.enabled:
        logger.info("vllm-ctsd: disabled (VLLM_CTSD_ENABLE not set)")
        vllm_ctsd._REGISTERED = True
        return

    logger.info(
        f"vllm-ctsd: enabling with breadth={cfg.breadth} "
        f"depth={cfg.depth} tree_size={cfg.tree_size}"
    )

    # Patch 1 — model runner
    try:
        import vllm.v1.worker.gpu_model_runner as gmr_module
        from vllm_ctsd.model_runner import CTSDGPUModelRunner

        gmr_module.GPUModelRunner = CTSDGPUModelRunner

        # Also patch the local import binding used inside gpu_worker.py
        import vllm.v1.worker.gpu_worker as gw

        if hasattr(gw, "GPUModelRunner"):
            gw.GPUModelRunner = CTSDGPUModelRunner
        if hasattr(gw, "GPUModelRunnerV1"):
            gw.GPUModelRunnerV1 = CTSDGPUModelRunner
    except Exception:
        logger.exception("vllm-ctsd: failed to patch GPUModelRunner")

    # Patch 2 — scheduler lookahead reservation
    try:
        from vllm.v1.core.sched.scheduler import Scheduler

        _orig_init = Scheduler.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            cfg_now = CTSDConfig.from_env()
            if cfg_now.enabled:
                # Reserve enough KV slots for the tree. +1 for the root
                # that the first forward pass has already written.
                self.num_lookahead_tokens = cfg_now.tree_size + 1
                # Do NOT touch num_spec_tokens. Leaving it at 0 prevents
                # vLLM from entering spec-decode code paths, which are
                # guarded by len(scheduled_spec_decode_tokens) > 0.

        Scheduler.__init__ = _patched_init
    except Exception:
        logger.exception("vllm-ctsd: failed to patch Scheduler")

    vllm_ctsd._REGISTERED = True
