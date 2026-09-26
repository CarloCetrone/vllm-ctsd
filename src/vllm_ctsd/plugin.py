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

    # 1. Force attention backend to TREE_ATTN via AttentionConfig.__init__
    try:
        from vllm.config import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        _orig_ac_init = AttentionConfig.__init__

        def _patched_ac_init(self, *args, **kwargs):
            _orig_ac_init(self, *args, **kwargs)
            if getattr(self, "backend", None) is None:
                self.backend = AttentionBackendEnum.TREE_ATTN
                logger.info("vllm-ctsd: forced attention backend to TREE_ATTN")

        AttentionConfig.__init__ = _patched_ac_init
    except Exception:
        logger.exception("vllm-ctsd: failed to patch AttentionConfig.__init__")

    # 2. Also patch EngineArgs.__post_init__ to guarantee TREE_ATTN sticks
    try:
        import vllm.engine.arg_utils as au
        from vllm.config import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        _orig_post_init = au.EngineArgs.__post_init__

        def _patched_post_init(self, *args, **kwargs):
            _orig_post_init(self, *args, **kwargs)
            if getattr(self, "attention_config", None) is None:
                self.attention_config = AttentionConfig()
            if self.attention_config.backend is None:
                self.attention_config.backend = AttentionBackendEnum.TREE_ATTN
                logger.info("vllm-ctsd: forced EngineArgs.attention_config.backend to TREE_ATTN")
            if getattr(self, "attention_backend", None) is None:
                self.attention_backend = AttentionBackendEnum.TREE_ATTN

        au.EngineArgs.__post_init__ = _patched_post_init
    except Exception:
        logger.exception("vllm-ctsd: failed to patch EngineArgs.__post_init__")

    # 3. Register vendored CTSD attention custom ops
    try:
        from vllm_ctsd.kernels import register_ctsd_attention_ops
        register_ctsd_attention_ops()
    except Exception:
        logger.exception("vllm-ctsd: failed to register CTSD attention ops")

    # 4. Patch TreeAttention backend to use CTSDTreeAttentionImpl
    try:
        import vllm.v1.attention.backends.tree_attn as tree_attn_module
        from vllm_ctsd.tree_attention import CTSDTreeAttentionImpl
        tree_attn_module.TreeAttentionImpl = CTSDTreeAttentionImpl
        if hasattr(tree_attn_module, "TreeAttentionBackend"):
            tree_attn_module.TreeAttentionBackend.get_impl_cls = staticmethod(
                lambda: CTSDTreeAttentionImpl
            )
    except Exception:
        logger.exception("vllm-ctsd: failed to patch TreeAttentionImpl")

    # 5. Patch Model Runner
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

    # 6. Patch Scheduler lookahead reservation
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
