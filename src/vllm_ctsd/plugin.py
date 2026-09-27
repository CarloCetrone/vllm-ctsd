import logging
import os
import vllm_ctsd
from vllm_ctsd.config import CTSDConfig

logger = logging.getLogger("vllm_ctsd")


def register():
    cfg = CTSDConfig.from_env()
    if not cfg.enabled:
        logger.info("vllm-ctsd: disabled (VLLM_CTSD_ENABLE not set)")
        vllm_ctsd._REGISTERED = True
        return

    if getattr(vllm_ctsd, "_ACTIVATED", False):
        return

    logger.info(
        f"vllm-ctsd: enabling with breadth={cfg.breadth} "
        f"depth={cfg.depth} tree_size={cfg.tree_size}"
    )

    # 0. Patch suppress_stdout to avoid UnsupportedOperation: fileno in Jupyter/Colab
    try:
        import sys
        from contextlib import contextmanager
        import vllm.utils.system_utils as su

        _orig_suppress = su.suppress_stdout

        @contextmanager
        def _safe_suppress_stdout():
            try:
                sys.stdout.fileno()
            except Exception:
                yield
                return
            with _orig_suppress():
                yield

        su.suppress_stdout = _safe_suppress_stdout
        try:
            import vllm.distributed.parallel_state as ps
            ps.suppress_stdout = _safe_suppress_stdout
        except Exception:
            pass
        try:
            import vllm.distributed.utils as du
            du.suppress_stdout = _safe_suppress_stdout
        except Exception:
            pass
    except Exception:
        pass

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
            # NOTE: attention_backend is intentionally NOT set. It is mutually
            # exclusive with attention_config.backend in vLLM arg_utils.py:1806-1810.

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

    # 4b. Intercept standard attention backends (FlashAttention, Triton, Flashinfer)
    # to seamlessly route CTSDTreeAttentionMetadata to CTSDTreeAttentionImpl
    try:
        from vllm_ctsd.tree_attention import CTSDTreeAttentionMetadata, CTSDTreeAttentionImpl
        from vllm.v1.attention.backend import AttentionType

        def _wrap_attention_forward(orig_forward):
            def _ctsd_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=None,
                output_scale=None,
                output_block_scale=None,
                **kwargs,
            ):
                if isinstance(attn_metadata, CTSDTreeAttentionMetadata):
                    tree_impl = getattr(layer, "_ctsd_tree_impl", None)
                    if tree_impl is None:
                        tree_impl = CTSDTreeAttentionImpl(
                            num_heads=self.num_heads,
                            head_size=self.head_size,
                            scale=self.scale,
                            num_kv_heads=self.num_kv_heads,
                            alibi_slopes=getattr(self, "alibi_slopes", None),
                            sliding_window=getattr(layer, "sliding_window", None),
                            kv_cache_dtype=self.kv_cache_dtype,
                            logits_soft_cap=getattr(self, "logits_soft_cap", None),
                            attn_type=getattr(self, "attn_type", AttentionType.DECODER),
                            kv_sharing_target_layer_name=getattr(self, "kv_sharing_target_layer_name", None),
                        )
                        tree_impl.sliding_window = getattr(self, "sliding_window", (-1, -1))
                        layer._ctsd_tree_impl = tree_impl
                    return tree_impl.forward(
                        layer=layer,
                        query=query,
                        key=key,
                        value=value,
                        kv_cache=kv_cache,
                        attn_metadata=attn_metadata,
                        output=output,
                        output_scale=output_scale,
                        output_block_scale=output_block_scale,
                        **kwargs,
                    )
                return orig_forward(
                    self,
                    layer=layer,
                    query=query,
                    key=key,
                    value=value,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    output=output,
                    output_scale=output_scale,
                    output_block_scale=output_block_scale,
                    **kwargs,
                )
            return _ctsd_forward

        try:
            from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
            if not getattr(FlashAttentionImpl, "_ctsd_patched", False):
                FlashAttentionImpl.forward = _wrap_attention_forward(FlashAttentionImpl.forward)
                FlashAttentionImpl._ctsd_patched = True
        except Exception:
            pass

        try:
            from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
            if not getattr(TritonAttentionImpl, "_ctsd_patched", False):
                TritonAttentionImpl.forward = _wrap_attention_forward(TritonAttentionImpl.forward)
                TritonAttentionImpl._ctsd_patched = True
        except Exception:
            pass

        try:
            from vllm.v1.attention.backends.flashinfer import FlashinferAttentionImpl
            if not getattr(FlashinferAttentionImpl, "_ctsd_patched", False):
                FlashinferAttentionImpl.forward = _wrap_attention_forward(FlashinferAttentionImpl.forward)
                FlashinferAttentionImpl._ctsd_patched = True
        except Exception:
            pass
    except Exception:
        logger.exception("vllm-ctsd: failed to intercept attention backend forwards")

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

    # 7. Patch KVCacheManager.allocate_slots to guarantee tree slots during prefill & decode
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        _orig_allocate_slots = KVCacheManager.allocate_slots

        def _patched_allocate_slots(self, request, num_new_tokens, *args, **kwargs):
            cfg_now = CTSDConfig.from_env()
            if cfg_now.enabled:
                curr_lookahead = kwargs.get("num_lookahead_tokens", 0)
                needed = cfg_now.tree_size + 1
                if curr_lookahead < needed:
                    kwargs["num_lookahead_tokens"] = needed
            return _orig_allocate_slots(self, request, num_new_tokens, *args, **kwargs)

        KVCacheManager.allocate_slots = _patched_allocate_slots
    except Exception:
        logger.exception("vllm-ctsd: failed to patch KVCacheManager.allocate_slots")

    vllm_ctsd._REGISTERED = True
    vllm_ctsd._ACTIVATED = True
