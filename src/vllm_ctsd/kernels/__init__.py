import torch

def _ctsd_unified_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    from vllm.forward_context import get_attention_context
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)
    return output


def _ctsd_unified_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


def register_ctsd_attention_ops():
    from vllm.utils.torch_utils import direct_register_custom_op
    try:
        direct_register_custom_op(
            op_name="ctsd_tree_attention",
            op_func=_ctsd_unified_attention_impl,
            fake_impl=_ctsd_unified_attention_fake,
        )
    except Exception as e:
        import logging
        logging.getLogger("vllm_ctsd").debug("register_ctsd_attention_ops notice: %s", e)
