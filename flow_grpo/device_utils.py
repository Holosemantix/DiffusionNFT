import os
from typing import Optional

import torch


def _patch_torch_pytree_for_transformers() -> None:
    """Bridge torch 2.1 private pytree API name expected by transformers >=4.56.

    Must run before importing torch_npu (which transitively imports transformers
    via torch.distributed.fsdp -> torch.onnx -> transformers).
    """

    try:
        import torch.utils._pytree as pytree
    except Exception:
        return

    if not hasattr(pytree, "register_pytree_node") and hasattr(pytree, "_register_pytree_node"):

        def register_pytree_node(type_, flatten_fn, unflatten_fn, *args, **kwargs):
            kwargs.pop("serialized_type_name", None)
            kwargs.pop("to_dumpable_context", None)
            kwargs.pop("from_dumpable_context", None)
            return pytree._register_pytree_node(type_, flatten_fn, unflatten_fn, *args, **kwargs)

        pytree.register_pytree_node = register_pytree_node


_patch_torch_pytree_for_transformers()

# Importing torch_npu registers the NPU backend so torch.npu.is_available() returns
# True on Ascend hosts. Must happen before any device probing. No-op elsewhere.
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


def get_device(local_rank: int = 0) -> torch.device:
    """Auto-select NPU > CUDA > CPU."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def get_pin_memory(device: torch.device) -> bool:
    """NPU 上建议关闭 pin_memory，避免不必要开销或兼容问题。"""
    return device.type == "cuda"


def get_amp_context(device_type: str, enabled: bool, dtype: torch.dtype):
    """获取适合当前设备的 autocast 上下文管理器。"""
    if not enabled:
        import contextlib
        return contextlib.nullcontext()

    if device_type == "npu":
        try:
            from torch.npu.amp import autocast
            return autocast(enabled=True, dtype=dtype)
        except ImportError:
            from torch.amp import autocast
            return autocast(device_type="npu", enabled=True, dtype=dtype)

    from torch.cuda.amp import autocast
    return autocast(enabled=True, dtype=dtype)


def get_grad_scaler(device_type: str, enabled: bool):
    """获取适合当前设备的 GradScaler；如不可用则返回 None。"""
    if not enabled:
        return None

    if device_type == "npu":
        try:
            from torch.npu.amp import GradScaler
            return GradScaler()
        except ImportError:
            try:
                from torch.amp import GradScaler
                return GradScaler(device="npu")
            except Exception:
                return None

    from torch.cuda.amp import GradScaler
    return GradScaler()
