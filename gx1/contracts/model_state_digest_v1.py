"""Canonical tensor-state digest shared by trainer and bundle verification.

This intentionally hashes tensor names, dtype, shape and contiguous CPU bytes,
not a ``torch.save`` file.  The latter binds artifact bytes; this digest binds
evidence produced while a particular in-memory model state was evaluated.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


def canonical_model_state_sha256(state_dict: Mapping[str, Any]) -> str:
    """Return the exact semantic digest for a model ``state_dict``."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("[MODEL_STATE_DIGEST_STATE_DICT_INVALID]")
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        if not isinstance(name, str) or not name:
            raise RuntimeError("[MODEL_STATE_DIGEST_KEY_INVALID]")
        if not all(
            hasattr(tensor, attribute)
            for attribute in ("detach", "cpu", "contiguous", "numpy", "shape", "dtype")
        ):
            raise RuntimeError("[MODEL_STATE_DIGEST_TENSOR_INVALID]")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


__all__ = ["canonical_model_state_sha256"]
