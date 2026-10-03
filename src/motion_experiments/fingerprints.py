"""Exact, typed fingerprints for immutable arrays and processor structures."""
from __future__ import annotations

import hashlib

import numpy as np


def array_fingerprint(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise TypeError('Object arrays cannot be fingerprinted as raw bytes')
    return dict(type='numpy.ndarray', dtype=value.dtype.str, shape=list(value.shape),
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def fingerprint(value):
    """Preserve container types, scalar values, dtypes, shapes and every byte."""
    import torch

    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(type='torch.Tensor', dtype=str(value.dtype), shape=list(value.shape),
                    sha256=hashlib.sha256(raw).hexdigest())
    if isinstance(value, np.ndarray):
        return array_fingerprint(value)
    if isinstance(value, np.generic):
        return dict(type='numpy.scalar', value=array_fingerprint(value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError('Processor mapping keys must be strings')
        return dict(type='dict', fields={key:fingerprint(value[key]) for key in sorted(value)})
    if isinstance(value, (tuple, list)):
        return dict(type=type(value).__name__, items=[fingerprint(item) for item in value])
    if value is None or type(value) in (bool, int, float, str):
        # hex preserves signed zero and nonfinite values without invalid JSON.
        scalar = value.hex() if type(value) is float else value
        return dict(type=type(value).__name__, value=scalar)
    raise TypeError(f'Unsupported fingerprint type: {type(value).__name__}')
