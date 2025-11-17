"""Central place to define the numeric dtype used across the project."""

from __future__ import annotations

import numpy as np

_DEFAULT = np.float64
try:
    _EXTENDED = np.longdouble
    _has_extended = np.finfo(_EXTENDED).eps < np.finfo(_DEFAULT).eps
except (AttributeError, ValueError):
    _has_extended = False

DTYPE = _EXTENDED if _has_extended else _DEFAULT


def as_dtype(array_like):
    """Convert input to the global dtype."""
    return np.asarray(array_like, dtype=DTYPE)


def zeros(shape):
    return np.zeros(shape, dtype=DTYPE)


def eye(size):
    return np.eye(size, dtype=DTYPE)


def random_normal(mean: float, std: float, size):
    return np.random.normal(loc=mean, scale=std, size=size).astype(DTYPE, copy=False)
