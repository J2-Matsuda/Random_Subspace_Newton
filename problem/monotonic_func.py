"""Monotonic scalar functions used to wrap base problems.

Functions are defined on the scalar objective value f(x) and provide
its value, first derivative, and second derivative with respect to that
scalar. These are combined with the base problem via the chain rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from utils.dtypes import DTYPE, as_dtype


@dataclass(frozen=True)
class MonotonicFunc:
    value: Callable[[float], float]
    first: Callable[[float], float]
    second: Callable[[float], float]


def _identity(t: float) -> float:
    return float(t)


def _identity_first(_: float) -> float:
    return 1.0


def _identity_second(_: float) -> float:
    return 0.0


def _exp(t: float) -> float:
    return float(np.exp(t, dtype=DTYPE))


def _exp_first(t: float) -> float:
    return float(np.exp(t, dtype=DTYPE))


def _exp_second(t: float) -> float:
    return float(np.exp(t, dtype=DTYPE))


def _neg_log(t: float) -> float:
    return float(-np.log(t, dtype=DTYPE))


def _neg_log_first(t: float) -> float:
    return float(-1.0 / as_dtype(t))


def _neg_log_second(t: float) -> float:
    t_dt = as_dtype(t)
    return float(1.0 / (t_dt * t_dt))


MONOTONIC_REGISTRY: Dict[str, MonotonicFunc] = {
    "x": MonotonicFunc(value=_identity, first=_identity_first, second=_identity_second),
    "exp": MonotonicFunc(value=_exp, first=_exp_first, second=_exp_second),
    "-log": MonotonicFunc(value=_neg_log, first=_neg_log_first, second=_neg_log_second),
}


def get_monotonic_func(name: str) -> MonotonicFunc:
    """Return monotonic function definition."""
    if name not in MONOTONIC_REGISTRY:
        raise KeyError(f"Unknown monotonicity: {name}")
    return MONOTONIC_REGISTRY[name]


@dataclass
class MonotoneTransformedProblem:
    """Wrap a base problem with a scalar monotonic transform m(f(x)).

    For a base objective f, the transformed objective is m(f(x)).
    Gradient and Hessian follow from the chain rule:
        ∇(m∘f) = m'(f(x)) ∇f(x)
        ∇²(m∘f) = m''(f(x)) ∇f(x)∇f(x)^T + m'(f(x)) ∇²f(x)
    """

    base_problem: Any
    monotonic: MonotonicFunc

    def __post_init__(self) -> None:
        self.dim = self.base_problem.dim

    def value(self, x: np.ndarray) -> float:
        inner = float(self.base_problem.value(x))
        return self.monotonic.value(inner)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        inner = float(self.base_problem.value(x))
        base_grad = as_dtype(self.base_problem.gradient(x))
        scale = as_dtype(self.monotonic.first(inner))
        return as_dtype(scale * base_grad)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        inner = float(self.base_problem.value(x))
        base_grad = as_dtype(self.base_problem.gradient(x))
        base_hess = as_dtype(self.base_problem.hessian(x))
        first = as_dtype(self.monotonic.first(inner))
        second = as_dtype(self.monotonic.second(inner))
        outer = np.outer(base_grad, base_grad)
        return as_dtype(first * base_hess + second * outer)

    def initial_point(self) -> np.ndarray:
        return self.base_problem.initial_point()
