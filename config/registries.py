"""Registry helpers for problems and algorithms."""

from __future__ import annotations

from typing import Any, Dict, Type

from algorithm.algorithms.gd import GradientDescent
from algorithm.algorithms.rk_rsrnm import RKRsrnm
from algorithm.algorithms.rsrnm import RSRNM
from problem.problems.quadratic import QuadraticProblem
from problem.problems.rosenbrock import RosenbrockProblem

PROBLEM_REGISTRY: Dict[str, Type] = {
    "quadratic": QuadraticProblem,
    "rosenbrock": RosenbrockProblem,
}

ALGORITHM_REGISTRY: Dict[str, Type] = {
    "gd": GradientDescent,
    "rsrnm": RSRNM,
    "rk_rsrnm": RKRsrnm,
}


def build_problem(spec: Dict[str, Any]) -> Any:
    key = spec.get("key")
    if key not in PROBLEM_REGISTRY:
        raise KeyError(f"Unknown problem key: {key}")
    params = spec.get("params", {})
    return PROBLEM_REGISTRY[key](**params)


def build_algorithm(spec: Dict[str, Any]) -> Any:
    key = spec.get("key")
    if key not in ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm key: {key}")
    params = spec.get("params", {})
    return ALGORITHM_REGISTRY[key](params=params)
