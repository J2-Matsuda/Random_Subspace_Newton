"""Registry helpers for problems and algorithms."""

from __future__ import annotations

from typing import Any, Dict, Type

from algorithm.algorithms.gd import GradientDescent
from algorithm.algorithms.rk_rsrnm import RKRsrnm
from algorithm.algorithms.rk_rsrnm_debug import RKRsrnmDebug
from algorithm.algorithms.rk_rsrnm_v2 import RKRsrnmV2
from algorithm.algorithms.rk_rsrnm_v3 import RKRsrnmV3
from algorithm.algorithms.rk_rsrn_mid import RKRsrnMid
from algorithm.algorithms.rsrnm import RSRNM
from algorithm.algorithms.rsrn_mid import RSRNMid
from problem.monotonic_func import MonotoneTransformedProblem, get_monotonic_func
from problem.problems.logistic_regression import LogisticRegressionProblem
from problem.problems.quadratic import QuadraticProblem
from problem.problems.rosenbrock import RosenbrockProblem

PROBLEM_REGISTRY: Dict[str, Type] = {
    "quadratic": QuadraticProblem,
    "rosenbrock": RosenbrockProblem,
    "logistic_regression": LogisticRegressionProblem,
}

ALGORITHM_REGISTRY: Dict[str, Type] = {
    "gd": GradientDescent,
    "rsrnm": RSRNM,
    "rk_rsrnm": RKRsrnm,
    "rk_rsrnm_debug": RKRsrnmDebug,
    "rk_rsrnm_v2": RKRsrnmV2,
    "rk_rsrnm_v3": RKRsrnmV3,
    "rk_rsrn_mid": RKRsrnMid,
    "rsrn_mid": RSRNMid,
}


def build_problem(spec: Dict[str, Any]) -> Any:
    key = spec.get("key")
    if key not in PROBLEM_REGISTRY:
        raise KeyError(f"Unknown problem key: {key}")
    params = spec.get("params", {})
    base_problem = PROBLEM_REGISTRY[key](**params)
    monotonicity = spec.get("monotonicity")
    if monotonicity:
        monotonic = get_monotonic_func(monotonicity)
        return MonotoneTransformedProblem(base_problem=base_problem, monotonic=monotonic)
    return base_problem


def build_algorithm(spec: Dict[str, Any]) -> Any:
    key = spec.get("key")
    if key not in ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm key: {key}")
    params = spec.get("params", {})
    return ALGORITHM_REGISTRY[key](params=params)
