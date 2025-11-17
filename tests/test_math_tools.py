import numpy as np

from algorithm.algorithms.rk_rsrnm import RKRsrnm
from algorithm.algorithms.rsrnm import RSRNM
from utils.math_tools import ArmijoParams, armijo_backtracking, rk_residual


def test_sample_p_rows_are_orthonormal():
    rows, dim = 4, 10
    P = RSRNM._sample_p(rows, dim)
    gram = P @ P.T
    assert P.shape == (rows, dim)
    assert np.allclose(gram, np.eye(rows), atol=1e-6)


def test_projected_matrix_is_spd():
    rows, dim = 3, 8
    H = np.diag(np.linspace(1.0, 10.0, dim))
    P = RSRNM._sample_p(rows, dim)
    mat = P @ H @ P.T + 1e-4 * np.eye(rows)
    eigenvalues = np.linalg.eigvalsh(mat)
    assert eigenvalues.min() > 0


def test_armijo_condition_satisfied():
    def value_fn(x: np.ndarray) -> float:
        return 0.5 * float(x @ x)

    x = np.array([1.0, 1.0])
    grad = x.copy()
    direction = -grad
    params = ArmijoParams(t0=1.0, alpha=1e-4, beta=0.5)
    alpha, new_val = armijo_backtracking(
        value_fn, x, direction, grad, params, fx=value_fn(x)
    )
    assert alpha > 0
    lhs = new_val
    rhs = value_fn(x) + params.alpha * alpha * (grad @ direction)
    assert lhs <= rhs + 1e-12


def test_rk_inner_loop_reduces_residual():
    dim = 6
    hessian = np.eye(dim)
    grad_unit = np.ones(dim) / np.sqrt(dim)
    algo = RKRsrnm(
        {
            "s": 2,
            "rs_reg": 1e-8,
            "r": 3,
            "L": 5,
            "rk_reg": 1e-8,
        }
    )
    y, iters, residual = algo._rk_inner_loop(
        y_init=np.zeros(dim),
        hessian=hessian,
        grad_unit=grad_unit,
        r_dim=3,
        L=5,
        rk_reg=1e-8,
        rk_tol=None,
        y_cap=1e6,
    )
    assert iters >= 1
    initial_residual = rk_residual(hessian, np.zeros(dim), grad_unit)
    assert residual < initial_residual
    assert np.isfinite(residual)
