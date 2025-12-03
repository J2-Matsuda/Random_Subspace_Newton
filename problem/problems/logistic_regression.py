"""Binary logistic regression problem with synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.dtypes import DTYPE, as_dtype, zeros


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class LogisticRegressionProblem:
    """Logistic regression with data generated from a planted parameter.

    Data generation follows:
        1. X ~ N(0, I)
        2. If beta is not provided, sample beta ~ N(0, I)
        3. p_i = sigma(X_i beta + intercept)
        4. y_i ~ Bernoulli(p_i)
    """

    dim: int
    n_samples: int
    seed: int = 0
    beta: np.ndarray | None = None
    intercept: float = 0.0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.X = as_dtype(rng.normal(size=(self.n_samples, self.dim)))
        if self.beta is None:
            beta_vec = as_dtype(rng.normal(size=self.dim))
        else:
            beta_vec = as_dtype(self.beta)
        self.beta_true = beta_vec
        logits = self.X @ beta_vec + as_dtype(self.intercept)
        probs = _sigmoid(logits)
        self.y = as_dtype(rng.binomial(1, probs))
        self.dim = self.X.shape[1]

    def value(self, w: np.ndarray) -> float:
        w = as_dtype(w)
        logits = self.X @ w
        # logistic loss: mean of log(1+exp(logits)) - y*logits
        loss = np.logaddexp(0.0, logits) - self.y * logits
        return float(np.mean(loss))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        w = as_dtype(w)
        logits = self.X @ w
        probs = _sigmoid(logits)
        diff = probs - self.y
        grad = (self.X.T @ diff) / as_dtype(self.n_samples)
        return as_dtype(grad)

    def hessian(self, w: np.ndarray) -> np.ndarray:
        w = as_dtype(w)
        logits = self.X @ w
        probs = _sigmoid(logits)
        weights = probs * (1.0 - probs)
        weighted_X = self.X * weights[:, None]
        hess = (self.X.T @ weighted_X) / as_dtype(self.n_samples)
        return as_dtype(hess)

    def initial_point(self) -> np.ndarray:
        return zeros(self.dim)

    @property
    def x_star(self) -> np.ndarray:
        return self.beta_true
