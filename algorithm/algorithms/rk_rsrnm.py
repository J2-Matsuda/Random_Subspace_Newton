"""RK+RSRNM（要件定義に基づく）アルゴリズム実装。

目的：
  - 内側ループで「RK 風スケッチ解法」により y を改良（Hy ≈ g/||g||）
  - 外側では y を先頭行とする行型スケッチ P を構成し，RSRNM 方向で Armijo 更新

数式との対応：
  - y 初期化： y_0^k = (1 - g_{k-1}^T H_{k-1}(x_k - x_{k-1}) / ||g_{k-1}||^2) y_L^{k-1}
               + (x_k - x_{k-1}) / ||g_{k-1}||
  - RK 風更新： y_{l+1} = y_l - Q^T (Q H Q^T + λ_rk I)^{-1} Q (H y_l - g/||g||)
  - 方向構成：  u = y_L / ||y_L||,  P = [u; Ẑ]（Ẑ は u の直交補から行直交化）
                s = - P^T (P H P^T + λ_rs I)^{-1} P g
  - Armijo：   f(x + α s) ≤ f(x) + c1 α g^T s を満たす α をバックトラックで探索
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from algorithm.base import AlgorithmBase, AlgorithmResult, Logger
from analysis import metrics
from utils.math_tools import (
    ArmijoParams,
    armijo_backtracking,
    norm,
    orthonormalize_rows,
    project_rows_orthogonal,
    rk_residual as rk_residual_fn,
    sample_gaussian_matrix,
    solve_regularized_system,
    symmetrize,
)


class RKRsrnm(AlgorithmBase):
    """RK+RSRNM の最小実装クラス。

    期待パラメータ（self.params）：
        max_iters: int         # 反復上限
        tol: float             # 収束判定（||∇f|| <= tol）
        s: int                 # 行型スケッチ P の行数（サブスペース次元）
        r: int                 # 内側スケッチ Q の行数
        L: int                 # 内側 RK 風更新の繰り返し回数
        rs_reg: float          # (P H P^T + rs_reg I) の正則化
        rk_reg: float          # (Q H Q^T + rk_reg I) の正則化
        rk_tol: float|None     # 内側残差の閾値（任意）
        y_cap: float           # y のノルム上限（数値暴走の保険）
        warm_start: bool       # y のウォームスタートを使うか
        y0: array-like         # k=0 の初期 y
        alpha0, c1, rho:       # Armijo パラメータ
    """

    def run(self, problem, logger: Logger) -> AlgorithmResult:
        """アルゴリズム本体。問題とロガーを受けて反復を実行する。"""
        params = self.params

        # ---- 実行パラメータの取得（既定値つき） ----
        max_iters = params.get("max_iters", 200)
        tol = params.get("tol", 1e-6)

        # 行型スケッチ P の行数（s）と，内側 Q の行数（r），RK 内ループ回数（L）
        rs_rows = max(0, params.get("s", min(8, problem.dim)))
        r_dim = max(1, params.get("r", min(8, problem.dim)))
        L = max(1, params.get("L", 10))

        # 小行列の正則化項（安定化の鍵）
        rs_reg = params.get("rs_reg", 1e-8)
        rk_reg = params.get("rk_reg", 1e-8)

        # 内側残差閾値と暴走対策
        rk_tol = params.get("rk_tol", None)
        y_cap = params.get("y_cap", 1e6)

        # y の初期化戦略
        warm_start = params.get("warm_start", True)
        y0_vec = np.asarray(params.get("y0", np.zeros(problem.dim)), dtype=float)

        # Armijo の設定（α をバックトラックで探索）
        armijo_params = ArmijoParams(
            alpha0=params.get("alpha0", 1.0),
            c1=params.get("c1", 1e-4),
            rho=params.get("rho", 0.5),
        )

        # ---- 初期点 ----
        x = problem.initial_point().astype(float)
        fx = problem.value(x)
        history = []
        converged = False
        prev_state: Optional[Dict[str, np.ndarray | float]] = None  # y, g, H, x などを保持

        # ---- 反復 ----
        for k in range(max_iters):
            grad = problem.gradient(x)
            grad_norm = float(np.linalg.norm(grad))
            hessian = self._hessian(problem, x)

            # 収束判定：||∇f|| <= tol
            if grad_norm <= tol:
                converged = True
                row = metrics.build_row(
                    iteration=k,
                    value=fx,
                    grad=grad,
                    x=x,
                    extra=self._rk_extras(
                        alpha=0.0,
                        subspace_rows=1 + rs_rows,  # 先頭行 u + 直交補 rs_rows 行
                        r_dim=r_dim,
                        L=0,
                        rs_reg=rs_reg,
                        rk_reg=rk_reg,
                        rk_iters=0,
                        rk_residual=0.0,
                    ),
                )
                logger.log(row)
                history.append(row)
                break

            # g を正規化して rhs = g/||g|| を作る（Hy ≈ rhs を解くイメージ）
            grad_unit = grad / (grad_norm + 1e-16)

            # y の初期化：ウォームスタートなら提案式，そうでなければ y0
            y_init = self._initialise_y(
                current_x=x,
                default=y0_vec,
                warm_start=warm_start,
                prev_state=prev_state,
            )

            # ---- 内側：RK 風スケッチ更新（L 回）----
            y_vec, inner_iters, inner_residual = self._rk_inner_loop(
                y_init=y_init,
                hessian=hessian,
                grad_unit=grad_unit,
                r_dim=r_dim,
                L=L,
                rk_reg=rk_reg,
                rk_tol=rk_tol,
                y_cap=y_cap,
            )

            # ---- 外側：方向 s の構成（行型 P：先頭行 u=y/||y||，残りは直交補から採用）----
            direction, P_rows = self._construct_direction(
                y_vec=y_vec,
                hessian=hessian,
                grad=grad,
                rs_rows=rs_rows,
                rs_reg=rs_reg,
            )

            # 念のため下降方向に矯正（g^T s < 0 を保証）
            direction = self._ensure_descent(direction, grad)

            # ---- Armijo で α を探索し，更新 ----
            alpha, new_fx = armijo_backtracking(
                problem.value, x, direction, grad, armijo_params, fx=fx
            )

            # ログ（k 時点の値を記録）
            row = metrics.build_row(
                iteration=k,
                value=fx,
                grad=grad,
                x=x,
                extra=self._rk_extras(
                    alpha=alpha,
                    subspace_rows=P_rows,
                    r_dim=r_dim,
                    L=L,
                    rs_reg=rs_reg,
                    rk_reg=rk_reg,
                    rk_iters=inner_iters,
                    rk_residual=inner_residual,
                ),
            )
            logger.log(row)
            history.append(row)

            # α=0 の場合はこれ以上前進できないので打ち切り
            if alpha == 0.0:
                break

            # 次イテレーション用の状態を保存（ウォームスタートで使用）
            prev_state = {
                "x": x.copy(),
                "grad": grad.copy(),
                "grad_norm": grad_norm,
                "hessian": hessian.copy(),
                "y": y_vec.copy(),
            }

            # 前進
            x = x + alpha * direction
            fx = new_fx

        # ---- 終了処理 ----
        logger.flush()
        final_grad = problem.gradient(x)
        return AlgorithmResult(
            x=x,
            f=problem.value(x),
            grad_norm=float(np.linalg.norm(final_grad)),
            iterations=len(history),
            converged=converged,
            history=history,
        )

    # =========================
    # 小さなユーティリティ群
    # =========================

    @staticmethod
    def _hessian(problem, x: np.ndarray) -> np.ndarray:
        """H(x) を取得（未実装なら I を使用）。対称化して返す。"""
        hessian = getattr(problem, "hessian", lambda _: None)(x)
        if hessian is None:
            hessian = np.eye(x.shape[0])
        return symmetrize(hessian)

    @staticmethod
    def _initialise_y(
        *,
        current_x: np.ndarray,
        default: np.ndarray,
        warm_start: bool,
        prev_state: Optional[Dict[str, np.ndarray | float]],
    ) -> np.ndarray:
        """y の初期化。

        - warm_start=True かつ prev_state があれば，提案式で y_0^k を構成
        - それ以外は default（y0）を返す
        """
        if not warm_start or prev_state is None:
            return default.copy()

        prev_grad_norm = float(prev_state["grad_norm"])
        if prev_grad_norm <= 0:
            return default.copy()

        x_diff = current_x - prev_state["x"]
        g_prev = prev_state["grad"]
        H_prev = prev_state["hessian"]
        y_prev = prev_state["y"]

        # 係数：1 - g_{k-1}^T H_{k-1} (x_k - x_{k-1}) / ||g_{k-1}||^2
        numerator = g_prev @ (H_prev @ x_diff)
        coeff = 1.0 - numerator / (prev_grad_norm**2 + 1e-16)

        # 追加項：(x_k - x_{k-1}) / ||g_{k-1}||
        update = x_diff / (prev_grad_norm + 1e-16)

        return coeff * y_prev + update

    @staticmethod
    def _rk_inner_loop(
        *,
        y_init: np.ndarray,
        hessian: np.ndarray,
        grad_unit: np.ndarray,
        r_dim: int,
        L: int,
        rk_reg: float,
        rk_tol: Optional[float],
        y_cap: float,
    ) -> Tuple[np.ndarray, int, float]:
        """RK 風の内側ループ（L 回）。

        1) 残差 r = H y - g/||g||
        2) Q ~ N(0, 1/r_dim) の行型スケッチで r を投影して小系を解く
        3) y ← y - Q^T (QHQ^T + λ I)^{-1} Q r
        """
        y = y_init.copy()
        residual_value = float("inf")
        performed = 0

        for l in range(L):
            performed = l + 1

            # 残差計算（安定化のため norm() は utils のやつを使用）
            residual_vec = hessian @ y - grad_unit
            residual_value = norm(residual_vec)

            # 数値異常の回復（安全策：rhs を模した y に切替）
            if not np.isfinite(residual_value):
                y = grad_unit.copy()
                residual_value = norm(hessian @ y - grad_unit)
                break

            # しきい値が与えられていれば早期停止
            if rk_tol is not None and residual_value <= rk_tol:
                break

            # 行型スケッチ Q： N(0, 1/r_dim)
            Q = sample_gaussian_matrix(r_dim, hessian.shape[0], 1.0 / r_dim)

            # qhq = Q H Q^T を対称化して小系を安定化
            qhq = symmetrize(Q @ hessian @ Q.T)
            rhs = Q @ residual_vec

            # (Q H Q^T + rk_reg I) δ = Q r を解いて y を更新
            delta = solve_regularized_system(qhq, rhs, rk_reg)
            y = y - Q.T @ delta

            # y の暴走（NaN/Inf，過大ノルム）を検出したら安全側に倒す
            if not np.all(np.isfinite(y)) or norm(y) > y_cap:
                y = grad_unit.copy()
                break

        # 最終残差（ログ用）
        final_residual = rk_residual_fn(hessian, y, grad_unit)
        return y, performed, final_residual

    def _construct_direction(
        self,
        *,
        y_vec: np.ndarray,
        hessian: np.ndarray,
        grad: np.ndarray,
        rs_rows: int,
        rs_reg: float,
    ) -> Tuple[np.ndarray, int]:
        """RSRNM 方向 s を構成する。

        - 先頭行 u = y/||y|| を固定
        - 残り rs_rows 行は u の直交補からガウス→射影→行直交化で生成
        - s = - P^T (P H P^T + λ I)^{-1} P g
        """
        # y がほぼゼロなら勾配正規化で代用
        y_norm = norm(y_vec)
        if y_norm < 1e-12:
            y_vec = grad / (norm(grad) + 1e-16)
            y_norm = norm(y_vec)

        # 先頭行
        u = (y_vec / y_norm)[None, :]

        # 直交補行列の生成（rs_rows=0 なら先頭行のみ）
        if rs_rows > 0:
            rows = self._orth_complement(rs_rows, hessian.shape[0], u[0])
            P = np.vstack([u, rows])
        else:
            P = u

        # 小系（s×s）を組んで解く
        phpt = symmetrize(P @ hessian @ P.T)
        rhs = P @ grad
        delta = solve_regularized_system(phpt, rhs, rs_reg)

        # 方向
        direction = -P.T @ delta
        return direction, P.shape[0]

    @staticmethod
    def _orth_complement(rows: int, dim: int, unit_vec: np.ndarray) -> np.ndarray:
        """unit_vec（長さ n）に直交な行ベクトルを rows 本生成し，行直交化して返す。"""
        attempts = 0
        variance = 1.0 / max(rows, 1)  # N(0, 1/s) の分散に対応
        while attempts < 5:
            attempts += 1
            gaussian = sample_gaussian_matrix(rows, dim, variance)  # 行型
            projected = project_rows_orthogonal(gaussian, unit_vec)  # 直交補へ射影
            row_norms = np.linalg.norm(projected, axis=1)
            if np.all(row_norms > 1e-10):
                return orthonormalize_rows(projected)  # 行直交化
        # うまくいかないときのフォールバック：単位行列から拝借
        identity = np.eye(dim)
        projected = project_rows_orthogonal(identity[:rows], unit_vec)
        return orthonormalize_rows(projected)

    @staticmethod
    def _rk_extras(
        *,
        alpha: float,
        subspace_rows: int,
        r_dim: int,
        L: int,
        rs_reg: float,
        rk_reg: float,
        rk_iters: int,
        rk_residual: float,
    ) -> Dict[str, float]:
        """CSV/JSONL に載せる追加メトリクスを dict 化。"""
        return {
            "alpha": float(alpha),
            "subspace_dim_s": subspace_rows,
            "inner_dim_r": r_dim,
            "L": L,
            "rs_reg": float(rs_reg),
            "rk_reg": float(rk_reg),
            "rk_iters": rk_iters,
            "rk_residual": float(rk_residual),
        }

    @staticmethod
    def _ensure_descent(direction: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """念のため下降方向に調整（g^T s >= 0 の場合は -g に置き換え）。"""
        if grad @ direction >= 0:
            return -grad
        return direction