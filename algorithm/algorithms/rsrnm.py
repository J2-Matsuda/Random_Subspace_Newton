"""Random Subspace Regularised Newton Method（RSRNM, 直解版）

このモジュールは，行型ランダムスケッチ P（形状 s×n）を用いて
制限二次モデルの小規模系を構築・直接解き，Armijo バックトラッキングで
ステップ長を決めて更新する最小実装です。

更新式（方向）:
    s_k = - P_k^T ( P_k H_k P_k^T + λ I )^{-1} ( P_k g_k )

ラインサーチ:
    Armijo 条件  f(x_k + α s_k) ≤ f(x_k) + c1 α ⟨g_k, s_k⟩ を満たす α を
    ρ で縮小しながら探索
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from algorithm.base import AlgorithmBase, AlgorithmResult, Logger
from analysis import metrics
from utils.math_tools import (
    ArmijoParams,
    armijo_backtracking,
    orthonormalize_rows,
    sample_gaussian_matrix,
    solve_regularized_system,
    symmetrize,
)


class RSRNM(AlgorithmBase):
    """RSRNM（直解）の最小実装クラス。

    期待するパラメータ (self.params):
        max_iters: int      反復上限
        tol: float          収束判定の閾値（||∇f|| ≤ tol）
        s: int              スケッチ行列 P の行数（= サブスペース次元）
        rs_reg: float       小系 (P H P^T) に加える正則化 λ
        alpha0, c1, rho:    Armijo バックトラッキングの各パラメータ
    """

    def run(self, problem, logger: Logger) -> AlgorithmResult:
        """アルゴリズム本体を実行して結果を返す。

        流れ:
            1) 勾配 g_k, ヘッセ H_k を評価
            2) 行型スケッチ P_k（s×n）をサンプルし，P_k H_k P_k^T を形成
            3) 小系 (P H P^T + λI) δ = P g を直接解いて方向 s = -P^T δ を取得
            4) Armijo バックトラックで α を求め，x ← x + α s
            5) ||∇f|| ≤ tol または α=0 で停止
        """
        params = self.params

        # --- 反復条件・閾値 ---
        max_iters = params.get("max_iters", 200)
        tol = params.get("tol", 1e-6)

        # --- サブスペース次元（P の行数）と正則化 ---
        s_dim = max(1, params.get("s", min(8, problem.dim)))
        rs_reg = params.get("rs_reg", 1e-8)

        # --- Armijo バックトラッキングの設定 ---
        armijo_params = ArmijoParams(
            alpha0=params.get("alpha0", 1.0),
            c1=params.get("c1", 1e-4),
            rho=params.get("rho", 0.5),
        )

        # --- 初期化 ---
        x = problem.initial_point().astype(float)
        fx = problem.value(x)
        history = []
        converged = False
        self._init_progress_tracker(max_iters)  # 進捗表示などがあれば有効化

        # ===============================
        #        主要ループ
        # ===============================
        for k in range(max_iters):
            grad = problem.gradient(x)
            grad_norm = float(np.linalg.norm(grad))
            hessian = self._hessian(problem, x)  # 未提供なら I を仮採用

            # ---- 収束判定：||∇f|| ≤ tol ----
            if grad_norm <= tol:
                converged = True
                row = metrics.build_row(
                    iteration=k,
                    value=fx,
                    grad=grad,
                    x=x,
                    extra=self._rs_extras(alpha=0.0, s_dim=s_dim, rs_reg=rs_reg),
                )
                logger.log(row)
                history.append(row)
                self._report_progress(k)
                break

            # ---- 行型スケッチ P（s×n）を生成 → 行直交化で数値安定化 ----
            P = self._sample_p(rows=s_dim, dim=problem.dim)

            # ---- 小系を構築：phpt = P H P^T，rhs = P g ----
            phpt = symmetrize(P @ hessian @ P.T)  # 対称化で誤差吸収
            rhs = P @ grad

            # ---- 小系 (phpt + λI) δ = rhs を直接解：δ を得て s = -P^T δ ----
            delta_subspace = solve_regularized_system(phpt, rhs, rs_reg)
            direction = -P.T @ delta_subspace

            # 念のため下降方向に矯正（⟨g, s⟩≥0 なら -g に置換）
            direction = self._ensure_descent(direction, grad)

            # ---- Armijo バックトラッキングで α を探索 ----
            alpha, new_fx = armijo_backtracking(
                problem.value, x, direction, grad, armijo_params, fx=fx
            )

            # ---- ログ出力（k 時点の値を記録）----
            row = metrics.build_row(
                iteration=k,
                value=fx,
                grad=grad,
                x=x,
                extra=self._rs_extras(alpha=alpha, s_dim=s_dim, rs_reg=rs_reg),
            )
            logger.log(row)
            history.append(row)
            self._report_progress(k)

            # ステップが確定しない場合（α=0）は打ち切り
            if alpha == 0.0:
                break

            # ---- 前進 ----
            x = x + alpha * direction
            fx = new_fx

        # --- 終了処理・要約 ---
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
    # 小ユーティリティ群
    # =========================

    @staticmethod
    def _hessian(problem, x: np.ndarray) -> np.ndarray:
        """H(x) を取得（未実装なら I を使用）。対称化して返す。"""
        hessian = getattr(problem, "hessian", lambda _: None)(x)
        if hessian is None:
            hessian = np.eye(x.shape[0])
        return symmetrize(hessian)

    @staticmethod
    def _sample_p(rows: int, dim: int) -> np.ndarray:
        """行型ガウススケッチ P（rows×dim）を生成して行直交化して返す。

        分布は要件通り，各要素 ~ N(0, 1/rows) を採用。
        その後，行直交化（QR を列に当てて転置）で数値安定性を高める。
        """
        variance = 1.0 / rows
        gaussian = sample_gaussian_matrix(rows, dim, variance)
        return orthonormalize_rows(gaussian)

    @staticmethod
    def _rs_extras(alpha: float, s_dim: int, rs_reg: float) -> Dict[str, float]:
        """CSV/JSONL に載せる追加メトリクスを dict 化。"""
        return {
            "alpha": float(alpha),
            "subspace_dim_s": s_dim,  # P の行数（サブスペース次元）
            "inner_dim_r": 0,         # 直解なので 0
            "L": 0,                   # 内側反復なし
            "rs_reg": float(rs_reg),  # 小系の正則化
            "rk_reg": 0.0,            # 直解なので 0
            "rk_iters": 0,            # 直解なので 0
            "rk_residual": 0.0,       # 直解なので 0
        }

    @staticmethod
    def _ensure_descent(direction: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """下降方向を保証する（⟨g, s⟩≥0 の場合は -g に置換）。"""
        if grad @ direction >= 0:
            return -grad
        return direction
