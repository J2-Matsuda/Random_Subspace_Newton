"""RK+RSRNM algorithm using scaled u variables (u = s * y).

このクラスは RK + RS-RNM アルゴリズムの「u 版」実装である。
理論上の内側変数 y_l^k に対して u_l^k = s_k * y_l^k（s_k = ||g_k||）という
スケーリングを導入することで、内側の RK 更新から 1/||g_k|| を取り除き、
数値的に安定な形で H_k^{-1} g_k を近似する。

・内側ループ：  u_l^k を Randomized Kaczmarz で H_k u = g_k の解に収束させる
・外側ループ：  u_L^k を用いて RS-RNM の部分空間 P_k を構成し方向を求める
   （実装では P_k の第 1 行が u_L^k^T、残りの行は u_L^k と直交する行列 \tilde P_k）
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from algorithm.base import AlgorithmBase, AlgorithmResult, Logger
from analysis import metrics
from utils.dtypes import DTYPE, as_dtype, eye
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


class RKRsrnmV3(AlgorithmBase):
    """Scaling-aware RK+RSRNM (u-version).

    数値計算上の工夫として、理論で用いる y_l^k の代わりに
    u_l^k = s_k * y_l^k,  s_k = ||g_k|| を内部変数として保持する。

    - 内側 RK は H_k u = g_k の形にし、右辺から 1/||g_k|| を削除することで
      非常に小さい勾配ノルムに対しても安定な反復を行う。
    - 反復終了後は u_L^k をそのまま RS-RNM の部分空間に使い、
      P_k = [u_L^k^T; \tilde P_k] を構成する（\tilde P_k は u_L^k と直交）。
    """

    def run(self, problem, logger: Logger) -> AlgorithmResult:
        """メインの反復ループを実行する。

        Parameters
        ----------
        problem : 最適化問題（value, gradient, hessian を持つオブジェクト）
        logger : ログ出力インターフェース

        流れ
        ----
        for k in range(max_iters):
            1. 勾配 g_k, ヘッセ行列 H_k, s_k = ||g_k|| を計算
            2. ||g_k|| <= tol なら収束とみなして終了
            3. u_0^k を初期化（warm_start の場合は r_k, φ_k を用いた近接初期化）
            4. RK 内側ループで H_k u = g_k を近似的に解く（u_L^k を得る）
            5. u_L^k を用いて RS-RNM の部分空間 P_k を構成し方向 d_k を算出
            6. Armijo 直線探索でステップ長 t_k を決定し、x_{k+1} = x_k + t_k d_k で更新
            7. 前ステップ情報 prev_state を保存して次の warm start に利用
        """
        params = self.params
        max_iters = params.get("max_iters", 200)
        tol = params.get("tol", 1e-6)

        # RS-RNM のサブスペース次元 s、および RK のサブスペース次元 r
        rs_rows = max(0, params.get("s", min(8, problem.dim)))
        r_dim = max(1, params.get("r", min(8, problem.dim)))

        # RK の内側反復回数 L
        L = max(1, params.get("L", 10))

        # 正則化パラメータ（RS 部分空間と RK サブ問題）
        rs_reg = params.get("rs_reg", 1e-8)
        rk_reg = params.get("rk_reg", 1e-8)

        # RK 内側ループの停止しきい値（残差ベース）。None のときは回数 L 固定。
        rk_tol = params.get("rk_tol", None)

        # u のノルムがこれを超えるとき、安全のためにリセットする上限
        y_cap = params.get("y_cap", 1e6)

        # warm_start: 前反復の情報を用いて u_0^k を賢く初期化するかどうか
        warm_start = params.get("warm_start", True)

        # 初期の y_0^0, u_0^0（u_0^0 はあれば優先して使う）
        y0_vec = as_dtype(params.get("y0", np.zeros(problem.dim)))
        raw_u0 = params.get("u0")
        u0_vec = as_dtype(raw_u0) if raw_u0 is not None else None

        # Armijo 直線探索のパラメータ設定
        armijo_params = ArmijoParams(
            t0=params.get("t0", 1.0),
            alpha=params.get("alpha", 1e-4),
            beta=params.get("beta", 0.5),
            max_backtracks=params.get("max_backtracks", 50),
            min_alpha=params.get("min_alpha", 1e-16),
        )

        # 初期点と目的関数値
        x = as_dtype(problem.initial_point())
        fx = problem.value(x)

        history: list[Dict[str, float]] = []
        converged = False

        # 前ステップの状態（x_{k-1}, g_{k-1}, H_{k-1}, u_L^{k-1}, s_{k-1}）を保持する
        prev_state: Optional[Dict[str, np.ndarray | float]] = None

        self._init_progress_tracker(max_iters)

        # 数値安定化用のごく小さい eps
        eps = as_dtype(1e-16)

        for k in range(max_iters):
            # --- 1. 勾配・ヘッセ行列の評価 ---
            grad = as_dtype(problem.gradient(x))
            grad_norm = float(np.linalg.norm(grad))
            hessian = self._hessian(problem, x)
            s_k = float(grad_norm)  # s_k = ||g_k||

            # --- 2. 収束判定（勾配ノルムベース） ---
            if grad_norm <= tol:
                converged = True
                row = metrics.build_row(
                    iteration=k,
                    value=fx,
                    grad=grad,
                    x=x,
                    extra=self._rk_extras(
                        t_k=0.0,
                        subspace_rows=1 + rs_rows,
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
                self._report_progress(k)
                break

            # --- 3. u_0^k の初期化（warm start or default） ---
            u_init = self._initialise_u(
                current_x=x,
                grad=grad,
                s_k=s_k,
                default_u=u0_vec,
                default_y=y0_vec,
                warm_start=warm_start,
                prev_state=prev_state,
                eps=eps,
            )

            # --- 4. 内側 RK ループ： H_k u = g_k を近似的に解く ---
            u_vec, inner_iters, inner_residual = self._rk_inner_loop(
                u_init=u_init,
                hessian=hessian,
                grad=grad,
                r_dim=r_dim,
                L=L,
                rk_reg=rk_reg,
                rk_tol=rk_tol,
                y_cap=y_cap,
            )

            # --- 5. RS-RNM の方向構成 ---
            # ここで P_k の第 1 行として u_L^k^T を用い、残りの行を u_L^k と直交させる
            direction, P_rows = self._construct_direction(
                u_vec=u_vec,
                hessian=hessian,
                grad=grad,
                rs_rows=rs_rows,
                rs_reg=rs_reg,
            )
            # 念のため、方向が降下方向になっているか確認し、必要なら -grad に差し替える
            direction = self._ensure_descent(direction, grad)

            # --- 6. Armijo 直線探索（ステップ長 t_k の決定） ---
            t_k, new_fx = armijo_backtracking(
                problem.value, x, direction, grad, armijo_params, fx=fx
            )

            # ログ行追加
            row = metrics.build_row(
                iteration=k,
                value=fx,
                grad=grad,
                x=x,
                extra=self._rk_extras(
                    t_k=t_k,
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
            self._report_progress(k)

            # t_k = 0 の場合は Armijo による進行が止まったとみなし、ループ終了
            if t_k == 0.0:
                break

            # --- 7. 次の warm start のために現在の状態を保存 ---
            prev_state = {
                "x": x.copy(),
                "grad": grad.copy(),
                "hessian": hessian.copy(),
                "u": u_vec.copy(),
                "s": s_k,
            }

            # --- 8. 外側反復の更新 x_{k+1} = x_k + t_k d_k ---
            x = x + t_k * direction
            fx = new_fx

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

    @staticmethod
    def _hessian(problem, x: np.ndarray) -> np.ndarray:
        """ヘッセ行列 H(x) を返す。

        problem が hessian(x) を実装していない場合は単位行列 I を用いる。
        必要に応じて symmetrize して対称行列にする。
        """
        hessian = getattr(problem, "hessian", lambda _: None)(x)
        if hessian is None:
            hessian = eye(x.shape[0])
        return symmetrize(hessian)

    @staticmethod
    def _initialise_u(
        *,
        current_x: np.ndarray,
        grad: np.ndarray,
        s_k: float,
        default_u: Optional[np.ndarray],
        default_y: np.ndarray,
        warm_start: bool,
        prev_state: Optional[Dict[str, np.ndarray | float]],
        eps: float,
    ) -> np.ndarray:
        """u_0^k の初期値を設定する。

        ・warm_start = False または prev_state が None:
            - default_u が与えられていればそれをコピー
            - そうでなければ、y_0^0 * s_k を初期値として用いる
        ・warm_start = True かつ prev_state がある場合:
            - 論文の式に対応する
                r_k = s_k / s_{k-1},  φ_k = g_{k-1}^T H_{k-1} (x_k - x_{k-1}) / ||g_{k-1}||^2
              を用いて
                u_0^k = r_k (1 - φ_k) u_L^{k-1} + r_k (x_k - x_{k-1})
              を計算する。
            - 数値的なゼロ割りを避けるため、分母には eps を加えている。
        """
        # warm start を使わない場合、あるいは前ステートが無い場合は単純な初期化
        if not warm_start or prev_state is None:
            if default_u is not None:
                return default_u.copy()
            # s_k が正なら y * s_k、そうでない場合はスケール 1 で初期化
            return as_dtype(default_y * (s_k if s_k > 0 else 1.0))

        # ここから warm_start: k >= 1 のケース
        s_prev = float(prev_state["s"])
        if s_prev <= 0:
            # 前回の勾配ノルムが異常に小さい場合は warm start を諦めてデフォルトへ
            return as_dtype(default_y * (s_k if s_k > 0 else 1.0))

        # r_k = s_k / s_{k-1}
        r_k = s_k / (s_prev + eps)

        # x_k - x_{k-1}
        x_diff = current_x - prev_state["x"]

        # φ_k = g_{k-1}^T H_{k-1} (x_k - x_{k-1}) / ||g_{k-1}||^2
        g_prev = prev_state["grad"]
        H_prev = prev_state["hessian"]
        u_prev = prev_state["u"]
        numerator = g_prev @ (H_prev @ x_diff)
        phi_k = numerator / (s_prev**2 + eps)

        # u_0^k = r_k (1 - φ_k) u_L^{k-1} + r_k (x_k - x_{k-1})
        return r_k * (1.0 - phi_k) * u_prev + r_k * x_diff

    @staticmethod
    def _rk_inner_loop(
        *,
        u_init: np.ndarray,
        hessian: np.ndarray,
        grad: np.ndarray,
        r_dim: int,
        L: int,
        rk_reg: float,
        rk_tol: Optional[float],
        y_cap: float,
    ) -> Tuple[np.ndarray, int, float]:
        """RK による内側反復で u を更新する。

        解きたい方程式は H u = g である。

        各ステップ l で行っていること:
            1. residual_vec = H u - g を計算し、そのノルムで収束判定
            2. ランダム行列 Q_l^k をサンプル
            3. サブ問題 (Q H Q^T) δ = Q residual を正則化付きで解く
            4. u <- u - Q^T δ で更新

        数値的に不安定な状況（残差が NaN/inf、u が発散するなど）の場合は、
        u を grad にリセットして安全側に倒す。
        """
        u = u_init.copy()
        performed = 0
        for l in range(L):
            performed = l + 1

            # 残差 r = H u - g
            residual_vec = hessian @ u - grad
            residual_value = norm(residual_vec)

            # NaN や inf が出た場合は安全のため u を grad にリセット
            if not np.isfinite(residual_value):
                u = grad.copy()
                residual_value = norm(hessian @ u - grad)
                break

            # 残差が十分小さいなら内側ループを早期終了
            if rk_tol is not None and residual_value <= rk_tol:
                break

            # ランダム行列 Q_l^k のサンプリング（r_dim 行, dim 列）
            Q = sample_gaussian_matrix(r_dim, hessian.shape[0], 1.0 / r_dim)

            # qhq = Q H Q^T （対称化付き）
            qhq = symmetrize(Q @ hessian @ Q.T)
            rhs = Q @ residual_vec

            # (Q H Q^T) δ = Q residual を正則化付きで解く
            delta = solve_regularized_system(qhq, rhs, rk_reg)

            # u <- u - Q^T δ
            u = u - Q.T @ delta

            # u が NaN / inf を含む、またはノルムが大きくなりすぎた場合はリセット
            if not np.all(np.isfinite(u)) or norm(u) > y_cap:
                u = grad.copy()
                break

        # 最終的な RK 残差（ログ用）
        final_residual = rk_residual_fn(hessian, u, grad)
        return u, performed, final_residual

    def _construct_direction(
        self,
        *,
        u_vec: np.ndarray,
        hessian: np.ndarray,
        grad: np.ndarray,
        rs_rows: int,
        rs_reg: float,
    ) -> Tuple[np.ndarray, int]:
        """RS-RNM の更新方向 d_k を構成する（u_L^k ベース）。

        1. ベースベクトルとして u_vec（理論上は u_L^k）を用いる
           （ノルムが極端に小さい場合は fallback として勾配を利用）
        2. P_k の第 1 行を u_L^k^T とし、残り rs_rows 行は u_L^k と直交するように構成：
               P_k = [ u_L^k^T ; \tilde P_k ],   \tilde P_k u_L^k = 0
           直交補空間の構成には正規化した単位ベクトルのみを使い、
           第 1 行そのものはスケールを変えない。
        3. サブ問題 (P H P^T) δ = P g を正則化付きで解き、d = -P^T δ を得る
        """
        # ベースベクトル（P_k の第 1 行になる）
        base = u_vec.copy()
        base_norm = norm(base)

        # u_L^k のノルムが非常に小さい場合は、代わりに勾配を用いる
        if base_norm < 1e-12:
            base = grad / (norm(grad) + 1e-16)
            base_norm = norm(base)

        # P_k の第 1 行： u_L^k^T（または fallback の勾配ベクトル）
        first_row = base[None, :]

        # 直交補空間を構成する際に使う単位ベクトル
        unit_vec = base / (base_norm + 1e-16)

        if rs_rows > 0:
            # unit_vec（≒ u_L^k）に直交するランダム行列を構成し、正規直交化する
            rows = self._orth_complement(rs_rows, hessian.shape[0], unit_vec)
            P = np.vstack([first_row, rows])
        else:
            P = first_row

        # PH P^T と、右辺 P g を構成してサブ問題を解く
        phpt = symmetrize(P @ hessian @ P.T)
        rhs = P @ grad
        delta = solve_regularized_system(phpt, rhs, rs_reg)

        # d = -P^T δ
        direction = -P.T @ delta
        return direction, P.shape[0]

    @staticmethod
    def _orth_complement(rows: int, dim: int, unit_vec: np.ndarray) -> np.ndarray:
        """unit_vec に直交する rs_rows 本の正規直交ベクトルを生成する。

        まずガウス分布からランダム行列をサンプルし、各行を unit_vec に直交射影してから
        Gram-Schmidt 的な正規直交化（orthonormalize_rows）を行う。

        5 回まで試行しても十分なノルムの行が得られない場合は、
        単位行列の先頭 rows 行から同様に構成するフォールバックに切り替える。
        """
        attempts = 0
        variance = 1.0 / max(rows, 1)
        while attempts < 5:
            attempts += 1
            gaussian = sample_gaussian_matrix(rows, dim, variance)
            projected = project_rows_orthogonal(gaussian, unit_vec)
            row_norms = np.linalg.norm(projected, axis=1)
            if np.all(row_norms > 1e-10):
                return orthonormalize_rows(projected)

        # フォールバック：単位行列の行を使って直交補空間を構成
        identity = np.eye(dim)
        projected = project_rows_orthogonal(identity[:rows], unit_vec)
        return orthonormalize_rows(projected)

    @staticmethod
    def _rk_extras(
        *,
        t_k: float,
        subspace_rows: int,
        r_dim: int,
        L: int,
        rs_reg: float,
        rk_reg: float,
        rk_iters: int,
        rk_residual: float,
    ) -> Dict[str, float]:
        """ログ用の付加情報（RK/RS 関連のパラメータと実績値）をまとめて返す。"""
        return {
            "t_k": float(t_k),
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
        """方向ベクトルが降下方向になっているか確認し、必要なら修正する。

        grad^T d >= 0 の場合、d は降下方向ではないので、
        最低限の保険として -grad（最急降下方向）に差し替える。
        """
        if grad @ direction >= 0:
            return -grad
        return direction
