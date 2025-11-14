"""数値実験（RK / RSRNM 系）のコマンドライン実行エントリポイント。

- YAML で与えた実験設定を読み込み
- 問題・手法をレジストリから組み立て
- CSV / JSONL にログを出力し、必要なら収束図を保存
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

from analysis.logger import ExperimentLogger
from analysis.plot_convergence import plot_from_csv
from config.registries import build_algorithm, build_problem
from utils.io import ensure_dir, load_yaml, timestamp_tag


# ------------------------------------------------------------
# 引数処理
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解釈する。

    Returns:
        argparse.Namespace: --config と --logdir を含む名前空間
    """
    parser = argparse.ArgumentParser(
        description="RK/RSRNM 系の数値実験を実行します（YAML で問題・手法を指定）。"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="実験設定 YAML のパス（例: input/rk_rsrnm_min.yml）",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="log",
        help="ログ（CSV/JSONL/PNG）を保存するディレクトリ（既定: log）",
    )
    return parser.parse_args()


# ------------------------------------------------------------
# 実行ユーティリティ
# ------------------------------------------------------------
def prepare_paths(
    log_root: Path, folder_name: str, file_stem: str
) -> tuple[Path, Path, Path]:
    """ログ出力用のパスを用意する。

    Args:
        log_root: ルートとなるログディレクトリ
        experiment_name: 実験名（サブディレクトリ名やファイル名の接頭辞に使用）

    Returns:
        (log_base, csv_path, jsonl_path)
    """
    log_base = ensure_dir(log_root / folder_name)
    csv_path = log_base / f"{file_stem}.csv"
    jsonl_path = log_base / f"{file_stem}.jsonl"
    return log_base, csv_path, jsonl_path


def pretty_result_line(name: str, result) -> str:
    """最後に表示するサマリ文字列を整形する。"""
    return (
        f"[{name}] "
        f"f={result.f:.3e}, ||grad||={result.grad_norm:.3e}, "
        f"iters={result.iterations}, converged={result.converged}"
    )


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- 設定読み込み ---
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = load_yaml(config_path)
    except Exception as e:
        print(f"[ERROR] 設定ファイルの読み込みに失敗しました: {args.config}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # --- 乱数シード設定（再現性のため） ---
    seed = int(config.get("seed", 0))
    np.random.seed(seed)

    # --- 実験名とログパスの準備 ---
    timestamp = timestamp_tag()
    experiment_name = config.get("experiment_name", f"exp_{timestamp}")
    config_stem = Path(args.config).stem
    run_folder = f"{config_stem}_{timestamp}"
    log_base, csv_path, jsonl_path = prepare_paths(
        Path(args.logdir), run_folder, experiment_name
    )
    try:
        shutil.copy2(config_path, log_base / config_path.name)
    except Exception:
        print(
            "[WARN] 設定ファイルのコピーに失敗しましたが、実験は継続します。",
            file=sys.stderr,
        )
        traceback.print_exc()

    # --- 問題・手法の構築 ---
    try:
        problem = build_problem(config["problem"])
    except KeyError as e:
        print("[ERROR] config['problem'] セクションが不足または不正です。", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    try:
        algorithm = build_algorithm(config["algorithm"])
    except KeyError as e:
        print("[ERROR] config['algorithm'] セクションが不足または不正です。", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # --- ロガーの初期化と実行 ---
    logging_cfg = config.get("logging", {})
    csv_metrics = logging_cfg.get("csv_metrics")
    logger = ExperimentLogger(
        csv_path=csv_path, jsonl_path=jsonl_path, csv_columns=csv_metrics
    )
    result = algorithm.run(problem, logger)

    # --- 収束図の保存（任意） ---
    if logging_cfg.get("save_plot", False):
        metrics = logging_cfg.get("plot_metrics", ["grad_norm"])
        plot_path = log_base / f"{experiment_name}_convergence.png"
        try:
            plot_from_csv(csv_path, metrics, plot_path)
        except Exception:
            # 図生成は失敗しても実験自体は継続可能にしておく
            print("[WARN] 収束図の生成に失敗しました（実験結果自体は保存済みです）。", file=sys.stderr)
            traceback.print_exc()

    # --- サマリの標準出力 ---
    print(pretty_result_line(experiment_name, result))
    print(f"ログ保存先: {log_base.resolve()}")


if __name__ == "__main__":
    main()
