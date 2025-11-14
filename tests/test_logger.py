from pathlib import Path

from analysis.logger import ExperimentLogger


def test_experiment_logger_filters_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "out.csv"
    logger = ExperimentLogger(csv_path=csv_path, csv_columns=["k", "f"])
    logger.log({"k": 0, "f": 1.23, "grad_norm": 9.0})
    logger.log({"k": 1, "f": 2.34})
    logger.flush()

    content = csv_path.read_text().strip().splitlines()
    assert content[0].split(",") == ["k", "f"]
    assert content[1] == "0,1.23"
    assert content[2] == "1,2.34"
