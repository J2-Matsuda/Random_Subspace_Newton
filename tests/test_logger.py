from pathlib import Path

from analysis.logger import ExperimentLogger


def test_experiment_logger_filters_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "out.csv"
    logger = ExperimentLogger(csv_path=csv_path, csv_columns=["k", "f"])
    logger.log({"k": 0, "f": 1.23, "grad_norm": 9.0})
    logger.log({"k": 1, "f": 2.34})
    logger.flush()

    content = csv_path.read_text().strip().splitlines()
    header = content[0].split(",")
    assert header == ["k", "f", "cpu_time", "cpu_time_sum"]
    first = content[1].split(",")
    second = content[2].split(",")
    assert first[:2] == ["0", "1.23"]
    assert second[:2] == ["1", "2.34"]
    assert float(first[2]) >= 0.0
    assert float(second[2]) >= 0.0
    assert float(first[3]) >= float(first[2])
    assert float(second[3]) >= float(first[3])
