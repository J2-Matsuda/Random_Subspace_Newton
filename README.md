# Random Subspace Newton Experiments

Minimal research framework for evaluating Random Subspace Regularised Newton (RSRNM) and the proposed RK+RSRNM method with Armijo line search.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running an experiment

```bash
python main.py --config input/rsrnm_min.yml
python main.py --config input/rk_rsrnm_min.yml
```

## Plotting multiple runs

Use `scripts/output_customizer.py` with a YAML config (see `output/grad_compare.yml`) to overlay metrics from several CSV logs:

```bash
python scripts/output_customizer.py --config output/grad_compare.yml
python scripts/cpu_vs_grad.py --config output/cpu_grad.yml
```

Each run stores its outputs under `log/<config_name>_<timestamp>/` (e.g., `log/rk_rsrnm_min_202511140934/`), containing the CSV/JSONL logs, a copy of the input YAML, and the convergence plot for that experiment.  
You can control what goes into each artifact via the YAML logging section, e.g.

```yaml
logging:
  include_timestamp: true
  save_plot: true
  csv_metrics: ["k", "f", "grad_norm", "t_k"]
  plot_metrics: ["grad_norm", "f"]
  debug: false          # set true to enable rk_rsrnm_debug wrapping
  debug_csv: "log/run_debug.csv"
```

Only the listed columns are kept in the CSV/JSONL, while `plot_metrics` drives the convergence figure.
Set `include_timestamp` to `false` if you want the log folder/file names to match `experiment_name` exactly without the timestamp suffix.
If you also add `debug: true` (only for `rk_rsrnm`), the run will switch to the `rk_rsrnm_debug` variant, printing per-iteration diagnostics and saving them to `debug_csv` (default: `<logdir>/<experiment_name>_debug.csv`).
