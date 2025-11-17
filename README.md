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

Each run stores its outputs under `log/<config_name>_<timestamp>/` (e.g., `log/rk_rsrnm_min_202511140934/`), containing the CSV/JSONL logs, a copy of the input YAML, and the convergence plot for that experiment.  
You can control what goes into each artifact via the YAML logging section, e.g.

```yaml
logging:
  save_plot: true
  csv_metrics: ["k", "f", "grad_norm", "t_k"]
  plot_metrics: ["grad_norm", "f"]
```

Only the listed columns are kept in the CSV/JSONL, while `plot_metrics` drives the convergence figure.
