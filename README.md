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

Logs (CSV + JSONL) and convergence plots are stored under `log/<experiment_name>/`.
