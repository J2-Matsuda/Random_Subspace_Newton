# ARSN large-scale numerical experiments (RK + Diagonal Shift)

This repository provides a matrix-free implementation of Anchored Random Subspace Newton (ARSN) with:
- inner solver: Randomized Kaczmarz (RK)
- regularizer: Diagonal Shift (Fuji et al.)

The implementation is faithful to the algorithm and avoids explicit Hessian formation by using HVPs only.
Gaussian sketches are handled in operator mode by default to reduce memory usage.

## Directory structure

```
project_root/
  run_experiment.py
  plot_results.py
  generate_data.py
  requirements.txt
  README.md
  configs/
    example_quadratic.yml
    example_logistic.yml
  arsn/
    __init__.py
    main.py
    inner/
      __init__.py
      rk.py
    regularizer/
      __init__.py
      diag_shift.py
    problems/
      __init__.py
      base.py
      quadratic.py
      logistic.py
    linalg/
      __init__.py
      gaussian_sketch.py
      linear_operator_utils.py
    utils/
      __init__.py
      config.py
      logging.py
      timer.py
```

## Requirements

```
pip install -r requirements.txt
```

Note: SciPy must be built against your installed NumPy. If you see an `_ARRAY_API` error,
install a compatible NumPy/SciPy pair (e.g., `numpy<2` with older SciPy, or upgrade SciPy for NumPy 2).

## Quick start

Generate data (optional):

```
python generate_data.py quadratic --out data/quadratic_spd.npz --n 200 --cond 100
python generate_data.py logistic --out data/logistic_synth.npz --m 2000 --n 200 --reg_lambda 1e-3
```

Generate data via YAML (multiple datasets in one run):

```
python generate_data.py --config configs/data_tasks.yml
```

Logistic data generation (default behavior):
- X ~ N(0, x_scale^2 I)
- beta ~ N(0, beta_scale^2 I) unless explicitly provided
- p = sigmoid(X beta)
- y ~ Bernoulli(p) (labels in {0,1} if labels01=true, otherwise {-1,1})

Run ARSN:

```
python run_experiment.py --config configs/example_quadratic.yml
python run_experiment.py --config configs/example_logistic.yml
```

Run RSRNM (no inner RK; P=S only):

```
python run_experiment.py --config configs/example_rsrnm_quadratic.yml
python run_experiment.py --config configs/example_rsrnm_logistic.yml
```

Outputs are saved under `outputs/<run_name>/`:
- `result.csv`
- `config_copy.yml`
- `summary.json`

## CSV columns

The default CSV includes the following columns (1 row per outer iteration):

- iter
- f
- grad_norm
- step_norm
- gTd
- alpha
- armijo_iters
- time_iter_sec
- time_cum_sec
- hvp_calls_iter
- hvp_calls_cum
- yk_resid_rel
- inner_T
- inner_r
- inner_minres_total_iters
- eta
- lambda_min_PHPT
- Lambda_shift
- seed_outer
- seed_Sk
- seed_Rk_base

You can restrict columns via `logging.record_columns` in the config.

## Plotting

```
python plot_results.py --csv outputs/example_quadratic/result.csv --x iter --y grad_norm --logy --out grad_norm.png
python plot_results.py --csv outputs/example_quadratic/result.csv --x time_cum_sec --y grad_norm step_norm --logy --out metrics.png
```

Plot via YAML (multiple figures, multiple CSVs per figure):

```
python plot_results.py --config configs/plot_tasks.yml
```

Notes:
- `plot.series.csv` accepts glob patterns (e.g., `outputs/*/result.csv`).
- `label_from: basename_dir` uses the run directory name as the legend label.
- `plot.series.label` overrides the legend label for a series.
- `plot.figures[].x_label` / `y_label` (or `xlabel` / `ylabel`) set axis display names.
- `plot.figures[].y_labels` can map y-column names to legend display names.
- If a column is missing in a CSV, that series is skipped with a warning.

## Config reference (YAML)

Key sections:

```
algorithm:
  name: arsn | rsrnm

problem:
  name: quadratic | logistic
  data_path: path/to/data.npz
  reg_lambda: 1.0e-3   # optional override for logistic

arsn:
  max_iter: 200
  tol_grad: 1.0e-6
  s: 30
  beta: 1.0e-4
  tau: 0.5
  alpha0: 1.0
  max_ls_iters: 50
  seed: 0
  verbose: true
  print_every: 5

inner:
  method: rk
  rk:
    T: 10
    r: 20
    ridge: 1.0e-12
    minres_tol: 1.0e-6
    minres_maxit: 200
    warm_start: true

regularizer:
  method: diag_shift
  diag_shift:
    c1: 2.0
    c2: 1.0
    gamma: 1.0

sketch:
  mode: operator | explicit
  block_size: 256
  dtype: float64

output:
  dir: outputs
  run_name: example_run
  csv_name: result.csv

logging:
  record_columns: null
```

RSRNM-specific section (used when `algorithm.name: rsrnm`):

```
rsrnm:
  max_iter: 200
  tol_grad: 1.0e-6
  s: 30
  beta: 1.0e-4
  tau: 0.5
  alpha0: 1.0
  max_ls_iters: 50
  seed: 0
```

## Notes on large-scale implementation

- HVP-only: `Problem.hvp(x, v)` is used everywhere; Hessians are never formed.
- Gaussian sketches are operator-based by default.
- RK inner linear systems use MINRES with a matrix-free LinearOperator.
- The subspace system (s+1) x (s+1) is dense and solved directly.

## RSRNM vs ARSN

RSRNM removes the inner RK step and uses only the random subspace sketch:
P = S instead of P = [y_hat; S]. The rest of the pipeline (Diagonal Shift + Armijo) is unchanged.
