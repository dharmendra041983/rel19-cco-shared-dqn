# Reproduce the VCC revision from source

Paper: **Shared Deep Q-Learning for Multi-Cell Coverage Optimization: Transfer
and Action-Screening Limits**.

This public revision contains code and instructions. It does not distribute the
new seed-level results, experiment logs, trained checkpoints, or manuscript.
The full sequence below generates the revised experiment outputs locally; the
older files in `results/` are not inputs to these VCC experiments.

## Environment

Use Python 3.12 and the pinned versions in `requirements_vcc.txt`. The recorded
experiment environment used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, and Matplotlib 3.10.8. CPU execution is sufficient.

Run from the repository root. The commands below use a Linux shell. The native
ns-3 adapter requires Linux and a C++20-capable g++ compiler.

```bash
python -m pip install -r requirements_vcc.txt
OPENBLAS_NUM_THREADS=1 python scripts/validate_vcc.py
```

The numerical check creates `vcc_results/validation.json`. It checks MLP, graph,
and PPO gradients, an independent scalar radio calculation, a frozen-state
no-op, and the oracle acceptance invariant. It is not a physical-model validation.

## Main experiments and controls

```bash
for seed in 0 1 2 3 4; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python vcc_experiments.py --seed "$seed"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python vcc_safety.py --seed "$seed"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python vcc_ablations.py --seed "$seed"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python vcc_margin_sweep.py --seed "$seed"
done
```

The first script trains the matched main controllers for 250 episodes of 40
joint steps, evaluates the eight held-out conditions, and runs the paired
warm-start experiment. It saves `seed0.json` through `seed4.json` and the DQN,
graph-DQN, and sum-VDN checkpoints. Run it before the remaining scripts.

The gate audit writes `safety_seed*.json`. The normalization controls train mean
VDN and write `normalization_seed*.json` and its checkpoints. The margin sweep
writes `margin_seed*.json` and fitted gate forests as `gate_seed*.joblib`.
All these files are generated outputs, not files included in this code release.

Training uses topology IDs 0--5; warm-start validation uses 6--9; final transfer
evaluation uses 10--13. Seeds 0--4 vary training and user realizations. The same
topology identifiers are reused across layout families. No held-out metric
selects a hyperparameter, gate margin, or model checkpoint.

Five seeds may run in separate processes, with one process per seed. Do not run
two jobs writing the same seed. Main learning uses seven cells; size transfer
uses frozen policies. Mean-VDN training and gate calibration follow the settings
in the source configuration dictionaries.

## Native ns-3 propagation comparison

The official ns-3.44 wheel supplies native libraries and headers. The bridge
calls C++ directly; Python bindings and cppyy are not required. Libraries are
external dependencies and are not redistributed here.

```bash
python -m pip install --no-deps --target ns3_runtime ns3==3.44.post0
python validation/build_ns3_bridge.py --ns3-root ns3_runtime/ns3
for seed in 0 1 2 3 4; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python vcc_ns3_validation.py --seed "$seed"
done
OPENBLAS_NUM_THREADS=1 python validation/validate_ablations.py
```

The adapter writes `ns3_build.json`, including compiler and library/source
hashes. The runs write `ns3_seed*.json`; the final check writes
`ablation_validation.json`. Run the main, gate, normalization, and margin
experiments first because these checks consume their generated files.

## Regenerate statistics, figures, and table fragments

```bash
OPENBLAS_NUM_THREADS=1 python scripts/build_vcc_results.py
OPENBLAS_NUM_THREADS=1 python scripts/build_vcc_ablations.py
```

The builders create the `manuscript/` output directory if necessary. They write
vector-PDF plots and LaTeX table/text fragments there, plus `aggregated.json`
and `ablations_aggregated.json` in `vcc_results/`. The submitted manuscript and
its LaTeX source are not part of this public code release.

## Interpretation and historical results

The statistical replicates are five seed-level aggregates. Paired Student-t
intervals use four degrees of freedom and are conditional on four fixed
held-out geometries per condition. They are not population-level intervals
over deployed networks. Results can depend on numerical-library versions.

Native ns-3 comparisons substitute propagation kernels while retaining custom
antenna, association, interference, and scheduling components. They disable
shadowing in every paired condition and clamp native-model horizontal distances
below 10 m. Absolute KPIs are not field predictions. Action-screening margins
are empirical; calibration does not provide a guarantee after distribution shift.

The legacy `rsrp` variable is a received-power proxy, not standardized NR RSRP.
The main received-power model, user dynamics, and throughput remain abstractions.
All inputs are generated synthetically by the code.

Earlier archived IPCCC outputs under `results/` and `figures/` already existed
in the public repository and are retained separately. To reproduce the original
PPO implementation, use commit `336fba183de6beef5ad30d87f31446c96e6d125d`.
The corrected PPO code does not regenerate the historical PPO numbers.
