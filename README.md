# rel19-cco-shared-dqn

Code and reproducibility materials accompanying the manuscript
**“Deployment-Aware Shared DQN for Rel-19 Coverage and Capacity
Optimization: Transfer, Safety, and SON Coexistence.”**

The manuscript is under review. This repository intentionally contains the
code, machine-readable results, and figure-generation materials only; it does
not include the submitted paper PDF.

All network realizations and KPI samples are generated synthetically by the
included simulator. No proprietary operator, customer, employer, or
production-network data is included.

## Contents

```text
rel19-cco-shared-dqn/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── cco_experiments_corrected.py
├── ppo_safety_scale_experiments.py
├── robustness_experiments.py
├── scripts/
│   ├── generate_figures.py
│   ├── aggregate_results.py
│   └── verify_release.py
├── results/
│   ├── original_five_seed_aggregated.json
│   ├── additional_ppo_safety_scale_aggregated.json
│   ├── robustness_aggregated.json
│   ├── robustness_seed0.json
│   ├── robustness_seed1.json
│   ├── robustness_seed2.json
│   ├── robustness_seed3.json
│   ├── robustness_seed4.json
│   └── derived_strengthening_statistics.json
├── figures/
│   ├── figure2_sample_efficiency.pdf
│   └── figure3_coexistence.pdf
├── CITATION.cff
└── RELEASE_VALIDATION.txt
```

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- CPU execution; no GPU is required

Install with pip:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate rel19-cco-paper
```

## Full experiment commands

Run from the repository root:

```bash
python cco_experiments_corrected.py --exp all --seeds 0,1,2,3,4
python ppo_safety_scale_experiments.py --seeds 0,1,2,3,4
python robustness_experiments.py --seeds 0,1,2,3,4
```

The archived JSON files in `results/` are the machine-readable outputs used
for the reported five-seed analyses.

## Regenerate Figures 2 and 3

```bash
python scripts/generate_figures.py
```

The script reads the archived aggregate JSON files in `results/` and writes
the updated vector PDFs to `figures/`.

## Verify the package

```bash
python scripts/verify_release.py
```

## Notes on timing

The repository does not claim a wall-clock training duration. Policy-forward
latency is not end-to-end control-loop latency; it excludes simulator,
telemetry, feature-construction, communication, and actuation delays.

## License and citation

The code is released under the BSD 3-Clause License. Citation metadata is in
`CITATION.cff`.

Repository:

```text
https://github.com/dharmendra041983/rel19-cco-shared-dqn
```
