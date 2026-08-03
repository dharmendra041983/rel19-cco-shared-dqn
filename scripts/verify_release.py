#!/usr/bin/env python3
"""Validate the code-only public reproducibility package."""
from __future__ import annotations

import json
import math
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "environment.yml",
    "cco_experiments_corrected.py",
    "ppo_safety_scale_experiments.py",
    "robustness_experiments.py",
    "scripts/generate_figures.py",
    "scripts/aggregate_results.py",
    "scripts/verify_release.py",
    "results/original_five_seed_aggregated.json",
    "results/additional_ppo_safety_scale_aggregated.json",
    "results/robustness_aggregated.json",
    "results/robustness_seed0.json",
    "results/robustness_seed1.json",
    "results/robustness_seed2.json",
    "results/robustness_seed3.json",
    "results/robustness_seed4.json",
    "results/derived_strengthening_statistics.json",
    "figures/figure2_sample_efficiency.pdf",
    "figures/figure3_coexistence.pdf",
    "CITATION.cff",
    "RELEASE_VALIDATION.txt",
]

missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
if missing:
    raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))

for name in [
    "cco_experiments_corrected.py",
    "ppo_safety_scale_experiments.py",
    "robustness_experiments.py",
    "scripts/generate_figures.py",
    "scripts/aggregate_results.py",
    "scripts/verify_release.py",
]:
    py_compile.compile(str(ROOT / name), doraise=True)

def check_finite(obj, source: Path) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            check_finite(value, source)
    elif isinstance(obj, list):
        for value in obj:
            check_finite(value, source)
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"Non-finite value in {source}")

for path in (ROOT / "results").glob("*.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    check_finite(data, path)

original = json.loads(
    (ROOT / "results/original_five_seed_aggregated.json").read_text(encoding="utf-8")
)
for key in ["E1", "E2", "E3", "E4", "E5"]:
    if key not in original:
        raise SystemExit(f"Missing {key} in core aggregate")

additional = json.loads(
    (ROOT / "results/additional_ppo_safety_scale_aggregated.json").read_text(
        encoding="utf-8"
    )
)
for key in ["PPO", "Matched_DQN", "Safety", "Variable_Size", "Complexity"]:
    if key not in additional:
        raise SystemExit(f"Missing {key} in PPO/safety/scale aggregate")

print("PASS: structure, Python syntax, JSON readability, and key sections are valid.")
