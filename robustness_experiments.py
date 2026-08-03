#!/usr/bin/env python3
"""IPCCC robustness, shield-sensitivity, and paired-CI experiments.

Trains the matched shared DQN and categorical PPO once per seed and adds:
  R1. Zero-shot traffic/propagation robustness on held-out geometries.
  R2. Shield-threshold sensitivity for delta in {1,2,3,5} percentage points.
  R3. Within-seed paired differences and Student-t 95% confidence intervals.

Run:
  python robustness_experiments.py --seeds 0,1,2,3,4

Dependencies: numpy, matplotlib only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

import cco_experiments_corrected as base
from ppo_safety_scale_experiments import (
    PPO, train_ppo, evaluate_policy, safety_audit
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "robustness_results"
OUT.mkdir(exist_ok=True)


def t95(mean: float, sample_std: float, n: int = 5):
    """Two-sided Student-t 95% CI using the sample standard deviation."""
    crit = 2.7764451051977987 if n == 5 else 1.96
    half = crit * sample_std / math.sqrt(n)
    return [float(mean - half), float(mean + half)]


def run_seed(seed: int, quick: bool = False):
    cfg = copy.deepcopy(base.CFG)
    cfg.update({
        "seed": seed,
        "ppo_hidden": 64,
        "ppo_lr": 3e-4,
        "ppo_gamma": cfg["gamma"],
        "ppo_gae_lambda": 0.95,
        "ppo_clip": 0.20,
        "ppo_entropy": 0.01,
        "ppo_vf_coef": 0.5,
        "ppo_epochs": 4 if quick else 10,
        "ppo_batch": 64,
    })
    np.random.seed(seed)
    episodes = 30 if quick else cfg["episodes_e1"]

    print(f"[seed {seed}] training matched DQN ({episodes} episodes)")
    dqn = base.DQN(seed)
    base.train(dqn, list(cfg["train_topos"]), episodes, cfg=cfg,
               curve=None, shield=True)

    print(f"[seed {seed}] training shared PPO ({episodes} episodes)")
    ppo = PPO(90_000 + seed, cfg)
    train_ppo(ppo, list(cfg["train_topos"]), episodes, cfg,
              shield=True, curve=None)

    dqn_policy = lambda e: dqn.act(e.features(), 0.0)
    ppo_policy = lambda e: ppo.act(e.features(), deterministic=True)

    scenarios = {
        "nominal": {},
        "high_load": {"n_ues": 300},
        "high_hotspot": {"hotspot_frac": 0.60},
        "severe_shadowing": {"shadow_sigma_db": 12.0},
    }
    robustness = {}
    for name, changes in scenarios.items():
        c = copy.deepcopy(cfg)
        c.update(changes)
        print(f"[seed {seed}] robustness: {name}")
        d = evaluate_policy(dqn_policy, list(c["test_topos"]), c,
                            shield=True, run_seed=seed + 120_000)
        p = evaluate_policy(ppo_policy, list(c["test_topos"]), c,
                            shield=True, run_seed=seed + 120_000)
        robustness[name] = {
            "dqn": d,
            "ppo": p,
            "paired_dqn_minus_ppo_coverage_pp":
                100.0 * (d["coverage"] - p["coverage"]),
            "paired_dqn_minus_ppo_mean_tput_mbps":
                d["mean_tput_mbps"] - p["mean_tput_mbps"],
        }

    thresholds = [0.01, 0.02, 0.03, 0.05]
    sensitivity = {}
    for delta in thresholds:
        c = copy.deepcopy(cfg)
        c["shield_cov_drop"] = delta
        key = f"delta_{int(round(100*delta))}pp"
        print(f"[seed {seed}] shield sensitivity: {key}")
        d_eval = evaluate_policy(dqn_policy, list(c["test_topos"]), c,
                                 shield=True, run_seed=seed + 130_000)
        p_eval = evaluate_policy(ppo_policy, list(c["test_topos"]), c,
                                 shield=True, run_seed=seed + 130_000)
        d_audit = safety_audit(dqn_policy, list(c["test_topos"]), c,
                               shield=True, run_seed=seed + 130_000)
        p_audit = safety_audit(ppo_policy, list(c["test_topos"]), c,
                               shield=True, run_seed=seed + 130_000)
        sensitivity[key] = {
            "delta_pp": 100.0 * delta,
            "dqn": {"evaluation": d_eval, "audit": d_audit},
            "ppo": {"evaluation": p_eval, "audit": p_audit},
            "paired_dqn_minus_ppo_coverage_pp":
                100.0 * (d_eval["coverage"] - p_eval["coverage"]),
        }

    return {
        "config": {"seed": seed, "episodes": episodes},
        "robustness": robustness,
        "shield_sensitivity": sensitivity,
    }


def aggregate(summaries):
    def walk(vals):
        first = vals[0]
        if isinstance(first, dict):
            return {k: walk([v[k] for v in vals]) for k in first
                    if k != "config"}
        if isinstance(first, (int, float, np.number)):
            arr = np.asarray(vals, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std())
            sample_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            return {
                "mean": mean,
                "std": std,
                "sample_std_for_ci": sample_std,
                "n_seeds": len(arr),
                "t95_ci": t95(mean, sample_std, len(arr)) if len(arr) == 5 else None,
                "per_seed": [float(x) for x in arr],
            }
        return first

    return {
        "config": {"seeds": [s["config"]["seed"] for s in summaries],
                   "episodes": summaries[0]["config"]["episodes"]},
        "robustness": walk([s["robustness"] for s in summaries]),
        "shield_sensitivity": walk([s["shield_sensitivity"] for s in summaries]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    summaries = []
    for seed in seeds:
        s = run_seed(seed, args.quick)
        summaries.append(s)
        with open(OUT / f"robustness_seed{seed}.json", "w") as f:
            json.dump(s, f, indent=2)

    agg = aggregate(summaries)
    with open(OUT / "robustness_aggregated.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))
    print(f"Saved: {OUT / 'robustness_aggregated.json'}")


if __name__ == "__main__":
    main()
