#!/usr/bin/env python3
"""Additional IPCCC experiments for the CCO paper.

Adds to cco_experiments_corrected.py:
  A1. Shared categorical PPO baseline under the same state/action/reward,
      online interaction budget, shield, topology split, and evaluation protocol.
  A2. Explicit action-level safety audit for DQN and PPO.
  A3. Zero-shot variable-size transfer from 7-cell training to 19-cell testing.
  A4. Parameter-count and batched inference-latency measurements.

Run from the folder containing cco_experiments_corrected.py:
  python3 cco_ipccc_additional_experiments.py --seeds 0,1,2,3,4

Dependencies: numpy, matplotlib only.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "cco_experiments_corrected.py"
spec = importlib.util.spec_from_file_location("cco_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

OUT = HERE / "additional_results"
FIG = HERE / "additional_figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


class PPO:
    """Shared categorical PPO with separate actor and critic MLPs."""

    def __init__(self, seed: int, cfg: dict):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        h = int(cfg.get("ppo_hidden", 64))
        self.actor = base.MLP([base.N_FEAT, h, h, base.N_ACT], self.rng)
        self.critic = base.MLP([base.N_FEAT, h, h, 1], self.rng)
        self.actor_opt = base.Adam(self.actor.params(), cfg.get("ppo_lr", 3e-4))
        self.critic_opt = base.Adam(self.critic.params(), cfg.get("ppo_lr", 3e-4))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.maximum(ez.sum(axis=1, keepdims=True), 1e-12)

    def distribution(self, states: np.ndarray):
        logits, _ = self.actor.forward(states)
        probs = self._softmax(logits)
        values, _ = self.critic.forward(states)
        return probs, values[:, 0]

    def act(self, states: np.ndarray, deterministic: bool = True) -> np.ndarray:
        probs, _ = self.distribution(states)
        if deterministic:
            return np.argmax(probs, axis=1)
        return np.asarray([
            self.rng.choice(base.N_ACT, p=p) for p in probs
        ], dtype=int)

    def sample(self, states: np.ndarray):
        probs, values = self.distribution(states)
        actions = np.asarray([
            self.rng.choice(base.N_ACT, p=p) for p in probs
        ], dtype=int)
        logp = np.log(np.maximum(probs[np.arange(len(actions)), actions], 1e-12))
        return actions, logp, values

    def update(self, states, actions, old_logp, advantages, returns):
        cfg = self.cfg
        clip = cfg.get("ppo_clip", 0.20)
        ent_coef = cfg.get("ppo_entropy", 0.01)
        vf_coef = cfg.get("ppo_vf_coef", 0.5)
        epochs = int(cfg.get("ppo_epochs", 10))
        batch = int(cfg.get("ppo_batch", 64))
        n = len(states)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(epochs):
            order = self.rng.permutation(n)
            for st in range(0, n, batch):
                idx = order[st:st + batch]
                s = states[idx]
                a = actions[idx]
                old = old_logp[idx]
                adv = advantages[idx]
                ret = returns[idx]
                bsz = max(1, len(idx))

                logits, actor_acts = self.actor.forward(s)
                probs = self._softmax(logits)
                chosen = np.maximum(probs[np.arange(bsz), a], 1e-12)
                logp = np.log(chosen)
                ratio = np.exp(logp - old)

                # Clipped-surrogate derivative. For positive advantages the
                # upper clip is active; for negative advantages the lower clip
                # is active. The clipped branch has zero derivative.
                active = np.where(adv >= 0.0,
                                  ratio <= 1.0 + clip,
                                  ratio >= 1.0 - clip)
                dloss_dlogp = np.where(active, -adv * ratio / bsz, 0.0)
                onehot = np.zeros_like(probs)
                onehot[np.arange(bsz), a] = 1.0
                dout_actor = dloss_dlogp[:, None] * (onehot - probs)

                # Entropy bonus: dH/dz_j = -p_j(log p_j + H).
                log_probs = np.log(np.maximum(probs, 1e-12))
                entropy = -np.sum(probs * log_probs, axis=1, keepdims=True)
                dout_actor += (ent_coef / bsz) * probs * (log_probs + entropy)
                grads = self.actor.backward(actor_acts, dout_actor)
                self.actor_opt.step(self.actor.params(), grads)

                values, critic_acts = self.critic.forward(s)
                err = values[:, 0] - ret
                dout_critic = (vf_coef * err / bsz)[:, None]
                grads_v = self.critic.backward(critic_acts, dout_critic)
                self.critic_opt.step(self.critic.params(), grads_v)


def train_ppo(agent: PPO, topo_seeds, episodes: int, cfg: dict,
              shield: bool = True, curve=None, log_every: int = 5):
    gamma = cfg.get("ppo_gamma", cfg["gamma"])
    lam = cfg.get("ppo_gae_lambda", 0.95)
    horizon = cfg["steps_per_ep"]

    for ep in range(episodes):
        ts = topo_seeds[ep % len(topo_seeds)]
        env = base.Env(ts, cfg=cfg)
        env.rng = np.random.default_rng(
            6_000_000 + 100_000 * cfg["seed"] + 1000 * int(ts) + ep)
        state = env.reset(randomize_init=True)

        states, actions, old_logp, rewards, values = [], [], [], [], []
        for _ in range(horizon):
            act, lp, val = agent.sample(state)
            next_state, reward, _, _ = env.step(act, shield=shield)
            states.append(state.copy())
            actions.append(act.copy())
            old_logp.append(lp.copy())
            rewards.append(float(reward))
            values.append(val.copy())
            state = next_state

        _, next_values = agent.distribution(state)
        states_a = np.asarray(states)              # [T,N,F]
        actions_a = np.asarray(actions)            # [T,N]
        old_logp_a = np.asarray(old_logp)           # [T,N]
        values_a = np.asarray(values)               # [T,N]
        rewards_a = np.asarray(rewards)[:, None]    # [T,1]

        adv = np.zeros_like(values_a)
        gae = np.zeros(env.n)
        nv = next_values
        for t in reversed(range(horizon)):
            delta = rewards_a[t, 0] + gamma * nv - values_a[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
            nv = values_a[t]
        ret = adv + values_a

        agent.update(states_a.reshape(-1, base.N_FEAT),
                     actions_a.reshape(-1),
                     old_logp_a.reshape(-1),
                     adv.reshape(-1),
                     ret.reshape(-1))

        completed = ep + 1
        if curve is not None and (completed == 1 or completed % log_every == 0):
            metrics = evaluate_policy(
                lambda e: agent.act(e.features(), deterministic=True),
                list(topo_seeds[:2]), cfg, n_init=1, steps=25,
                shield=shield, run_seed=cfg["seed"] + 50_000)
            curve.append((completed, metrics["coverage"],
                          metrics["mean_tput_mbps"]))
    return agent


def evaluate_policy(policy_fn: Callable, topo_seeds, cfg: dict,
                    n_init=None, steps=None, shield=True, run_seed=None):
    n_init = cfg["eval_n_init"] if n_init is None else n_init
    steps = cfg["eval_steps"] if steps is None else steps
    run_seed = cfg["seed"] if run_seed is None else run_seed
    rows = []
    for ts in topo_seeds:
        for init in range(n_init):
            env = base.Env(ts, cfg=cfg)
            base._reset_eval_env(env, run_seed, ts, init)
            kh = []
            for _ in range(steps):
                a = np.asarray(policy_fn(env), dtype=int)
                _, _, k, _ = env.step(a, shield=shield)
                kh.append(k)
            rows.append({k: float(np.mean([r[k] for r in kh])) for k in kh[0]})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def safety_audit(policy_fn: Callable, topo_seeds, cfg: dict,
                 shield: bool, n_init=None, steps=None, run_seed=None):
    """Audit candidate and final applied action violations on identical states."""
    n_init = cfg["eval_n_init"] if n_init is None else n_init
    steps = cfg["eval_steps"] if steps is None else steps
    run_seed = cfg["seed"] if run_seed is None else run_seed
    candidate_viol = applied_viol = rollbacks = total = 0
    max_candidate_drop = 0.0
    drops = []

    for ts in topo_seeds:
        for init in range(n_init):
            env = base.Env(ts, cfg=cfg)
            base._reset_eval_env(env, run_seed, ts, init)
            for _ in range(steps):
                # Exogenous dynamics first, matching the corrected shield.
                if cfg["dynamic"]:
                    env._churn()
                    env._compute()
                ref = env.kpis()["coverage"]
                snap_tilt = env.tilt.copy()
                snap_power = env.power.copy()
                a = np.asarray(policy_fn(env), dtype=int)
                env.prev_tilt = snap_tilt.copy()
                env.prev_power = snap_power.copy()
                env.apply(a)
                env._compute()
                cand = env.kpis()["coverage"]
                drop = max(0.0, ref - cand)
                drops.append(drop)
                max_candidate_drop = max(max_candidate_drop, drop)
                violates = cand < ref - cfg["shield_cov_drop"] - 1e-12
                candidate_viol += int(violates)

                if shield and violates:
                    env.tilt = snap_tilt
                    env.power = snap_power
                    env._compute()
                    rollbacks += 1
                applied = env.kpis()["coverage"]
                applied_viol += int(
                    applied < ref - cfg["shield_cov_drop"] - 1e-12)
                total += 1

    return {
        "candidate_violation_rate_pct": 100.0 * candidate_viol / max(1, total),
        "applied_violation_rate_pct": 100.0 * applied_viol / max(1, total),
        "max_candidate_drop_pp": 100.0 * max_candidate_drop,
        "mean_candidate_drop_pp": 100.0 * float(np.mean(drops)),
        "rollbacks_per_100_steps": 100.0 * rollbacks / max(1, total),
    }


def count_params(mlp) -> int:
    return int(sum(np.prod(p.shape) for p in mlp.params()))


def latency_benchmark(dqn, ppo, cfg: dict, iterations: int = 20000):
    rng = np.random.default_rng(999)
    x = rng.normal(size=(cfg["n_sites"], base.N_FEAT))

    def bench(fn):
        for _ in range(500):
            fn(x)
        samples = []
        chunks = 20
        per = max(1, iterations // chunks)
        for _ in range(chunks):
            t0 = time.perf_counter_ns()
            for _ in range(per):
                fn(x)
            dt = time.perf_counter_ns() - t0
            samples.append(dt / per / cfg["n_sites"] / 1000.0)
        return {
            "mean_us_per_cell": float(np.mean(samples)),
            "p95_us_per_cell": float(np.percentile(samples, 95)),
        }

    d = bench(lambda z: dqn.act(z, 0.0))
    p = bench(lambda z: ppo.act(z, deterministic=True))
    return {
        "dqn_parameters": count_params(dqn.q),
        "ppo_actor_parameters": count_params(ppo.actor),
        "ppo_critic_parameters": count_params(ppo.critic),
        "ppo_total_parameters": count_params(ppo.actor) + count_params(ppo.critic),
        "dqn_latency": d,
        "ppo_latency": p,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def curve_budget_metrics(curve):
    arr = np.asarray(curve, dtype=float)
    out = {}
    for budget in (5, 10, 25, 50):
        sel = arr[arr[:, 0] <= budget]
        if len(sel):
            out[f"coverage_at_{budget}ep"] = float(sel[-1, 1])
    trap = getattr(np, "trapezoid", None) or np.trapz
    out["coverage_auc"] = float(
        trap(arr[:, 1], arr[:, 0]) / max(1e-12, arr[-1, 0] - arr[0, 0]))
    return out


def run_seed(seed: int, quick: bool):
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

    print(f"[seed {seed}] training matched shared DQN")
    dqn = base.DQN(seed)
    base.train(dqn, list(cfg["train_topos"]), episodes, cfg=cfg,
               curve=None, shield=True)

    print(f"[seed {seed}] training shared PPO")
    ppo_curve = []
    ppo = PPO(90_000 + seed, cfg)
    train_ppo(ppo, list(cfg["train_topos"]), episodes, cfg,
              shield=True, curve=ppo_curve, log_every=5)

    ppo_seen = evaluate_policy(
        lambda e: ppo.act(e.features(), deterministic=True),
        list(cfg["train_topos"]), cfg, shield=True)
    ppo_unseen = evaluate_policy(
        lambda e: ppo.act(e.features(), deterministic=True),
        list(cfg["test_topos"]), cfg, shield=True)
    dqn_seen = evaluate_policy(
        lambda e: dqn.act(e.features(), 0.0),
        list(cfg["train_topos"]), cfg, shield=True)
    dqn_unseen = evaluate_policy(
        lambda e: dqn.act(e.features(), 0.0),
        list(cfg["test_topos"]), cfg, shield=True)

    print(f"[seed {seed}] auditing safety")
    safety = {
        "dqn_no_shield": safety_audit(
            lambda e: dqn.act(e.features(), 0.0),
            list(cfg["train_topos"]), cfg, shield=False),
        "dqn_shielded": safety_audit(
            lambda e: dqn.act(e.features(), 0.0),
            list(cfg["train_topos"]), cfg, shield=True),
        "ppo_no_shield": safety_audit(
            lambda e: ppo.act(e.features(), deterministic=True),
            list(cfg["train_topos"]), cfg, shield=False),
        "ppo_shielded": safety_audit(
            lambda e: ppo.act(e.features(), deterministic=True),
            list(cfg["train_topos"]), cfg, shield=True),
    }

    # Zero-shot variable-size transfer. User count is scaled to keep the
    # nominal UE-per-cell density close to the seven-cell setting.
    cfg19 = copy.deepcopy(cfg)
    cfg19["n_sites"] = 19
    cfg19["n_ues"] = int(round(cfg["n_ues"] * 19 / 7))
    large_topos = (110, 111) if quick else (110, 111, 112, 113)
    print(f"[seed {seed}] evaluating 19-cell zero-shot transfer")
    dqn_19 = evaluate_policy(
        lambda e: dqn.act(e.features(), 0.0), large_topos, cfg19,
        n_init=2 if quick else cfg19["eval_n_init"], shield=True,
        run_seed=seed + 90_000)
    ppo_19 = evaluate_policy(
        lambda e: ppo.act(e.features(), deterministic=True), large_topos, cfg19,
        n_init=2 if quick else cfg19["eval_n_init"], shield=True,
        run_seed=seed + 90_000)

    complexity = latency_benchmark(dqn, ppo, cfg,
                                   iterations=3000 if quick else 20000)

    return {
        "config": {
            "seed": seed,
            "episodes": episodes,
            "train_topos": list(cfg["train_topos"]),
            "test_topos": list(cfg["test_topos"]),
            "large_topos": list(large_topos),
            "large_n_sites": 19,
            "large_n_ues": cfg19["n_ues"],
        },
        "PPO": {
            "seen": ppo_seen,
            "zero_shot_7cell": ppo_unseen,
            "seen_to_unseen_gap_pp": 100.0 * (
                ppo_seen["coverage"] - ppo_unseen["coverage"]),
            "learning": curve_budget_metrics(ppo_curve),
        },
        "Matched_DQN": {
            "seen": dqn_seen,
            "zero_shot_7cell": dqn_unseen,
            "seen_to_unseen_gap_pp": 100.0 * (
                dqn_seen["coverage"] - dqn_unseen["coverage"]),
        },
        "Safety": safety,
        "Variable_Size": {
            "dqn_19cell": dqn_19,
            "ppo_19cell": ppo_19,
            "dqn_7to19_drop_pp": 100.0 * (
                dqn_unseen["coverage"] - dqn_19["coverage"]),
            "ppo_7to19_drop_pp": 100.0 * (
                ppo_unseen["coverage"] - ppo_19["coverage"]),
        },
        "Complexity": complexity,
        "ppo_curve": ppo_curve,
    }


def aggregate(summaries):
    def walk(vals):
        first = vals[0]
        if isinstance(first, dict):
            out = {}
            for k in first:
                if k == "config" or k == "ppo_curve" or k in ("python", "platform", "numpy"):
                    continue
                out[k] = walk([v[k] for v in vals])
            return out
        if isinstance(first, (int, float, np.number)):
            arr = np.asarray(vals, dtype=float)
            return {"mean": float(arr.mean()), "std": float(arr.std()),
                    "n_seeds": len(arr)}
        return first

    out = {"config": summaries[0]["config"], "software": {
        "python": summaries[0]["Complexity"]["python"],
        "platform": summaries[0]["Complexity"]["platform"],
        "numpy": summaries[0]["Complexity"]["numpy"],
    }}
    for key in ("PPO", "Matched_DQN", "Safety", "Variable_Size", "Complexity"):
        out[key] = walk([s[key] for s in summaries])
    return out


def make_figures(summaries, agg):
    # PPO learning curve across seeds.
    curves = [np.asarray(s["ppo_curve"], dtype=float) for s in summaries]
    L = min(len(c) for c in curves)
    st = np.stack([c[:L] for c in curves])
    ep = st[0, :, 0]
    m = st[:, :, 1].mean(0) * 100
    sd = st[:, :, 1].std(0) * 100
    plt.figure(figsize=(5.2, 3.2))
    plt.plot(ep, m, marker="o", ms=3, label="Shared PPO")
    plt.fill_between(ep, m - sd, m + sd, alpha=0.18)
    plt.xlabel("Completed online episodes")
    plt.ylabel("Coverage [%]")
    plt.title(f"PPO learning; mean $\\pm$ 1 std over {len(summaries)} seeds",
              fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "ppo_learning.pdf")
    plt.savefig(FIG / "ppo_learning.png", dpi=200)
    plt.close()

    # Seen/held-out/19-cell comparison.
    labels = ["DQN\n7-cell", "DQN\n19-cell", "PPO\n7-cell", "PPO\n19-cell"]
    means = [
        agg["Matched_DQN"]["zero_shot_7cell"]["coverage"]["mean"] * 100,
        agg["Variable_Size"]["dqn_19cell"]["coverage"]["mean"] * 100,
        agg["PPO"]["zero_shot_7cell"]["coverage"]["mean"] * 100,
        agg["Variable_Size"]["ppo_19cell"]["coverage"]["mean"] * 100,
    ]
    stds = [
        agg["Matched_DQN"]["zero_shot_7cell"]["coverage"]["std"] * 100,
        agg["Variable_Size"]["dqn_19cell"]["coverage"]["std"] * 100,
        agg["PPO"]["zero_shot_7cell"]["coverage"]["std"] * 100,
        agg["Variable_Size"]["ppo_19cell"]["coverage"]["std"] * 100,
    ]
    plt.figure(figsize=(5.2, 3.2))
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.ylabel("Zero-shot coverage [%]")
    plt.tight_layout()
    plt.savefig(FIG / "variable_size_transfer.pdf")
    plt.savefig(FIG / "variable_size_transfer.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    summaries = []
    for seed in seeds:
        summary = run_seed(seed, args.quick)
        summaries.append(summary)
        with open(OUT / f"additional_seed{seed}.json", "w") as f:
            json.dump(summary, f, indent=2)

    agg = aggregate(summaries)
    with open(OUT / "additional_aggregated.json", "w") as f:
        json.dump(agg, f, indent=2)
    make_figures(summaries, agg)
    print(json.dumps(agg, indent=2))
    print(f"\nSaved: {OUT / 'additional_aggregated.json'}")


if __name__ == "__main__":
    main()
