#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rel-19 aligned AI/ML Coverage & Capacity Optimization (CCO) -- experiment suite.

Implements a lightweight 3GPP-style multi-cell system-level simulator and a
shared-policy DQN CCO agent with:
  * graph-lite state (one-hop mean-aggregated neighbour features),
  * a safety/rollback shield (guarantee-by-design KPI floor),
  * offline warm-start (fitted Q-iteration on logged data),
  * a coexisting energy-saving (ES) SON loop for stability experiments.

Experiments (mapped to the paper result sections):
  E1  KPI comparison vs baselines
  E2  Generalization to held-out topologies
  E3  Sample efficiency (offline warm-start)
  E4  Stability under SON coexistence
  E5  One-hop neighbour-context ablation

Usage:
  python3 cco_experiments_corrected.py --exp all            # full run (~15-40 min on a laptop)
  python3 cco_experiments_corrected.py --exp e1 --quick     # fast smoke test
Outputs:
  results/summary.json                 # single-seed summary
  results/summary_seed<N>.json         # multi-seed per-seed summaries
  results/summary_aggregated.json      # multi-seed aggregate
  figures/*.png                        # generated figures

Dependencies: numpy, matplotlib only.
"""

import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CFG = dict(
    # Radio / deployment (3GPP UMa-flavoured, simplified)
    n_sites=7,              # hex layout: 1 centre + 6 ring
    isd=500.0,              # inter-site distance [m]
    bs_height=25.0,
    ue_height=1.5,
    fc_ghz=3.5,
    bw_hz=20e6,
    noise_figure_db=9.0,
    ant_gain_dbi=8.0,
    theta_3db=10.0,         # vertical HPBW [deg]
    sla_db=20.0,            # side-lobe attenuation limit
    shadow_sigma_db=8.0,
    n_ues=200,
    hotspot_frac=0.4,
    n_hotspots=3,
    hotspot_sigma=80.0,
    # Topology diversity: each topology seed perturbs the nominal seven-site
    # layout while preserving a realistic connected deployment.
    topology_scale_jitter=0.12,   # +/- 12% ISD scaling
    topology_site_jitter=0.08,    # per-site position jitter as fraction of ISD
    neighbor_radius_factor=1.35,  # relative to median nearest-neighbor distance
    # Controllable parameters
    tilt_min=0.0, tilt_max=15.0, tilt_step=1.0, tilt_init=6.0,
    p_min=40.0,  p_max=46.0,  p_step=1.0,  p_init=43.0,
    # KPI thresholds
    sinr_edge_db=0.0,
    rsrp_cov_dbm=-104.0,
    # Reward
    w_cov=10.0, w_se=0.5, w_churn=0.02,
    # RL
    hidden=64,
    gamma=0.8,
    lr=1e-3,
    batch=64,
    buffer_size=120_000,
    warmup=500,
    target_every=400,
    eps_start=1.0, eps_end=0.05, eps_frac=0.6,
    steps_per_ep=40,
    episodes_e1=250,
    seed=0,
    eval_n_init=3,
    eval_steps=40,
    # Safety shield
    shield_cov_drop=0.02,       # rollback if coverage drops > 2 pp in one step
    # Few-shot fine-tuning (E2): reduced lr + low fixed exploration, so
    # adapting to the target topologies does not overwrite the transferred
    # policy outright.
    ft_lr_scale=0.2,
    ft_eps=0.05,
    # Coexisting energy-saving loop (E4). The "mild" regime is a
    # slow, gentle loop; the "adversarial" regime cycles fast with large
    # power steps and is the regime in which SON conflict actually occurs.
    # Both are reported -- the coordinator can only help where there is a
    # conflict to resolve, and saying so explicitly is part of the claim.
    es_mild=dict(period=10, frac=0.4, delta=3.0),
    es_adversarial=dict(period=3, frac=0.7, delta=6.0),
    # Dynamics (user churn) -- makes stability metrics meaningful
    dynamic=True,
    churn_frac=0.02,            # fraction of UEs re-dropped per step
    jitter_m=5.0,               # per-step positional jitter of hotspot UEs
    # Topology seeds
    train_topos=(0, 1, 2, 3, 4, 5),
    test_topos=(10, 11, 12, 13),
)

ACTIONS = ["noop", "tilt-", "tilt+", "pow-", "pow+"]
N_ACT = len(ACTIONS)
N_FEAT = 10

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# Per-seed curve/scalar store, used to redraw figures with mean +/- std
# bands across seeds instead of showing a single run. Keys map to lists
# with one entry per seed.
CURVE_STORE = {}


# ----------------------------------------------------------------------------
# Topology & environment
# ----------------------------------------------------------------------------
def hex_sites(n_sites, isd):
    """Nominal one-centre plus hexagonal-ring site layout."""
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = math.pi / 3 * k
        pts.append((isd * math.cos(a), isd * math.sin(a)))
    if n_sites > 7:  # second ring (up to 19)
        for k in range(6):
            a = math.pi / 3 * k
            pts.append((2 * isd * math.cos(a), 2 * isd * math.sin(a)))
            a2 = a + math.pi / 6
            pts.append((math.sqrt(3) * isd * math.cos(a2),
                        math.sqrt(3) * isd * math.sin(a2)))
    return np.asarray(pts[:n_sites], dtype=float)


def make_topology(topo_seed, cfg=CFG):
    """Create a deterministic but genuinely distinct topology per seed.

    Training and held-out topology IDs therefore differ in site geometry, not
    merely in user locations or shadowing. A global scale/rotation and small
    independent site perturbations retain the seven-site deployment structure
    while changing distances and adjacency geometry.
    """
    rng = np.random.default_rng(70_000 + int(topo_seed))
    pts = hex_sites(cfg["n_sites"], cfg["isd"])
    scale = 1.0 + rng.uniform(-cfg["topology_scale_jitter"],
                              cfg["topology_scale_jitter"])
    ang = rng.uniform(-math.pi / 12, math.pi / 12)
    rot = np.array([[math.cos(ang), -math.sin(ang)],
                    [math.sin(ang),  math.cos(ang)]])
    pts = (pts @ rot.T) * scale
    jitter = rng.normal(0.0, cfg["topology_site_jitter"] * cfg["isd"],
                        size=pts.shape)
    if len(jitter):
        jitter[0] = 0.0  # keep the reference/central site at the origin
    return pts + jitter


class Env:
    """Topology-specific, dynamic-user multi-cell downlink environment."""

    def __init__(self, topo_seed, cfg=CFG, mask_neighbors=False):
        self.cfg = cfg
        self.mask_neighbors = mask_neighbors
        self.topo_seed = int(topo_seed)
        self.rng = np.random.default_rng(1000 + self.topo_seed)
        self.sites = make_topology(self.topo_seed, cfg)
        self.n = len(self.sites)
        self.area_r = max(cfg["isd"] * 2.2,
                          float(np.max(np.linalg.norm(self.sites, axis=1))
                                + 1.1 * cfg["isd"]))
        # Hotspot centres are fixed for a topology but user realizations vary.
        ang = self.rng.uniform(0, 2 * math.pi, cfg["n_hotspots"])
        rad = self.rng.uniform(0.25, 0.9, cfg["n_hotspots"]) * self.area_r
        self.hot_centres = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)

        # True one-hop adjacency. The threshold scales with the realized
        # topology rather than using 2.4*ISD, which previously connected every
        # cell to every other cell in the seven-site layout.
        d = np.linalg.norm(self.sites[:, None, :] - self.sites[None, :, :],
                           axis=-1)
        d_no_self = d.copy()
        np.fill_diagonal(d_no_self, np.inf)
        nn_scale = float(np.median(np.min(d_no_self, axis=1)))
        radius = cfg["neighbor_radius_factor"] * nn_scale
        self.neigh = []
        for i in range(self.n):
            nb = np.where((d[i] > 1.0) & (d[i] <= radius))[0]
            if len(nb) == 0 and self.n > 1:
                nb = np.array([int(np.argmin(d_no_self[i]))])
            self.neigh.append(nb)
        self.neighbor_radius_m = radius

        self.noise_mw = 10 ** ((-174 + 10 * math.log10(cfg["bw_hz"])
                                + cfg["noise_figure_db"]) / 10)
        self.reset()

    # ---- users -------------------------------------------------------------
    def _drop_users(self, m):
        cfg = self.cfg
        n_hot = int(round(m * cfg["hotspot_frac"]))
        n_uni = m - n_hot
        r = self.area_r * np.sqrt(self.rng.uniform(0, 1, n_uni))
        a = self.rng.uniform(0, 2 * math.pi, n_uni)
        uni = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        idx = self.rng.integers(0, cfg["n_hotspots"], n_hot)
        hot = (self.hot_centres[idx]
               + self.rng.normal(0, cfg["hotspot_sigma"], (n_hot, 2)))
        pos = np.vstack([uni, hot])
        shad = self.rng.normal(0, cfg["shadow_sigma_db"], (self.n, m))
        is_hot = np.zeros(m, bool); is_hot[n_uni:] = True
        return pos, shad, is_hot

    def reset(self, randomize_init=False):
        cfg = self.cfg
        self.ue, self.shadow, self.is_hot = self._drop_users(cfg["n_ues"])
        self.tilt = np.full(self.n, cfg["tilt_init"])
        self.power = np.full(self.n, cfg["p_init"])
        if randomize_init:
            self.tilt += self.rng.integers(-2, 3, self.n)
            self.power += self.rng.integers(-1, 2, self.n)
            self._clip()
        self.prev_tilt = self.tilt.copy()
        self.prev_power = self.power.copy()
        self._compute()
        return self.features()

    def _clip(self):
        c = self.cfg
        self.tilt = np.clip(self.tilt, c["tilt_min"], c["tilt_max"])
        self.power = np.clip(self.power, c["p_min"], c["p_max"])

    def _churn(self):
        cfg = self.cfg
        m = cfg["n_ues"]
        k = max(1, int(m * cfg["churn_frac"]))
        idx = self.rng.integers(0, m, k)
        pos, shad, _ = self._drop_users(k)
        self.ue[idx] = pos
        self.shadow[:, idx] = shad
        jit = self.rng.normal(0, cfg["jitter_m"], (self.is_hot.sum(), 2))
        self.ue[self.is_hot] += jit

    # ---- radio -------------------------------------------------------------
    def _compute(self):
        cfg = self.cfg
        dx = self.sites[:, 0][:, None] - self.ue[:, 0][None, :]
        dy = self.sites[:, 1][:, None] - self.ue[:, 1][None, :]
        d2 = np.sqrt(dx * dx + dy * dy) + 1.0
        dh = cfg["bs_height"] - cfg["ue_height"]
        d3 = np.sqrt(d2 * d2 + dh * dh)
        pl = 28.0 + 22.0 * np.log10(d3) + 20.0 * np.log10(cfg["fc_ghz"])
        elev = np.degrees(np.arctan2(dh, d2))                       # [n, m]
        att = np.minimum(12.0 * ((elev - self.tilt[:, None])
                                 / cfg["theta_3db"]) ** 2, cfg["sla_db"])
        gain = cfg["ant_gain_dbi"] - att
        self.rsrp = self.power[:, None] + gain - pl - self.shadow   # dBm [n, m]
        self.serv = np.argmax(self.rsrp, axis=0)
        p_mw = 10 ** (self.rsrp / 10)
        tot = p_mw.sum(axis=0) + self.noise_mw
        m = self.ue.shape[0]
        p_serv = p_mw[self.serv, np.arange(m)]
        self.sinr = p_serv / (tot - p_serv)
        self.sinr_db = 10 * np.log10(self.sinr)
        self.rsrp_serv = self.rsrp[self.serv, np.arange(m)]
        self.load = np.bincount(self.serv, minlength=self.n)
        se = np.log2(1 + self.sinr)
        self.rate = cfg["bw_hz"] / np.maximum(self.load[self.serv], 1) * se
        # strongest interferer per UE
        p_int = p_mw.copy()
        p_int[self.serv, np.arange(m)] = -1
        self.top_int = np.argmax(p_int, axis=0)

    # ---- KPIs / reward -----------------------------------------------------
    def kpis(self):
        cfg = self.cfg
        cov = float(np.mean((self.sinr_db > cfg["sinr_edge_db"])
                            & (self.rsrp_serv > cfg["rsrp_cov_dbm"])))
        mean_tput = float(self.rate.mean() / 1e6)
        p5_tput = float(np.percentile(self.rate, 5) / 1e6)
        mean_se = float(np.log2(1 + self.sinr).mean())
        return dict(coverage=cov, mean_tput_mbps=mean_tput,
                    p5_tput_mbps=p5_tput, mean_se=mean_se)

    def reward(self):
        cfg = self.cfg
        k = self.kpis()
        churn = (np.abs(self.tilt - self.prev_tilt).sum()
                 + np.abs(self.power - self.prev_power).sum())
        return (cfg["w_cov"] * k["coverage"] + cfg["w_se"] * k["mean_se"]
                - cfg["w_churn"] * churn), k

    # ---- state -------------------------------------------------------------
    def features(self):
        cfg = self.cfg
        m = self.ue.shape[0]
        f = np.zeros((self.n, N_FEAT))
        edge = np.zeros(self.n); msinr = np.zeros(self.n)
        mrsrp = np.full(self.n, cfg["rsrp_cov_dbm"])
        for i in range(self.n):
            sel = self.serv == i
            if sel.any():
                edge[i] = np.mean(self.sinr_db[sel] < 0.0)
                msinr[i] = self.sinr_db[sel].mean()
                mrsrp[i] = self.rsrp_serv[sel].mean()
        victim = (self.sinr_db < 3.0)
        over = np.zeros(self.n)
        for i in range(self.n):
            over[i] = np.sum(victim & (self.top_int == i)) / max(1.0, m / self.n)
        load_share = self.load / max(1.0, m / self.n)
        f[:, 0] = self.tilt / cfg["tilt_max"]
        f[:, 1] = (self.power - cfg["p_min"]) / (cfg["p_max"] - cfg["p_min"])
        f[:, 2] = np.clip(load_share, 0, 3) / 3
        f[:, 3] = np.clip(msinr, -10, 30) / 30
        f[:, 4] = edge
        f[:, 5] = np.clip((mrsrp + 104) / 40, 0, 1.5)
        f[:, 6] = np.clip(over, 0, 1)
        for i in range(self.n):
            nb = self.neigh[i]
            if len(nb):
                f[i, 7] = edge[nb].mean()
                f[i, 8] = np.clip(load_share[nb], 0, 3).mean() / 3
                f[i, 9] = self.tilt[nb].mean() / cfg["tilt_max"]
        if self.mask_neighbors:
            f[:, 7:10] = 0.0     # ablation: strip one-hop neighbor context
        return f

    # ---- dynamics ----------------------------------------------------------
    def apply(self, actions):
        s = self.cfg["tilt_step"]; ps = self.cfg["p_step"]
        for i, a in enumerate(actions):
            if a == 1:   self.tilt[i] -= s
            elif a == 2: self.tilt[i] += s
            elif a == 3: self.power[i] -= ps
            elif a == 4: self.power[i] += ps
        self._clip()

    def step(self, actions, shield=True):
        """Advance one control epoch with an action-isolating safety check.

        Environmental churn is applied *before* the no-action reference KPI is
        measured. The shield then compares the candidate action against that
        same user/shadowing state, so a rollback is triggered only by
        action-induced degradation rather than exogenous user movement.
        """
        snap = (self.tilt.copy(), self.power.copy())
        self.prev_tilt, self.prev_power = snap[0].copy(), snap[1].copy()

        if self.cfg["dynamic"]:
            self._churn()
            self._compute()
        reference_cov = self.kpis()["coverage"]

        self.apply(actions)
        self._compute()
        safety_event = 0
        if shield:
            candidate_cov = self.kpis()["coverage"]
            if candidate_cov < reference_cov - self.cfg["shield_cov_drop"]:
                self.tilt, self.power = snap
                self._compute()
                safety_event = 1

        r, k = self.reward()
        if safety_event:
            r -= 1.0
        return self.features(), r, k, safety_event


# ----------------------------------------------------------------------------
# NumPy MLP + Adam + DQN
# ----------------------------------------------------------------------------
class MLP:
    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            lim = math.sqrt(6 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def params(self):
        return self.W + self.b

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def forward(self, x):
        acts = [x]
        for i in range(len(self.W)):
            x = x @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            acts.append(x)
        return x, acts

    def backward(self, acts, dout):
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d / d.shape[0]
            gb[i] = d.mean(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        return gW + gb


class Adam:
    def __init__(self, params, lr):
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)


class DQN:
    def __init__(self, seed, cfg=CFG):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.q = MLP([N_FEAT, cfg["hidden"], cfg["hidden"], N_ACT], rng)
        self.tq = MLP([N_FEAT, cfg["hidden"], cfg["hidden"], N_ACT], rng)
        self.tq.copy_from(self.q)
        self.opt = Adam(self.q.params(), cfg["lr"])
        self.buf_s = np.zeros((cfg["buffer_size"], N_FEAT))
        self.buf_a = np.zeros(cfg["buffer_size"], int)
        self.buf_r = np.zeros(cfg["buffer_size"])
        self.buf_s2 = np.zeros((cfg["buffer_size"], N_FEAT))
        self.ptr = 0; self.full = False
        self.grad_steps = 0
        self.rng = rng

    def push(self, s, a, r, s2):
        n = len(a)
        idx = (self.ptr + np.arange(n)) % self.cfg["buffer_size"]
        self.buf_s[idx] = s; self.buf_a[idx] = a
        self.buf_r[idx] = r; self.buf_s2[idx] = s2
        self.ptr = (self.ptr + n) % self.cfg["buffer_size"]
        self.full = self.full or self.ptr < n

    def size(self):
        return self.cfg["buffer_size"] if self.full else self.ptr

    def act(self, s, eps):
        qv, _ = self.q.forward(s)
        a = np.argmax(qv, axis=1)
        mask = self.rng.uniform(size=len(a)) < eps
        a[mask] = self.rng.integers(0, N_ACT, mask.sum())
        return a

    def learn(self):
        if self.size() < max(self.cfg["batch"], self.cfg["warmup"]):
            return None
        idx = self.rng.integers(0, self.size(), self.cfg["batch"])
        s, a = self.buf_s[idx], self.buf_a[idx]
        r, s2 = self.buf_r[idx], self.buf_s2[idx]
        q2, _ = self.tq.forward(s2)
        tgt = r + self.cfg["gamma"] * q2.max(axis=1)
        qv, acts = self.q.forward(s)
        td = qv[np.arange(len(a)), a] - tgt
        td_c = np.clip(td, -1, 1)                       # Huber gradient
        dout = np.zeros_like(qv)
        dout[np.arange(len(a)), a] = td_c
        grads = self.q.backward(acts, dout)
        self.opt.step(self.q.params(), grads)
        self.grad_steps += 1
        if self.grad_steps % self.cfg["target_every"] == 0:
            self.tq.copy_from(self.q)
        return float(np.mean(td * td))


# ----------------------------------------------------------------------------
# Baselines & auxiliary controllers
# ----------------------------------------------------------------------------
def rule_policy(env):
    """Classic SON heuristic: contain overshooters, boost weak edges."""
    f = env.features()
    a = np.zeros(env.n, int)
    for i in range(env.n):
        over, edge, rsrp = f[i, 6], f[i, 4], f[i, 5]
        if over > 0.30 and env.tilt[i] < env.cfg["tilt_max"]:
            a[i] = 2
        elif edge > 0.25 and rsrp < 0.5:
            a[i] = 4 if env.power[i] < env.cfg["p_max"] else 1
    return a


def random_search_config(env, n_trials, rng):
    """Select a static configuration on the current radio snapshot.

    The search score excludes action churn because this baseline chooses one
    configuration before rollout rather than issuing a sequence of actions.
    """
    best_score, best = -1e18, None
    for _ in range(n_trials):
        env.tilt = rng.uniform(env.cfg["tilt_min"],
                               env.cfg["tilt_max"], env.n).round()
        env.power = rng.uniform(env.cfg["p_min"],
                                env.cfg["p_max"], env.n).round()
        env._compute()
        k = env.kpis()
        score = env.cfg["w_cov"] * k["coverage"] + env.cfg["w_se"] * k["mean_se"]
        if score > best_score:
            best_score = score
            best = (env.tilt.copy(), env.power.copy())
    if best is None:
        raise RuntimeError("random search received zero trials")
    env.tilt, env.power = best
    env.prev_tilt, env.prev_power = env.tilt.copy(), env.power.copy()
    env._compute()
    return best


class EnergySaver:
    """Simple coexisting SON loop: down-powers the most lightly-loaded
    cells each period (bottom `frac` by load share). Using a relative
    rank instead of an absolute threshold guarantees the loop actually
    acts every period regardless of the absolute load distribution --
    an absolute threshold can silently never trigger in some deployment
    scenarios, which would make coordinated/uncoordinated runs
    indistinguishable by construction rather than by finding."""
    def __init__(self, period=10, frac=0.4, delta=3.0):
        self.period, self.frac, self.delta = period, frac, delta
        self.locked = set()

    def maybe_act(self, env, t):
        """Apply ES first at its decision epochs and retain cell locks."""
        if t % self.period:
            return False
        self.locked = set()
        share = env.load / max(1.0, env.ue.shape[0] / env.n)
        k = max(1, int(round(self.frac * env.n)))
        order = np.argsort(share)          # least-loaded cells first
        for i in order[:k]:
            env.power[i] = max(env.cfg["p_min"], env.power[i] - self.delta)
            self.locked.add(int(i))
        env._clip()
        env._compute()
        return True


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def _reset_eval_env(env, run_seed, topo_seed, init):
    """Use distinct, reproducible user realizations for each outer seed."""
    seed = 5_000_000 + 100_000 * int(run_seed) + 1000 * int(topo_seed) + int(init)
    env.rng = np.random.default_rng(seed)
    env.reset(randomize_init=(init > 0))


def _count_direction_reversals(history):
    """Count sign changes while ignoring zero-movement intervals."""
    h = np.asarray(history, dtype=float)
    if h.ndim != 2 or len(h) < 3:
        return 0
    d = np.diff(h, axis=0)
    total = 0
    for j in range(d.shape[1]):
        signs = np.sign(d[:, j])
        signs = signs[signs != 0]
        if len(signs) > 1:
            total += int(np.sum(signs[1:] * signs[:-1] < 0))
    return total


def _history_metrics(tilt_hist, power_hist, cfg, control_steps):
    th = np.asarray(tilt_hist, dtype=float)
    ph = np.asarray(power_hist, dtype=float)
    n = th.shape[1]
    dt = np.diff(th, axis=0) / max(cfg["tilt_step"], 1e-12)
    dp = np.diff(ph, axis=0) / max(cfg["p_step"], 1e-12)
    churn = (np.abs(dt).sum() + np.abs(dp).sum()) / max(1, control_steps * n)
    denom = max(1, control_steps * n)
    tilt_rev = 100.0 * _count_direction_reversals(th) / denom
    power_rev = 100.0 * _count_direction_reversals(ph) / denom
    return {
        "parameter_churn_per_cell_step": float(churn),
        "tilt_reversals_per_100_cell_steps": float(tilt_rev),
        "power_reversals_per_100_cell_steps": float(power_rev),
        "all_reversals_per_100_cell_steps": float(tilt_rev + power_rev),
    }


def evaluate_controller(policy_fn, topo_seeds, n_init=3, steps=40,
                        shield=True, mask_neighbors=False, run_seed=0,
                        preconfigure_fn=None):
    """Evaluate any controller under one common temporal protocol.

    KPIs are averaged over every rollout step, then over initializations and
    topologies. This prevents fixed/rule/random baselines from being measured
    on one static snapshot while the learned agent is measured dynamically.
    """
    rollout_kpis, stability, safety_rates = [], [], []
    for ts in topo_seeds:
        for init in range(n_init):
            env = Env(ts, mask_neighbors=mask_neighbors)
            _reset_eval_env(env, run_seed, ts, init)
            if preconfigure_fn is not None:
                preconfigure_fn(env, ts, init)

            tilt_hist = [env.tilt.copy()]
            power_hist = [env.power.copy()]
            k_hist = []
            safety = 0
            for _ in range(steps):
                actions = np.asarray(policy_fn(env), dtype=int)
                if actions.shape != (env.n,):
                    raise ValueError(f"controller returned {actions.shape}; expected {(env.n,)}")
                _, _, k, se = env.step(actions, shield=shield)
                safety += se
                k_hist.append(k)
                tilt_hist.append(env.tilt.copy())
                power_hist.append(env.power.copy())

            rollout_kpis.append({
                key: float(np.mean([x[key] for x in k_hist]))
                for key in k_hist[0]
            })
            stability.append(_history_metrics(tilt_hist, power_hist,
                                               env.cfg, steps))
            safety_rates.append(100.0 * safety / max(1, steps))

    out = {key: float(np.mean([x[key] for x in rollout_kpis]))
           for key in rollout_kpis[0]}
    for key in stability[0]:
        out[key] = float(np.mean([x[key] for x in stability]))
    out["safety_events_per_100_steps"] = float(np.mean(safety_rates))
    return out


def evaluate(agent, topo_seeds, n_init=None, steps=None, shield=True,
             mask_neighbors=False, run_seed=None):
    cfg = agent.cfg
    return evaluate_controller(
        lambda env: agent.act(env.features(), eps=0.0),
        topo_seeds,
        n_init=cfg["eval_n_init"] if n_init is None else n_init,
        steps=cfg["eval_steps"] if steps is None else steps,
        shield=shield,
        mask_neighbors=mask_neighbors,
        run_seed=cfg["seed"] if run_seed is None else run_seed,
    )


def evaluate_random_search(topo_seeds, n_trials, cfg=CFG):
    def configure(env, ts, init):
        rng = np.random.default_rng(9_000_000 + 100_000 * cfg["seed"]
                                    + 1000 * int(ts) + int(init))
        random_search_config(env, n_trials, rng)

    return evaluate_controller(
        lambda env: np.zeros(env.n, dtype=int),
        topo_seeds,
        n_init=cfg["eval_n_init"], steps=cfg["eval_steps"],
        shield=False, run_seed=cfg["seed"], preconfigure_fn=configure,
    )


def train(agent, topo_seeds, episodes, cfg=CFG, log_every=10, curve=None,
          shield=True, mask_neighbors=False, eps_override=None,
          curve_topos=None):
    total = episodes * cfg["steps_per_ep"]
    step_ct = 0
    for ep in range(episodes):
        ts = topo_seeds[ep % len(topo_seeds)]
        env = Env(ts, mask_neighbors=mask_neighbors)
        # Keep topology/hotspots fixed, but vary users and shadowing by episode
        # and outer experiment seed instead of replaying the same realization.
        env.rng = np.random.default_rng(2_000_000 + 100_000 * cfg["seed"]
                                        + 1000 * int(ts) + ep)
        s = env.reset(randomize_init=True)
        for _ in range(cfg["steps_per_ep"]):
            if eps_override is not None:
                eps = eps_override
            else:
                frac = min(1.0, step_ct / max(1.0, cfg["eps_frac"] * total))
                eps = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])
            a = agent.act(s, eps)
            s2, r, _, _ = env.step(a, shield=shield)
            agent.push(s, a, np.full(env.n, r), s2)
            agent.learn()
            s = s2
            step_ct += 1

        completed = ep + 1
        if curve is not None and (completed == 1 or completed % log_every == 0):
            val_topos = list(curve_topos or topo_seeds[:2])
            m = evaluate(agent, val_topos, n_init=1, steps=25,
                         mask_neighbors=mask_neighbors,
                         run_seed=cfg["seed"] + 50_000)
            curve.append((completed, m["coverage"], m["mean_tput_mbps"]))
    return agent


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def exp1(cfg, quick):
    print("=== E1: KPIs vs baselines ===")
    episodes = 30 if quick else cfg["episodes_e1"]
    topologies = list(cfg["train_topos"])
    res = {}

    # Every method uses the same topology IDs, user initializations, dynamic
    # rollout length, and temporal KPI averaging.
    res["fixed"] = evaluate_controller(
        lambda env: np.zeros(env.n, dtype=int), topologies,
        n_init=cfg["eval_n_init"], steps=cfg["eval_steps"],
        shield=False, run_seed=cfg["seed"])
    res["random_search"] = evaluate_random_search(
        topologies, 60 if quick else 300, cfg)
    res["rule_based"] = evaluate_controller(
        rule_policy, topologies,
        n_init=cfg["eval_n_init"], steps=cfg["eval_steps"],
        shield=False, run_seed=cfg["seed"])

    curve = []
    agent = DQN(cfg["seed"])
    train(agent, topologies, episodes, curve=curve, curve_topos=topologies[:2])
    res["dqn_shielded"] = evaluate(agent, topologies)

    # Same network initialization and environment protocol isolate the shield.
    ab = DQN(cfg["seed"])
    train(ab, topologies, episodes, shield=False)
    res["dqn_no_shield"] = evaluate(ab, topologies, shield=False)

    # figures
    c = np.array(curve)
    if len(c):
        CURVE_STORE.setdefault("e1_learning", []).append(c)
        fig, ax1 = plt.subplots(figsize=(5, 3.2))
        ax1.plot(c[:, 0], c[:, 1] * 100, "b-o", ms=3, label="Coverage")
        ax1.set_xlabel("Completed training episodes")
        ax1.set_ylabel("Coverage [%]", color="b")
        ax2 = ax1.twinx()
        ax2.plot(c[:, 0], c[:, 2], "r-s", ms=3, label="Mean throughput")
        ax2.set_ylabel("Mean UE throughput [Mbps]", color="r")
        fig.tight_layout(); fig.savefig("figures/learning_curve.png", dpi=200)
        plt.close(fig)
        np.savetxt("results/learning_curve.csv", c, delimiter=",",
                   header="completed_episodes,coverage,mean_tput_mbps", comments="")

    env = Env(cfg["train_topos"][0])
    _reset_eval_env(env, cfg["seed"], cfg["train_topos"][0], 0)
    sinr_before = env.sinr_db.copy()
    for _ in range(cfg["eval_steps"]):
        env.step(agent.act(env.features(), 0.0))
    fig = plt.figure(figsize=(5, 3.2))
    for d, lb in [(sinr_before, "Default config"), (env.sinr_db, "Proposed CCO")]:
        xs = np.sort(d); plt.plot(xs, np.linspace(0, 1, len(xs)), label=lb)
    plt.axvline(cfg["sinr_edge_db"], ls="--", c="gray", lw=0.8)
    plt.xlabel("DL SINR [dB]"); plt.ylabel("CDF"); plt.legend()
    plt.tight_layout(); plt.savefig("figures/sinr_cdf.png", dpi=200); plt.close(fig)

    return res, agent


def exp2(cfg, quick, agent):
    print("=== E2: generalization to held-out topologies ===")
    episodes_ft = 5 if quick else 20
    res = {"zero_shot_unseen": evaluate(agent, list(cfg["test_topos"])),
           "seen_reference": evaluate(agent, list(cfg["train_topos"]))}

    # Few-shot: warm-start from the transferred policy, then fine-tune on the
    # target topologies. Fine-tuning uses a reduced learning rate and a low,
    # fixed exploration rate: at full lr/epsilon the policy is effectively
    # retrained on a narrow 4-topology slice and forgets what it transferred.
    ft = DQN(cfg["seed"] + 2)
    ft.q.copy_from(agent.q); ft.tq.copy_from(agent.q)
    ft.opt = Adam(ft.q.params(), cfg["lr"] * cfg["ft_lr_scale"])
    train(ft, list(cfg["test_topos"]), episodes_ft,
          eps_override=cfg["ft_eps"])
    res["few_shot_unseen"] = evaluate(ft, list(cfg["test_topos"]))

    # Target-trained reference: trained from scratch on held-out topology IDs
    # with the same full training budget. It is a reference, not an oracle or
    # guaranteed upper bound.
    orc = DQN(cfg["seed"] + 3)
    train(orc, list(cfg["test_topos"]),
          30 if quick else cfg["episodes_e1"])
    res["target_trained_reference"] = evaluate(orc, list(cfg["test_topos"]))

    labels = ["seen_reference", "zero_shot_unseen", "few_shot_unseen", "target_trained_reference"]
    vals = [res[l]["coverage"] * 100 for l in labels]
    fig = plt.figure(figsize=(5, 3.2))
    plt.bar(["Seen", "Zero-shot", f"Few-shot\n({episodes_ft} ep)", "Target-trained"], vals)
    plt.ylabel("Coverage [%]"); plt.tight_layout()
    plt.savefig("figures/generalization.png", dpi=200); plt.close(fig)
    return res


def exp3(cfg, quick):
    print("=== E3: sample efficiency via offline warm-start ===")
    episodes = 30 if quick else 150
    # 1) log data with behaviour (rule + exploration) policy
    logger = DQN(cfg["seed"] + 4)
    rng = np.random.default_rng(7)
    n_log = 4000 if quick else 20000
    log_episode = 0
    while logger.size() < n_log:
        ts = cfg["train_topos"][log_episode % len(cfg["train_topos"])]
        env = Env(ts)
        env.rng = np.random.default_rng(3_000_000 + 100_000 * cfg["seed"]
                                        + 1000 * int(ts) + log_episode)
        s = env.reset(randomize_init=True)
        for _ in range(cfg["steps_per_ep"]):
            a = rule_policy(env)
            mask = rng.uniform(size=env.n) < 0.2
            a[mask] = rng.integers(0, N_ACT, mask.sum())
            s2, r, _, _ = env.step(a)
            logger.push(s, a, np.full(env.n, r), s2)
            s = s2
        log_episode += 1
    # 2) offline FQI on the frozen buffer
    warm = DQN(cfg["seed"] + 5)
    warm.buf_s, warm.buf_a = logger.buf_s, logger.buf_a
    warm.buf_r, warm.buf_s2 = logger.buf_r, logger.buf_s2
    warm.ptr, warm.full = logger.ptr, logger.full
    for _ in range(1000 if quick else 6000):
        warm.learn()
    res = {"offline_only": evaluate(warm, list(cfg["train_topos"]))}
    # 3) online fine-tune vs from scratch
    cw, cs = [], []
    train(warm, list(cfg["train_topos"]), episodes, curve=cw, log_every=5)
    scratch = DQN(cfg["seed"] + 6)
    train(scratch, list(cfg["train_topos"]), episodes, curve=cs, log_every=5)
    res["warm_final"] = evaluate(warm, list(cfg["train_topos"]))
    res["scratch_final"] = evaluate(scratch, list(cfg["train_topos"]))
    cw, cs = np.array(cw), np.array(cs)
    CURVE_STORE.setdefault("e3_warm", []).append(cw)
    CURVE_STORE.setdefault("e3_scratch", []).append(cs)

    # Stable sample-efficiency metrics: coverage at fixed online budgets and
    # normalized area under the learning curve. No threshold-to-target metric
    # is emitted because its per-seed variance was too large for a sound claim.
    for budget in (5, 10, 25, 50):
        for name, cur in (("warm", cw), ("scratch", cs)):
            sel = cur[cur[:, 0] <= budget]
            if len(sel):
                res[f"cov_at_{budget}ep_{name}"] = float(sel[-1, 1])
        wk, sk = f"cov_at_{budget}ep_warm", f"cov_at_{budget}ep_scratch"
        if wk in res and sk in res:
            res[f"gain_at_{budget}ep_pp"] = 100.0 * (res[wk] - res[sk])

    _trap = getattr(np, "trapezoid", None) or np.trapz  # numpy <2.0 compat
    for name, cur in (("warm", cw), ("scratch", cs)):
        res[f"auc_{name}"] = float(_trap(cur[:, 1], cur[:, 0])
                                   / max(1e-9, cur[-1, 0] - cur[0, 0]))
    res["auc_gain_pp"] = 100.0 * (res["auc_warm"] - res["auc_scratch"])

    fig = plt.figure(figsize=(5, 3.2))
    plt.plot(cs[:, 0], cs[:, 1] * 100, "r-s", ms=3, label="From scratch")
    plt.plot(cw[:, 0], cw[:, 1] * 100, "b-o", ms=3, label="Offline warm-start")
    plt.xlabel("Completed online training episodes"); plt.ylabel("Coverage [%]"); plt.legend()
    plt.tight_layout(); plt.savefig("figures/sample_efficiency.png", dpi=200)
    plt.close(fig)
    return res


def exp4(cfg, quick, agent):
    print("=== E4: stability under SON coexistence ===")
    steps = 120 if quick else 300
    res = {}

    for regime, es_kw in (("mild", cfg["es_mild"]),
                          ("adversarial", cfg["es_adversarial"])):
        for mode in ("cco_only", "coex_uncoordinated", "coex_coordinated"):
            rollout_rows, cov_curves = [], []
            for ts in cfg["train_topos"][:3]:
                for init in range(2):
                    env = Env(ts)
                    _reset_eval_env(env, cfg["seed"] + 70_000, ts, init)
                    es = EnergySaver(**es_kw) if "coex" in mode else None

                    # Include ES and CCO substeps in the parameter history so a
                    # down-power followed by an immediate up-power is counted
                    # rather than hidden by end-of-step sampling.
                    tilt_hist = [env.tilt.copy()]
                    power_hist = [env.power.copy()]
                    cov_hist, power_mean_hist = [], []
                    wanted = blocked = locked_opportunities = 0

                    for t in range(steps):
                        if es is not None:
                            acted = es.maybe_act(env, t)  # ES acts first
                            if acted:
                                tilt_hist.append(env.tilt.copy())
                                power_hist.append(env.power.copy())

                        a = agent.act(env.features(), 0.0)
                        if es is not None:
                            locked_opportunities += len(es.locked)
                            for i in es.locked:
                                if a[i] != 0:
                                    wanted += 1
                                    if mode == "coex_coordinated":
                                        blocked += 1
                            if mode == "coex_coordinated":
                                for i in es.locked:
                                    a[i] = 0

                        _, _, k, _ = env.step(a)
                        cov_hist.append(k["coverage"])
                        power_mean_hist.append(float(env.power.mean()))
                        tilt_hist.append(env.tilt.copy())
                        power_hist.append(env.power.copy())

                    ch = np.asarray(cov_hist)
                    hm = _history_metrics(tilt_hist, power_hist, cfg, steps)
                    row = {
                        "mean_coverage": float(ch.mean()),
                        "coverage_temporal_std": float(ch[len(ch)//2:].std()),
                        "mean_tx_power_dbm": float(np.mean(
                            power_mean_hist[len(power_mean_hist)//2:])),
                        "contention_rate_pct": float(
                            100.0 * wanted / max(1, locked_opportunities)),
                        "blocked_action_rate_pct": float(
                            100.0 * blocked / max(1, wanted)),
                        **hm,
                    }
                    rollout_rows.append(row)
                    cov_curves.append(ch)

            key = f"{regime}_{mode}"
            res[key] = {
                metric: float(np.mean([r[metric] for r in rollout_rows]))
                for metric in rollout_rows[0]
            }
            cav = np.mean(np.stack(cov_curves), axis=0)
            CURVE_STORE.setdefault(f"e4_{key}", []).append(cav)

    # Explicit paired comparisons for the regime designed to induce conflict.
    u = res["adversarial_coex_uncoordinated"]
    c = res["adversarial_coex_coordinated"]
    res["adversarial_coordination_effect"] = {
        "coverage_delta_pp": 100.0 * (c["mean_coverage"] - u["mean_coverage"]),
        "coverage_variability_reduction_pct": 100.0 * (
            u["coverage_temporal_std"] - c["coverage_temporal_std"]
        ) / max(1e-12, u["coverage_temporal_std"]),
        "churn_reduction_pct": 100.0 * (
            u["parameter_churn_per_cell_step"]
            - c["parameter_churn_per_cell_step"]
        ) / max(1e-12, u["parameter_churn_per_cell_step"]),
        "reversal_reduction_pct": 100.0 * (
            u["all_reversals_per_100_cell_steps"]
            - c["all_reversals_per_100_cell_steps"]
        ) / max(1e-12, u["all_reversals_per_100_cell_steps"]),
        "mean_power_delta_db": (c["mean_tx_power_dbm"]
                                - u["mean_tx_power_dbm"]),
    }

    fig = plt.figure(figsize=(5, 3.2))
    for mode, lab, col in (("cco_only", "CCO only", "C0"),
                           ("coex_uncoordinated", "CCO + ES (uncoordinated)", "C1"),
                           ("coex_coordinated", "CCO + ES (coordinated)", "C2")):
        cur = CURVE_STORE[f"e4_adversarial_{mode}"][-1]
        plt.plot(cur * 100, label=lab, lw=1, color=col)
    plt.xlabel("Control step"); plt.ylabel("Coverage [%]")
    plt.title("adversarial ES regime", fontsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figures/coexistence.png", dpi=200)
    plt.close(fig)
    return res


def exp5(cfg, quick, agent):
    """Ablation: does the one-hop neighbor-aggregated (graph-lite) context
    actually drive zero-shot transfer, or would parameter sharing alone do
    just as well? Trains an otherwise-identical agent with neighbor-aggregate
    features (state dims 7-9) zeroed out throughout training and evaluation,
    and compares its seen-vs-unseen generalization gap to the full agent's."""
    print("=== E5: ablation -- graph-lite neighbor features ===")
    episodes = 30 if quick else cfg["episodes_e1"]
    ab = DQN(cfg["seed"] + 8)
    train(ab, list(cfg["train_topos"]), episodes, mask_neighbors=True)

    res = {
        "full_seen": evaluate(agent, list(cfg["train_topos"])),
        "full_zero_shot": evaluate(agent, list(cfg["test_topos"])),
        "ablated_seen": evaluate(ab, list(cfg["train_topos"]),
                                  mask_neighbors=True),
        "ablated_zero_shot": evaluate(ab, list(cfg["test_topos"]),
                                       mask_neighbors=True),
    }
    res["full_gap_pp"] = ((res["full_seen"]["coverage"]
                           - res["full_zero_shot"]["coverage"]) * 100)
    res["ablated_gap_pp"] = ((res["ablated_seen"]["coverage"]
                              - res["ablated_zero_shot"]["coverage"]) * 100)
    res["unseen_absolute_gain_pp"] = (
        res["full_zero_shot"]["coverage"]
        - res["ablated_zero_shot"]["coverage"]) * 100
    res["gap_widening_without_neighbors_pp"] = (
        res["ablated_gap_pp"] - res["full_gap_pp"])

    labels = ["Full\n(seen)", "Full\n(zero-shot)", "Ablated\n(seen)",
              "Ablated\n(zero-shot)"]
    vals = [res["full_seen"]["coverage"] * 100,
            res["full_zero_shot"]["coverage"] * 100,
            res["ablated_seen"]["coverage"] * 100,
            res["ablated_zero_shot"]["coverage"] * 100]
    colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"]
    fig = plt.figure(figsize=(5, 3.2))
    plt.bar(labels, vals, color=colors)
    plt.ylabel("Coverage [%]")
    plt.tight_layout()
    plt.savefig("figures/ablation_neighbors.png", dpi=200)
    plt.close(fig)
    return res




# ----------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def run_pipeline(cfg, quick, which="all"):
    """Run the requested experiment(s) once under cfg['seed'] and return a
    summary dict. E2/E4/E5 need the E1 agent, so E1 always runs first when
    any of those are requested."""
    summary = {"config": _clean(dict(cfg))}
    agent = None
    if which in ("all", "e1", "e2", "e4", "e5"):
        summary["E1"], agent = exp1(cfg, quick)
    if which in ("all", "e2"):
        summary["E2"] = exp2(cfg, quick, agent)
    if which in ("all", "e3"):
        summary["E3"] = exp3(cfg, quick)
    if which in ("all", "e4"):
        summary["E4"] = exp4(cfg, quick, agent)
    if which in ("all", "e5"):
        summary["E5"] = exp5(cfg, quick, agent)
    return summary


def aggregate(summaries):
    """Mean/std across a list of same-shaped summary dicts (E1..E5 only;
    'config' is carried over as-is from the first run since it isn't
    numeric)."""
    agg = {"config": summaries[0]["config"]}

    def walk(dicts):
        first = dicts[0]
        out = {}
        for k, v0 in first.items():
            vals = [d[k] for d in dicts]
            if isinstance(v0, dict):
                out[k] = walk(vals)
            elif isinstance(v0, (int, float)):
                out[k] = {"mean": float(np.mean(vals)),
                          "std": float(np.std(vals)),
                          "n_seeds": len(vals)}
        return out

    for exp_key in ("E1", "E2", "E3", "E4", "E5"):
        if exp_key in summaries[0]:
            agg[exp_key] = walk([s[exp_key] for s in summaries])
    return agg


def replot_aggregate_figures(agg):
    """Redraw the figures using every seed rather than just the last run.
    Line plots get a mean curve with a +/- 1 std shaded band; bar charts
    get error bars taken from the aggregated summary. Without this, the
    paper would show single-seed figures next to multi-seed tables."""
    n = None

    def _stack(key):
        """Stack per-seed curves, truncating to the shortest in case a run
        logged a different number of points."""
        curves = CURVE_STORE.get(key, [])
        if not curves:
            return None
        L = min(len(c) for c in curves)
        return np.stack([c[:L] for c in curves])

    # --- E1 learning curve (coverage + throughput vs episode) ------------
    st = _stack("e1_learning")
    if st is not None and st.shape[0] > 1:
        n = st.shape[0]
        ep = st[0, :, 0]
        cov_m, cov_s = st[:, :, 1].mean(0) * 100, st[:, :, 1].std(0) * 100
        tp_m, tp_s = st[:, :, 2].mean(0), st[:, :, 2].std(0)
        fig, ax1 = plt.subplots(figsize=(5, 3.2))
        ax1.plot(ep, cov_m, "b-o", ms=3, label="Coverage")
        ax1.fill_between(ep, cov_m - cov_s, cov_m + cov_s, color="b", alpha=0.18)
        ax1.set_xlabel("Training episode")
        ax1.set_ylabel("Coverage [%]", color="b")
        ax2 = ax1.twinx()
        ax2.plot(ep, tp_m, "r-s", ms=3, label="Mean throughput")
        ax2.fill_between(ep, tp_m - tp_s, tp_m + tp_s, color="r", alpha=0.18)
        ax2.set_ylabel("Mean UE throughput [Mbps]", color="r")
        ax1.set_title(f"mean $\\pm$ 1 std over {n} seeds", fontsize=8)
        fig.tight_layout()
        fig.savefig("figures/learning_curve.png", dpi=200)
        plt.close(fig)

    # --- E3 sample efficiency (warm-start vs scratch) --------------------
    sw, ss = _stack("e3_warm"), _stack("e3_scratch")
    if sw is not None and ss is not None and sw.shape[0] > 1:
        n = sw.shape[0]
        fig = plt.figure(figsize=(5, 3.2))
        for st_, c, lab, mk in ((ss, "r", "From scratch", "s"),
                                (sw, "b", "Offline warm-start", "o")):
            ep = st_[0, :, 0]
            m, s = st_[:, :, 1].mean(0) * 100, st_[:, :, 1].std(0) * 100
            plt.plot(ep, m, c + "-" + mk, ms=3, label=lab)
            plt.fill_between(ep, m - s, m + s, color=c, alpha=0.18)
        plt.xlabel("Completed online training episodes"); plt.ylabel("Coverage [%]")
        plt.title(f"mean $\\pm$ 1 std over {n} seeds", fontsize=8)
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/sample_efficiency.png", dpi=200)
        plt.close(fig)

    # --- E4 coexistence over time (adversarial regime) -------------------
    sets = [("e4_adversarial_cco_only", "CCO only", "C0"),
            ("e4_adversarial_coex_uncoordinated", "CCO + ES (uncoordinated)", "C1"),
            ("e4_adversarial_coex_coordinated", "CCO + ES (coordinated)", "C2")]
    stacks = {k: _stack(k) for k, _, _ in sets}
    if all(v is not None and v.shape[0] > 1 for v in stacks.values()):
        n = list(stacks.values())[0].shape[0]
        fig = plt.figure(figsize=(5, 3.2))
        for key, lab, col in sets:
            st_ = stacks[key]
            m, s = st_.mean(0) * 100, st_.std(0) * 100
            x = np.arange(len(m))
            plt.plot(x, m, color=col, lw=1, label=lab)
            plt.fill_between(x, m - s, m + s, color=col, alpha=0.15)
        plt.xlabel("Control step"); plt.ylabel("Coverage [%]")
        plt.title(f"adversarial ES regime; mean $\\pm$ 1 std over {n} seeds",
                  fontsize=8)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("figures/coexistence.png", dpi=200)
        plt.close(fig)

    # --- E2 generalization bars (with error bars) ------------------------
    if "E2" in agg:
        keys = ["seen_reference", "zero_shot_unseen", "few_shot_unseen",
                "target_trained_reference"]
        labs = ["Seen", "Zero-shot", "Few-shot", "Target-trained"]
        if all(k in agg["E2"] for k in keys):
            m = [agg["E2"][k]["coverage"]["mean"] * 100 for k in keys]
            s = [agg["E2"][k]["coverage"]["std"] * 100 for k in keys]
            n = agg["E2"][keys[0]]["coverage"]["n_seeds"]
            fig = plt.figure(figsize=(5, 3.2))
            plt.bar(labs, m, yerr=s, capsize=4)
            plt.ylabel("Coverage [%]")
            plt.title(f"mean $\\pm$ 1 std over {n} seeds", fontsize=8)
            plt.tight_layout()
            plt.savefig("figures/generalization.png", dpi=200)
            plt.close(fig)

    # --- E5 ablation bars (with error bars) ------------------------------
    if "E5" in agg:
        keys = ["full_seen", "full_zero_shot", "ablated_seen",
                "ablated_zero_shot"]
        labs = ["Full\n(seen)", "Full\n(zero-shot)", "Ablated\n(seen)",
                "Ablated\n(zero-shot)"]
        if all(k in agg["E5"] for k in keys):
            m = [agg["E5"][k]["coverage"]["mean"] * 100 for k in keys]
            s = [agg["E5"][k]["coverage"]["std"] * 100 for k in keys]
            n = agg["E5"][keys[0]]["coverage"]["n_seeds"]
            fig = plt.figure(figsize=(5, 3.2))
            plt.bar(labs, m, yerr=s, capsize=4,
                    color=["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"])
            plt.ylabel("Coverage [%]")
            plt.title(f"mean $\\pm$ 1 std over {n} seeds", fontsize=8)
            plt.tight_layout()
            plt.savefig("figures/ablation_neighbors.png", dpi=200)
            plt.close(fig)

    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["all", "e1", "e2", "e3", "e4", "e5"])
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--seed", type=int, default=0,
                    help="single-seed run (ignored if --seeds is given)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds to run and aggregate, "
                         "e.g. 0,1,2 -- always runs the full pipeline "
                         "(all 5 experiments) for each seed")
    args = ap.parse_args()

    if args.seeds:
        seed_list = [int(s) for s in args.seeds.split(",")]
        summaries = []
        for sd in seed_list:
            print(f"\n{'#'*70}\n# SEED {sd}\n{'#'*70}")
            CFG["seed"] = sd
            np.random.seed(sd)
            s = run_pipeline(CFG, args.quick, which="all")
            summaries.append(s)
            with open(f"results/summary_seed{sd}.json", "w") as f:
                json.dump(_clean(s), f, indent=2)
        agg = aggregate(summaries)
        replot_aggregate_figures(agg)
        with open("results/summary_aggregated.json", "w") as f:
            json.dump(_clean(agg), f, indent=2)
        print("\n" + "=" * 70)
        print(f"DONE. Ran seeds {seed_list}.")
        print("Send back results/summary_aggregated.json (mean +/- std "
              "across seeds -- this is what the paper tables should use).")
        print("Individual results/summary_seed<N>.json files are also "
              "saved if you want to inspect a specific run.")
        print(f"Figures in figures/ now aggregate all {len(seed_list)} "
              "seeds (mean with +/- 1 std bands / error bars).")
        print("=" * 70)
        print(json.dumps(_clean(agg), indent=2)[:4000])
    else:
        CFG["seed"] = args.seed
        np.random.seed(args.seed)
        summary = run_pipeline(CFG, args.quick, which=args.exp)
        with open("results/summary.json", "w") as f:
            json.dump(_clean(summary), f, indent=2)
        print("\n" + "=" * 70)
        print("DONE (single seed). Send back results/summary.json")
        print("(plus the PNGs in figures/ if you want me to sanity-check "
              "them). For paper-grade numbers, prefer --seeds 0,1,2,3,4 "
              "instead of a single run.")
        print("=" * 70)
        print(json.dumps(_clean(summary), indent=2)[:4000])


if __name__ == "__main__":
    main()