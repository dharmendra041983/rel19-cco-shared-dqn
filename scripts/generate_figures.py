#!/usr/bin/env python3
"""Regenerate IEEE-sized Figures 2 and 3 from archived aggregate JSON results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def figure2(original: dict, additional: dict, outdir: Path) -> None:
    episodes = np.asarray([5, 10, 25, 50], dtype=float)
    warm = np.asarray([
        [original["E3"][f"cov_at_{int(ep)}ep_warm"]["mean"],
         original["E3"][f"cov_at_{int(ep)}ep_warm"]["std"]]
        for ep in episodes
    ]) * 100.0
    scratch = np.asarray([
        [original["E3"][f"cov_at_{int(ep)}ep_scratch"]["mean"],
         original["E3"][f"cov_at_{int(ep)}ep_scratch"]["std"]]
        for ep in episodes
    ]) * 100.0
    ppo = np.asarray([
        [additional["PPO"]["learning"][f"coverage_at_{int(ep)}ep"]["mean"],
         additional["PPO"]["learning"][f"coverage_at_{int(ep)}ep"]["std"]]
        for ep in episodes
    ]) * 100.0

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.errorbar(episodes, warm[:, 0], yerr=warm[:, 1], fmt="o-",
                color="black", capsize=3, label="Warm-start DQN")
    ax.errorbar(episodes, scratch[:, 0], yerr=scratch[:, 1], fmt="s--",
                color="black", capsize=3, label="Scratch DQN")
    ax.errorbar(episodes, ppo[:, 0], yerr=ppo[:, 1], fmt="^:",
                color="black", capsize=3, label="Shared PPO")
    ax.set_xlabel("Online training episodes")
    ax.set_ylabel("Coverage [%]")
    ax.set_xticks(episodes)
    ax.set_xlim(3, 52)
    ax.set_ylim(70, 90)
    ax.grid(axis="y", linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, borderaxespad=0.0, handlelength=2.2,
              handletextpad=0.5, columnspacing=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(outdir / "figure2_sample_efficiency.pdf")
    fig.savefig(outdir / "figure2_sample_efficiency.png", dpi=220)
    plt.close(fig)


def figure3(original: dict, outdir: Path) -> None:
    keys = [
        ("adversarial_cco_only", "CCO only"),
        ("adversarial_coex_uncoordinated", "Uncoordinated"),
        ("adversarial_coex_coordinated", "Static lock"),
    ]
    labels = [label for _, label in keys]
    cov = np.asarray([
        [original["E4"][key]["mean_coverage"]["mean"],
         original["E4"][key]["mean_coverage"]["std"]]
        for key, _ in keys
    ]) * 100.0
    rev = np.asarray([
        [original["E4"][key]["all_reversals_per_100_cell_steps"]["mean"],
         original["E4"][key]["all_reversals_per_100_cell_steps"]["std"]]
        for key, _ in keys
    ])

    x = np.arange(len(labels), dtype=float)
    fig, ax1 = plt.subplots(figsize=(3.5, 3.0))
    ax1.bar(x, cov[:, 0], width=0.55, yerr=cov[:, 1], capsize=3,
            facecolor="white", edgecolor="black", hatch="//",
            linewidth=1.0, label="Coverage")
    ax1.set_ylabel("Coverage [%]")
    ax1.set_ylim(82, 94)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", linestyle=":", linewidth=0.7)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.errorbar(x, rev[:, 0], yerr=rev[:, 1], fmt="o--",
                 color="black", linewidth=1.2, markersize=4,
                 capsize=3, label="Reversals / 100")
    ax2.set_ylabel("Reversals / 100 cell-steps")
    ax2.set_ylim(0, 0.65)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False,
               borderaxespad=0.0, handlelength=2.0,
               handletextpad=0.5, columnspacing=1.0)
    fig.subplots_adjust(top=0.80, left=0.14, right=0.86, bottom=0.18)
    fig.savefig(outdir / "figure3_coexistence.pdf")
    fig.savefig(outdir / "figure3_coexistence.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=Path("results/original_five_seed_aggregated.json"))
    parser.add_argument("--additional", type=Path, default=Path("results/additional_ppo_safety_scale_aggregated.json"))
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    original = json.loads(args.original.read_text())
    additional = json.loads(args.additional.read_text())
    figure2(original, additional, args.outdir)
    figure3(original, args.outdir)


if __name__ == "__main__":
    main()
