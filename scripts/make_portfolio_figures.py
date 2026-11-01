"""Rebuild portfolio figures from the checked-in CIFAR-100 histories."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HISTORIES = RESULTS / "histories"

CONFIGS = [
    ("full", "Full upload", "cifar100_comm_full_dirichlet_rep*.csv", "#333333", "-"),
    (
        "sparse",
        "Sparse (tau=200)",
        "cifar100_comm_sparse_tau200.0_mu3_dirichlet_rep*.csv",
        "#1f77b4",
        "--",
    ),
    (
        "adaptive",
        "Adaptive (EMA)",
        "cifar100_comm_adaptive_dirichlet_rep*.csv",
        "#d62728",
        "-.",
    ),
]


def load_runs(pattern):
    files = sorted(HISTORIES.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No history files match {pattern}")
    return [pd.read_csv(path) for path in files]


def main():
    RESULTS.mkdir(exist_ok=True)
    all_runs = {}
    summary = []

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    for idx, (key, label, pattern, color, line_style) in enumerate(CONFIGS):
        runs = load_runs(pattern)
        all_runs[key] = runs
        top1 = np.array([run["top1"].to_numpy() for run in runs]) * 100
        mean = top1.mean(axis=0)
        std = top1.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
        rounds = np.arange(1, len(mean) + 1)
        ax.plot(rounds, mean, color=color, linestyle=line_style, linewidth=2.0,
                label=label)
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.14)

        finals = top1[:, -1]
        uploads = np.array([run["upload_ratio"].mean() for run in runs])
        savings = (1.0 - uploads) * 100
        summary.append(
            (
                key,
                finals.mean(),
                finals.std(ddof=1) if len(finals) > 1 else 0.0,
                uploads.mean() * 100,
                savings.mean(),
            )
        )

    ax.set_xlabel("Federated round")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("CIFAR-100 communication scheduling (3 repeats)")
    ax.set_xticks(np.arange(1, 21, 2))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "cifar100_comm_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for key, label, color, marker in [
        ("full", "Full upload", "#333333", "o"),
        ("sparse", "Sparse (tau=200)", "#1f77b4", "s"),
        ("adaptive", "Adaptive (EMA)", "#d62728", "D"),
    ]:
        row = next(item for item in summary if item[0] == key)
        ax.errorbar(
            row[4],
            row[1],
            yerr=row[2],
            fmt=marker,
            markersize=8,
            capsize=4,
            color=color,
            label=label,
        )
        ax.annotate(label, (row[4], row[1]), xytext=(6, 6),
                    textcoords="offset points", fontsize=9)

    ax.set_xlabel("Communication saving (%)")
    ax.set_ylabel("Final Top-1 accuracy (%)")
    ax.set_title("Communication-accuracy trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS / "cifar100_comm_pareto.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    adaptive_runs = all_runs["adaptive"]
    for idx, run in enumerate(adaptive_runs, start=1):
        rounds = np.arange(1, len(run) + 1)
        ax.plot(rounds, run["adaptive_tau"], linewidth=1.7, label=f"Repeat {idx}")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Adaptive threshold tau(t)")
    ax.set_title("Adaptive threshold trajectories (CIFAR-100)")
    ax.set_xticks(np.arange(1, 21, 2))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "cifar100_adaptive_threshold.png", dpi=200)
    plt.close(fig)

    print("Method, Final Top-1, Std, Upload ratio, Communication saving")
    for key, final_mean, final_std, upload, saving in summary:
        print(f"{key}, {final_mean:.2f}, {final_std:.2f}, {upload:.1f}, {saving:.1f}")


if __name__ == "__main__":
    main()
