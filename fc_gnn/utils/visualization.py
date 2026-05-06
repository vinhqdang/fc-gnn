"""Visualization utilities for FC-GNN results."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


def plot_results(results: Dict, output_dir: str = "results/figures") -> None:
    """Generate result plots: coverage, APSS, F1 comparisons."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    datasets = list(results.keys())
    if not datasets:
        return

    models = list(next(iter(results.values())).keys())
    colors = sns.color_palette("husl", len(models))

    # Coverage comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_to_plot = [("coverage", "Empirical Coverage (target=0.90)"),
                        ("apss", "Avg Prediction Set Size"),
                        ("macro_f1", "Macro-F1")]

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        for i, model in enumerate(models):
            vals = [results[ds].get(model, {}).get(metric, np.nan) for ds in datasets]
            ax.bar(np.arange(len(datasets)) + i * 0.1, vals,
                   width=0.09, label=model, color=colors[i], alpha=0.8)
        ax.set_xticks(np.arange(len(datasets)) + len(models) * 0.05)
        ax.set_xticklabels([ds[:10] for ds in datasets], rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, max(1.2, ax.get_ylim()[1]))
        if metric == "coverage":
            ax.axhline(0.90, color="red", linestyle="--", label="Target (90%)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/main_results.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_coverage_gap(results: Dict, output_dir: str = "results/figures") -> None:
    """Plot coverage gap across datasets."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    datasets = list(results.keys())
    models = list(next(iter(results.values())).keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    width = 0.8 / len(models)
    colors = sns.color_palette("husl", len(models))

    for i, model in enumerate(models):
        gaps = [results[ds].get(model, {}).get("coverage_gap", np.nan) for ds in datasets]
        ax.bar(x + i * width, gaps, width=width, label=model, color=colors[i], alpha=0.8)

    ax.set_xticks(x + len(models) * width / 2)
    ax.set_xticklabels([ds[:10] for ds in datasets], rotation=45, ha="right")
    ax.set_title("Coverage Gap (max |empirical - target| per community)")
    ax.set_ylabel("Coverage Gap")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coverage_gap.png", dpi=150, bbox_inches="tight")
    plt.close()
