"""Conformal prediction evaluation metrics."""

import numpy as np
from typing import Dict


def compute_cp_metrics(pred_sets: np.ndarray, true_labels: np.ndarray,
                        communities: np.ndarray = None,
                        alpha: float = 0.1) -> Dict[str, float]:
    """
    Compute all CP evaluation metrics.

    Args:
        pred_sets: [N, C] boolean prediction set membership
        true_labels: [N] ground truth labels
        communities: [N] community assignments (optional, for conditional coverage)
        alpha: target miscoverage rate

    Returns:
        dict of metric name → value
    """
    N, C = pred_sets.shape
    target_coverage = 1.0 - alpha

    # Coverage: fraction of nodes where true label is in prediction set
    covered = pred_sets[np.arange(N), true_labels]
    empirical_coverage = covered.mean()

    # Average prediction set size
    set_sizes = pred_sets.sum(axis=1)
    apss = set_sizes.mean()

    # Singleton hit proportion: |set| == 1 and correct
    singletons = (set_sizes == 1)
    shp = (singletons & covered).mean()

    metrics = {
        "coverage": float(empirical_coverage),
        "apss": float(apss),
        "shp": float(shp),
        "coverage_gap": 0.0,
        "cond_coverage_min": float(empirical_coverage),
        "cond_coverage_max": float(empirical_coverage),
        "n_communities": 1,
    }

    if communities is not None:
        unique_communities = np.unique(communities)
        cond_coverages = []
        for m in unique_communities:
            mask = communities == m
            if mask.sum() < 3:
                continue
            cov_m = pred_sets[mask][np.arange(mask.sum()), true_labels[mask]].mean()
            cond_coverages.append(float(cov_m))

        if cond_coverages:
            gaps = [abs(c - target_coverage) for c in cond_coverages]
            metrics["coverage_gap"] = float(np.max(gaps))
            metrics["cond_coverage_min"] = float(np.min(cond_coverages))
            metrics["cond_coverage_max"] = float(np.max(cond_coverages))
            metrics["n_communities"] = len(cond_coverages)

    return metrics
