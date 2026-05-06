"""Mondrian Conformal Prediction for graph-structured data."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class MondrianCP:
    """
    Marginal Conformal Prediction (for CF-GNN, DAPS, SNAPS baselines).
    Uses a single global quantile — no community conditioning.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile: Optional[float] = None

    def calibrate(self, cal_scores: np.ndarray) -> None:
        """Compute the (1-alpha) quantile of calibration scores."""
        n = len(cal_scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.quantile = float(np.quantile(cal_scores, level))

    def predict(self, test_scores: np.ndarray) -> np.ndarray:
        """
        Construct prediction sets.
        Args:
            test_scores: [N_test, C] per-class nonconformity scores
        Returns:
            pred_sets: [N_test, C] boolean prediction set membership
        """
        assert self.quantile is not None, "Call calibrate() first."
        return test_scores <= self.quantile


class FuzzyMondrianCP:
    """
    Fuzzy Mondrian Conformal Prediction (FC-GNN, RR-GNN).
    Partition calibration nodes into communities and compute per-community quantiles.
    Achieves conditional coverage guarantee within each community.
    """

    def __init__(self, alpha: float = 0.1, min_community_size: int = 5):
        self.alpha = alpha
        self.min_community_size = min_community_size
        self.community_quantiles: Dict[int, float] = {}
        self.global_quantile: Optional[float] = None
        self.n_communities: int = 0

    def calibrate(self, cal_scores: np.ndarray, cal_communities: np.ndarray) -> None:
        """
        Compute per-community quantiles.
        Args:
            cal_scores: [N_cal] nonconformity scores for true labels
            cal_communities: [N_cal] community assignment per calibration node
        """
        communities = np.unique(cal_communities)
        self.n_communities = len(communities)

        # Global fallback quantile
        n = len(cal_scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.global_quantile = float(np.quantile(cal_scores, min(level, 1.0)))

        for m in communities:
            mask = cal_communities == m
            s_m = cal_scores[mask]
            n_m = len(s_m)
            if n_m >= self.min_community_size:
                level_m = np.ceil((n_m + 1) * (1 - self.alpha)) / n_m
                self.community_quantiles[int(m)] = float(
                    np.quantile(s_m, min(level_m, 1.0))
                )
            else:
                # Small community: use global quantile
                self.community_quantiles[int(m)] = self.global_quantile

    def predict(self, test_scores: np.ndarray,
                test_communities: np.ndarray) -> np.ndarray:
        """
        Construct community-conditional prediction sets.
        Args:
            test_scores: [N_test, C] per-class nonconformity scores
            test_communities: [N_test] community assignment per test node
        Returns:
            pred_sets: [N_test, C] boolean prediction set membership
        """
        assert self.community_quantiles, "Call calibrate() first."
        N, C = test_scores.shape
        pred_sets = np.zeros((N, C), dtype=bool)
        for i in range(N):
            m = int(test_communities[i])
            q = self.community_quantiles.get(m, self.global_quantile)
            pred_sets[i] = test_scores[i] <= q
        return pred_sets

    def coverage_gap(self, test_scores: np.ndarray, test_labels: np.ndarray,
                     test_communities: np.ndarray) -> float:
        """Compute worst-case coverage gap across communities."""
        pred_sets = self.predict(test_scores, test_communities)
        target_coverage = 1.0 - self.alpha
        gaps = []
        for m in np.unique(test_communities):
            mask = test_communities == m
            if mask.sum() < self.min_community_size:
                continue
            covered = pred_sets[mask][np.arange(mask.sum()), test_labels[mask]]
            cov_m = covered.mean()
            gaps.append(abs(cov_m - target_coverage))
        return float(np.max(gaps)) if gaps else 0.0
