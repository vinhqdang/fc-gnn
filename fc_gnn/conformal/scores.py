"""Nonconformity score functions for conformal prediction."""

import torch
import numpy as np


class StandardScore:
    """Standard softmax nonconformity score: s(x, y) = 1 - p_y(x)."""

    def __call__(self, probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: [N, C] softmax probabilities
            y: [N] true class labels
        Returns:
            scores: [N] nonconformity scores
        """
        return 1.0 - probs[torch.arange(len(y)), y]

    def prediction_scores(self, probs: torch.Tensor) -> torch.Tensor:
        """Per-class scores for prediction set: [N, C]."""
        return 1.0 - probs


class FuzzyScore:
    """
    Sugeno-integral fuzzy nonconformity score.
    Wraps the model's built-in Sugeno scoring.
    """

    def __call__(self, model, x: torch.Tensor, edge_index: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.get_nonconformity_scores(x, edge_index, y)

    def prediction_scores(self, model, x: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.get_prediction_set_scores(x, edge_index)
