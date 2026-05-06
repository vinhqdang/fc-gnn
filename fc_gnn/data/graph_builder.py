"""Utilities for converting tabular flow data to graph format."""

import numpy as np
import torch
from typing import Optional


def build_graph(x: np.ndarray, y: np.ndarray,
                edge_index: Optional[np.ndarray] = None,
                window_size: int = 60) -> dict:
    """
    Convert tabular data to PyG-compatible dict.
    If no edge_index provided, build a k-NN graph from features.
    """
    if edge_index is None:
        from sklearn.neighbors import kneighbors_graph
        kg = kneighbors_graph(x, n_neighbors=10, mode="connectivity",
                               include_self=False)
        kg = kg.tocoo()
        edge_index = np.stack([kg.row, kg.col], axis=0).astype(np.int64)

    return {
        "x": torch.FloatTensor(x),
        "y": torch.LongTensor(y),
        "edge_index": torch.LongTensor(edge_index),
    }
