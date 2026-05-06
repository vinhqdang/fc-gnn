"""Community detection utilities for Mondrian CP partitioning."""

import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import to_networkx


def detect_communities(edge_index: torch.Tensor, n_nodes: int,
                        method: str = "louvain",
                        min_community_size: int = 5) -> np.ndarray:
    """
    Detect graph communities for Mondrian CP partitioning.

    Args:
        edge_index: [2, E] PyG edge index
        n_nodes: number of nodes
        method: 'louvain' or 'label_propagation'
        min_community_size: merge small communities into nearest large one

    Returns:
        community_labels: [N] integer community assignment per node
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    edges = edge_index.t().cpu().numpy()
    G.add_edges_from(edges)

    if method == "louvain":
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, random_state=42)
            labels = np.array([partition.get(i, 0) for i in range(n_nodes)])
        except Exception:
            labels = _label_propagation(G, n_nodes)
    elif method == "label_propagation":
        labels = _label_propagation(G, n_nodes)
    else:
        raise ValueError(f"Unknown community method: {method}")

    labels = _merge_small_communities(labels, min_community_size)
    return labels


def _label_propagation(G: nx.Graph, n_nodes: int) -> np.ndarray:
    """Fallback: networkx label propagation."""
    try:
        communities = nx.community.label_propagation_communities(G)
        labels = np.zeros(n_nodes, dtype=int)
        for cid, community in enumerate(communities):
            for node in community:
                labels[node] = cid
        return labels
    except Exception:
        # Final fallback: random assignment to sqrt(N) communities
        n_communities = max(2, int(np.sqrt(n_nodes)))
        return np.random.randint(0, n_communities, size=n_nodes)


def _merge_small_communities(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Merge communities smaller than min_size into community 0."""
    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)
    large_communities = unique[counts >= min_size]
    if len(large_communities) == 0:
        return np.zeros_like(labels)
    for c in unique:
        if counts[unique == c][0] < min_size:
            labels[labels == c] = large_communities[0]
    # Re-index
    unique_new = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique_new)}
    return np.array([remap[l] for l in labels])


def assign_to_community(node_features: torch.Tensor,
                         community_labels: np.ndarray,
                         cal_features: torch.Tensor,
                         cal_communities: np.ndarray) -> np.ndarray:
    """
    Assign test nodes to nearest calibration community by feature centroid.

    Args:
        node_features: [N_test, D] test node features
        community_labels: placeholder (unused, kept for API compat)
        cal_features: [N_cal, D] calibration node features
        cal_communities: [N_cal] calibration community assignments

    Returns:
        test_communities: [N_test] community assignment for test nodes
    """
    n_communities = int(cal_communities.max()) + 1
    # Compute per-community centroids from calibration
    centroids = []
    for c in range(n_communities):
        mask = cal_communities == c
        if mask.sum() > 0:
            centroid = cal_features[mask].mean(dim=0)
        else:
            centroid = cal_features.mean(dim=0)
        centroids.append(centroid)
    centroids = torch.stack(centroids, dim=0)  # [M, D]

    # Assign test nodes to nearest centroid
    node_features = node_features.to(centroids.device)
    dists = torch.cdist(node_features.float(), centroids.float())  # [N_test, M]
    return dists.argmin(dim=1).cpu().numpy()
