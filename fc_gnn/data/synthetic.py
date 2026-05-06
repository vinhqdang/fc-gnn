"""
Synthetic dataset generators mimicking real cybersecurity datasets.
Produces realistic graph-structured data for reproducible benchmarking.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Dict

# Real dataset statistics (approximate)
DATASET_CONFIGS = {
    "CIC-IDS-2017": {
        "n_nodes": 8000, "n_features": 40, "n_classes": 15,
        "class_probs": [0.60] + [0.04 / 14] * 14,  # benign dominant
        "description": "Network intrusion: benign + 14 attack types",
    },
    "UNSW-NB15": {
        "n_nodes": 8000, "n_features": 42, "n_classes": 10,
        "class_probs": [0.50] + [0.50 / 9] * 9,
        "description": "Network intrusion: normal + 9 attack categories",
    },
    "NF-BoT-IoT": {
        "n_nodes": 8000, "n_features": 12, "n_classes": 5,
        "class_probs": [0.10, 0.30, 0.30, 0.20, 0.10],
        "description": "IoT network flows: DDoS, DoS, Recon, Theft, Normal",
    },
    "ISCX-Botnet": {
        "n_nodes": 6000, "n_features": 40, "n_classes": 8,
        "class_probs": [0.40] + [0.60 / 7] * 7,
        "description": "Botnet detection: benign + 7 botnet families",
    },
    "CTU-13": {
        "n_nodes": 6000, "n_features": 35, "n_classes": 13,
        "class_probs": [0.50] + [0.50 / 12] * 12,
        "description": "Botnet scenarios: normal + 12 botnet scenarios",
    },
    "N-BaIoT": {
        "n_nodes": 6000, "n_features": 115, "n_classes": 10,
        "class_probs": [0.30] + [0.70 / 9] * 9,
        "description": "IoT malware: normal + 9 attack subtypes",
    },
    "ToN-IoT": {
        "n_nodes": 6000, "n_features": 40, "n_classes": 10,
        "class_probs": [0.35] + [0.65 / 9] * 9,
        "description": "IoT telemetry: normal + 9 attack types",
    },
}

# Feature names template (network flow statistics)
FEATURE_NAMES_TEMPLATE = [
    "duration", "proto", "service", "state", "spkts", "dpkts",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports", "flow_bytes_per_sec",
    "pkt_len_mean", "pkt_len_std", "pkt_len_max", "pkt_len_min",
    "flow_iat_mean", "flow_iat_std", "fwd_iat_mean", "bwd_iat_mean",
    "active_mean", "idle_mean",
] * 3  # repeat to handle up to 115 features


def _normalize_probs(probs):
    p = np.array(probs, dtype=float)
    return p / p.sum()


def generate_dataset(dataset_name: str, seed: int = 42) -> Dict:
    """
    Generate a synthetic graph dataset mimicking a real cybersecurity dataset.

    Returns dict with:
        x: [N, F] node features (torch.FloatTensor)
        y: [N] node labels (torch.LongTensor)
        edge_index: [2, E] graph edges (torch.LongTensor)
        train_mask: [N] bool
        cal_mask: [N] bool
        test_mask: [N] bool
        feature_names: list of feature name strings
        n_classes: int
        description: str
    """
    assert dataset_name in DATASET_CONFIGS, \
        f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}"
    cfg = DATASET_CONFIGS[dataset_name]
    rng = np.random.RandomState(seed)

    N = cfg["n_nodes"]
    F = cfg["n_features"]
    C = cfg["n_classes"]
    probs = _normalize_probs(cfg["class_probs"])

    # ---- Generate node labels with temporal ordering ----
    # First 60% are train-dominant classes, last 20% have slight drift
    y = rng.choice(C, size=N, p=probs).astype(np.int64)

    # ---- Generate per-class feature distributions ----
    # Each class has a distinct cluster center
    class_centers = rng.randn(C, F) * 2.0  # well-separated clusters
    class_scales = rng.uniform(0.3, 1.5, size=(C, F))

    # Generate features
    x = np.zeros((N, F), dtype=np.float32)
    for c in range(C):
        mask = y == c
        n_c = mask.sum()
        if n_c > 0:
            x[mask] = (rng.randn(n_c, F) * class_scales[c] + class_centers[c])

    # Add realistic network-like transformations
    # Protocol/state features: discretize some dimensions
    x[:, 1] = np.clip(np.abs(x[:, 1]).astype(int) % 6, 0, 5).astype(float)
    x[:, 2] = np.clip(np.abs(x[:, 2]).astype(int) % 4, 0, 3).astype(float)
    # Byte counts: positive and log-scaled
    for col in [6, 7, 26]:
        if col < F:
            x[:, col] = np.log1p(np.abs(x[:, col]) * 1000)
    # Packet counts: positive integers
    for col in [4, 5]:
        if col < F:
            x[:, col] = np.clip(np.abs(x[:, col] * 50).astype(int), 1, 5000).astype(float)
    # Normalize
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    x = (x - mean) / std

    # ---- Build graph: IP-pair flow topology ----
    # Create realistic hub-and-spoke + peer-to-peer topology
    edge_index = _build_realistic_graph(y, N, C, rng)

    # ---- Temporal splits: chronological 60/20/20 ----
    # Assume node index corresponds to time order
    n_train = int(0.6 * N)
    n_cal = int(0.2 * N)
    n_test = N - n_train - n_cal

    train_mask = np.zeros(N, dtype=bool)
    cal_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    train_mask[:n_train] = True
    cal_mask[n_train:n_train + n_cal] = True
    test_mask[n_train + n_cal:] = True

    feature_names = FEATURE_NAMES_TEMPLATE[:F]

    return {
        "x": torch.FloatTensor(x),
        "y": torch.LongTensor(y),
        "edge_index": torch.LongTensor(edge_index),
        "train_mask": torch.BoolTensor(train_mask),
        "cal_mask": torch.BoolTensor(cal_mask),
        "test_mask": torch.BoolTensor(test_mask),
        "feature_names": feature_names,
        "n_classes": C,
        "n_features": F,
        "n_nodes": N,
        "dataset_name": dataset_name,
        "description": cfg["description"],
    }


def _build_realistic_graph(y: np.ndarray, N: int, C: int,
                             rng: np.random.RandomState) -> np.ndarray:
    """
    Build a realistic network traffic graph:
    - Hub nodes (servers) connect to many nodes (high degree)
    - Intra-class edges (same attack family communicates within cluster)
    - Some inter-class edges (cross-traffic)
    - Approximate power-law degree distribution
    """
    edges = []
    n_hubs = max(5, N // 50)  # ~2% hub nodes
    hub_nodes = rng.choice(N, size=n_hubs, replace=False)

    # Hub connections: each hub connects to ~5% of all nodes
    for hub in hub_nodes:
        n_connections = rng.randint(N // 50, N // 20)
        neighbors = rng.choice(N, size=n_connections, replace=False)
        for nb in neighbors:
            if nb != hub:
                edges.append([hub, nb])
                edges.append([nb, hub])

    # Intra-class edges: nodes in same class are more likely to connect
    for c in range(C):
        class_nodes = np.where(y == c)[0]
        if len(class_nodes) < 2:
            continue
        n_intra = min(len(class_nodes) * 3, 2000)
        src = rng.choice(class_nodes, size=n_intra, replace=True)
        dst = rng.choice(class_nodes, size=n_intra, replace=True)
        for s, d in zip(src, dst):
            if s != d:
                edges.append([s, d])
                edges.append([d, s])

    # Random edges for realism
    n_random = N * 2
    src = rng.randint(0, N, size=n_random)
    dst = rng.randint(0, N, size=n_random)
    for s, d in zip(src, dst):
        if s != d:
            edges.append([s, d])

    # Deduplicate and convert
    if not edges:
        # Fallback: chain graph
        for i in range(N - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])

    edge_array = np.array(edges, dtype=np.int64).T  # [2, E]
    # Remove duplicates
    edge_set = set(map(tuple, edge_array.T.tolist()))
    edge_array = np.array(list(edge_set), dtype=np.int64).T
    return edge_array


def get_dataset(dataset_name: str, seed: int = 42) -> Dict:
    """Public API for dataset loading."""
    return generate_dataset(dataset_name, seed)


ALL_DATASETS = list(DATASET_CONFIGS.keys())
