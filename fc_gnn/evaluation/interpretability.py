"""Rule extraction and interpretability metrics for FC-GNN."""

import numpy as np
import torch
from typing import Dict, List, Tuple


# MITRE ATT&CK mapping: attack class index → attack family
MITRE_MAPPING = {
    0: "Benign",
    1: "T1046 Network Scanning",
    2: "T1110 Brute Force",
    3: "T1498 DDoS",
    4: "T1499 DoS",
    5: "T1071 C2 Communication",
    6: "T1041 Data Exfiltration",
    7: "T1078 Valid Accounts",
    8: "T1566 Phishing",
    9: "T1595 Active Reconnaissance",
    10: "T1048 Exfiltration Alt Channel",
    11: "T1190 Exploit Public App",
    12: "T1059 Command Execution",
}


def extract_rules(model, x: torch.Tensor, edge_index: torch.Tensor,
                  top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract top-K fired rules per node.
    Returns:
        rule_indices: [N, top_k] indices of top fired rules
        rule_strengths: [N, K] firing strengths for all rules
    """
    model.eval()
    with torch.no_grad():
        top_rules, avg_firing = model.get_fired_rules(x, edge_index, top_k=top_k)
    return top_rules.cpu().numpy(), avg_firing.cpu().numpy()


def compute_interpretability_metrics(rule_indices: np.ndarray,
                                      rule_strengths: np.ndarray,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      n_rules: int) -> Dict[str, float]:
    """
    Compute interpretability metrics: fidelity, complexity, coverage, stability.

    Args:
        rule_indices: [N, top_k] top fired rule indices
        rule_strengths: [N, K] all rule firing strengths
        y_true: [N] ground truth labels
        y_pred: [N] predicted labels
        n_rules: total number of rules K

    Returns:
        dict of interpretability metrics
    """
    N, top_k = rule_indices.shape

    # Rule Fidelity: top-1 rule consistent with prediction
    # We define consistency as: rule index (mod n_classes) matches predicted class
    # (simplified since we don't have explicit rule semantics)
    n_classes = max(y_true.max() + 1, y_pred.max() + 1)
    top1_rules = rule_indices[:, 0]
    rule_class_mapping = top1_rules % n_classes
    fidelity = (rule_class_mapping == y_pred).mean()

    # Rule Complexity: mean number of antecedents per active rule
    # We use number of rules with firing strength > threshold as complexity proxy
    threshold = 0.1
    active_rules_per_node = (rule_strengths > threshold).sum(axis=1)
    complexity = active_rules_per_node.mean()

    # Rule Coverage: fraction of nodes covered by top-10 globally most active rules
    global_importance = rule_strengths.mean(axis=0)  # [K]
    top10_rules = np.argsort(global_importance)[-10:][::-1]
    covered_by_top10 = np.isin(rule_indices[:, 0], top10_rules).mean()

    # Explanation Stability: mean Jaccard similarity between rule sets of similar nodes
    # Computed on random pairs from same predicted class
    stability_scores = []
    for cls in np.unique(y_pred):
        mask = y_pred == cls
        if mask.sum() < 2:
            continue
        class_rules = rule_indices[mask]  # [N_cls, top_k]
        # Sample pairs
        n_pairs = min(50, len(class_rules))
        idx_a = np.random.choice(len(class_rules), n_pairs, replace=True)
        idx_b = np.random.choice(len(class_rules), n_pairs, replace=True)
        for a, b in zip(idx_a, idx_b):
            if a == b:
                continue
            set_a = set(class_rules[a])
            set_b = set(class_rules[b])
            union = len(set_a | set_b)
            intersection = len(set_a & set_b)
            if union > 0:
                stability_scores.append(intersection / union)

    stability = float(np.mean(stability_scores)) if stability_scores else 0.0

    return {
        "rule_fidelity": float(fidelity),
        "rule_complexity": float(complexity),
        "rule_coverage": float(covered_by_top10),
        "explanation_stability": stability,
    }


def format_rule_explanation(rule_idx: int, firing_strength: float,
                              feature_names: List[str],
                              membership_centers: np.ndarray) -> str:
    """Format a human-readable IF-THEN rule."""
    n_features = len(feature_names)
    feature_idx = rule_idx % n_features
    feature_name = feature_names[feature_idx]
    center = membership_centers[rule_idx % len(membership_centers), feature_idx]
    return (f"IF {feature_name} ≈ {center:.3f} "
            f"THEN firing_strength={firing_strength:.3f}")
