"""
Training script for FC-GNN and all baselines.
Usage: python train.py --dataset CIC-IDS-2017 --model fc_gnn --epochs 80
"""

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path

from fc_gnn.data.synthetic import get_dataset
from fc_gnn.models.fc_gnn import FCGNN
from fc_gnn.models.baselines import (
    CFGNN, DAPSGNN, RRGNN, SNAPSGNN, FLGNN, FGATGNN, PlainGCN, PlainSAGE
)
from fc_gnn.utils.community import detect_communities, assign_to_community
from fc_gnn.conformal.mondrian_cp import MondrianCP, FuzzyMondrianCP
from fc_gnn.conformal.metrics import compute_cp_metrics
from fc_gnn.evaluation.metrics import compute_classification_metrics
from fc_gnn.evaluation.interpretability import extract_rules, compute_interpretability_metrics


MODELS = {
    "fc_gnn": FCGNN,
    "cfgnn": CFGNN,
    "daps": DAPSGNN,
    "rrgnn": RRGNN,
    "snaps": SNAPSGNN,
    "flgnn": FLGNN,
    "fgat": FGATGNN,
    "gcn": PlainGCN,
    "sage": PlainSAGE,
}

# Which models use Mondrian CP (community-conditional)
MONDRIAN_MODELS = {"fc_gnn", "rrgnn"}
# Which models use any CP
CP_MODELS = {"fc_gnn", "cfgnn", "daps", "rrgnn", "snaps"}


def get_model(model_name: str, in_channels: int, n_classes: int,
              hidden: int = 64, n_rules: int = 16) -> torch.nn.Module:
    """Instantiate a model by name."""
    kwargs = dict(in_channels=in_channels, hidden_channels=hidden,
                  out_channels=n_classes, num_layers=3)
    if model_name in ("fc_gnn", "flgnn", "fgat"):
        kwargs["n_rules"] = n_rules
    return MODELS[model_name](**kwargs)


def train_epoch(model, x, edge_index, y, train_mask, optimizer, l1_weight=1e-4):
    """Single training epoch."""
    model.train()
    optimizer.zero_grad()
    if hasattr(model, "sugeno"):  # FC-GNN
        mu = model(x, edge_index)
    else:
        out = model(x, edge_index)
        mu = out[0] if isinstance(out, tuple) else out

    loss = F.cross_entropy(mu[train_mask], y[train_mask])
    if hasattr(model, "l1_regularization"):
        loss = loss + l1_weight * model.l1_regularization()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_model(model, x, edge_index, y, mask):
    """Evaluate point-prediction performance."""
    model.eval()
    out = model(x, edge_index)
    mu = out[0] if isinstance(out, tuple) else out
    mu = mu if not hasattr(mu, 'detach') else mu
    pred = mu[mask].argmax(dim=-1).cpu().numpy()
    y_true = y[mask].cpu().numpy()
    y_prob = mu[mask].cpu().numpy()
    return compute_classification_metrics(y_true, pred, y_prob)


def train_and_evaluate(dataset_name: str, model_name: str, epochs: int = 80,
                        alpha: float = 0.1, hidden: int = 64, n_rules: int = 16,
                        seed: int = 42, verbose: bool = True) -> dict:
    """
    Full training + CP calibration + evaluation pipeline.
    Returns dict of all metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    data = get_dataset(dataset_name, seed=seed)
    x = data["x"]
    y = data["y"]
    edge_index = data["edge_index"]
    train_mask = data["train_mask"]
    cal_mask = data["cal_mask"]
    test_mask = data["test_mask"]
    n_classes = data["n_classes"]
    n_features = data["n_features"]

    if verbose:
        print(f"  Dataset: {dataset_name} | N={x.size(0)} F={n_features} C={n_classes}")
        print(f"  Train={train_mask.sum().item()} Cal={cal_mask.sum().item()} "
              f"Test={test_mask.sum().item()}")

    # Build model
    model = get_model(model_name, n_features, n_classes, hidden, n_rules)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Training
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, x, edge_index, y, train_mask, optimizer)
        scheduler.step()
        if verbose and epoch % 20 == 0:
            metrics = evaluate_model(model, x, edge_index, y, test_mask)
            print(f"  Epoch {epoch:3d} | loss={loss:.4f} | "
                  f"acc={metrics['accuracy']:.4f} | f1={metrics['macro_f1']:.4f}")

    train_time = time.time() - t0

    # Point-prediction metrics
    model.eval()
    test_metrics = evaluate_model(model, x, edge_index, y, test_mask)

    # CP Calibration & Prediction Sets
    cp_metrics = {}
    interp_metrics = {}

    if model_name in CP_MODELS:
        with torch.no_grad():
            # Nonconformity scores on calibration set: pass full graph for message passing,
            # then index cal nodes. get_nonconformity_scores computes for ALL nodes.
            cal_scores_all = model.get_nonconformity_scores(x, edge_index, y)
            cal_scores_np = cal_scores_all[cal_mask].cpu().numpy()

            # Per-class scores on test set
            test_class_scores = model.get_prediction_set_scores(x, edge_index)
            test_class_scores_np = test_class_scores[test_mask].cpu().numpy()

        # Community detection on calibration subgraph
        cal_indices = torch.where(cal_mask)[0]
        test_indices = torch.where(test_mask)[0]

        # Detect communities on full graph, restrict to cal/test nodes
        communities = detect_communities(edge_index, x.size(0), method="louvain")
        cal_communities = communities[cal_mask.cpu().numpy()]
        test_communities = communities[test_mask.cpu().numpy()]

        y_cal_np = y[cal_mask].cpu().numpy()
        y_test_np = y[test_mask].cpu().numpy()

        if model_name in MONDRIAN_MODELS:
            # Fuzzy Mondrian CP
            cp = FuzzyMondrianCP(alpha=alpha)
            cp.calibrate(cal_scores_np, cal_communities)
            pred_sets = cp.predict(test_class_scores_np, test_communities)
        else:
            # Marginal CP
            cp = MondrianCP(alpha=alpha)
            cp.calibrate(cal_scores_np)
            pred_sets = cp.predict(test_class_scores_np)

        cp_metrics = compute_cp_metrics(pred_sets, y_test_np,
                                         communities=test_communities if model_name in MONDRIAN_MODELS else None,
                                         alpha=alpha)

    # Interpretability metrics (FC-GNN only)
    if model_name == "fc_gnn":
        try:
            rule_indices, rule_strengths = extract_rules(model, x, edge_index, top_k=3)
            test_indices_np = test_mask.cpu().numpy()
            test_rule_idx = rule_indices[test_indices_np]
            test_rule_str = rule_strengths[test_indices_np]
            y_pred = model(x, edge_index).argmax(dim=-1).cpu().numpy()[test_indices_np]
            y_true_test = y[test_mask].cpu().numpy()
            interp_metrics = compute_interpretability_metrics(
                test_rule_idx, test_rule_str, y_true_test, y_pred, n_rules
            )
        except Exception as e:
            if verbose:
                print(f"  Warning: interpretability failed: {e}")
            interp_metrics = {}

    return {
        "dataset": dataset_name,
        "model": model_name,
        "train_time_s": train_time,
        **test_metrics,
        **cp_metrics,
        **interp_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CIC-IDS-2017")
    parser.add_argument("--model", default="fc_gnn",
                        choices=list(MODELS.keys()))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n_rules", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nTraining {args.model} on {args.dataset}")
    results = train_and_evaluate(
        args.dataset, args.model, args.epochs, args.alpha,
        args.hidden, args.n_rules, args.seed, verbose=True
    )
    print("\n=== Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
