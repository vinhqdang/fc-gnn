"""FC-GNN: Fuzzy-Conformal Graph Neural Network main model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm

from .fuzzy_layer import FuzzyMessagePassingLayer, SugenoFuzzyIntegral


class FCGNN(nn.Module):
    """
    Fuzzy-Conformal Graph Neural Network.
    - L fuzzy message-passing layers
    - MLP output head → fuzzy class memberships
    - Sugeno integral module for CP nonconformity scoring
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 3, n_rules: int = 16, dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.n_rules = n_rules

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Fuzzy message-passing layers
        self.fmpl_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.fmpl_layers.append(
                FuzzyMessagePassingLayer(hidden_channels, hidden_channels, n_rules)
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels))

        # Output MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        self.dropout = nn.Dropout(dropout)

        # Sugeno integral for CP scoring
        self.sugeno = SugenoFuzzyIntegral(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_embeddings: bool = False):
        """
        Args:
            x: [N, in_channels]
            edge_index: [2, E]
            return_embeddings: if True, also return node embeddings before head
        Returns:
            mu: [N, C] fuzzy membership (softmax output)
            embeddings (optional): [N, hidden]
        """
        h = F.relu(self.input_proj(x))
        h = self.dropout(h)

        for fmpl, bn in zip(self.fmpl_layers, self.bn_layers):
            h_new = fmpl(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual

        logits = self.head(h)
        mu = F.softmax(logits, dim=-1)  # fuzzy membership distribution

        if return_embeddings:
            return mu, h
        return mu

    def get_nonconformity_scores(self, x: torch.Tensor, edge_index: torch.Tensor,
                                  y: torch.Tensor) -> torch.Tensor:
        """Compute fuzzy nonconformity scores for calibration nodes."""
        mu = self.forward(x, edge_index)
        # s_v = 1 - S_λ(μ_v, y_v)
        s_integral = self.sugeno(mu, y)
        return 1.0 - s_integral

    def get_prediction_set_scores(self, x: torch.Tensor,
                                   edge_index: torch.Tensor) -> torch.Tensor:
        """Return per-class Sugeno scores for prediction set construction."""
        mu = self.forward(x, edge_index)
        all_scores = self.sugeno(mu)   # [N, C]
        return 1.0 - all_scores        # nonconformity per class

    def l1_regularization(self) -> torch.Tensor:
        return sum(layer.l1_regularization() for layer in self.fmpl_layers)

    def get_fired_rules(self, x: torch.Tensor, edge_index: torch.Tensor,
                        top_k: int = 3) -> torch.Tensor:
        """Return top-K fired rule indices per node for interpretability."""
        h = F.relu(self.input_proj(x))
        all_rule_firings = []
        for fmpl, bn in zip(self.fmpl_layers, self.bn_layers):
            mu = fmpl.membership(h)  # [N, K]
            all_rule_firings.append(mu)
            h_new = fmpl(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h = h + h_new
        # Average rule firing strengths across layers
        avg_firing = torch.stack(all_rule_firings, dim=0).mean(dim=0)  # [N, K]
        top_rules = avg_firing.topk(top_k, dim=-1).indices             # [N, top_k]
        return top_rules, avg_firing
