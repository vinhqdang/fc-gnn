"""Fuzzy Message-Passing Layer (FMPL) for FC-GNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GaussianMembership(nn.Module):
    """Learnable Gaussian membership functions per feature dimension."""

    def __init__(self, in_features: int, n_rules: int):
        super().__init__()
        self.in_features = in_features
        self.n_rules = n_rules
        # Centers and widths for each rule x feature
        self.centers = nn.Parameter(torch.randn(n_rules, in_features) * 0.1)
        self.log_widths = nn.Parameter(torch.zeros(n_rules, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, D] node features
        Returns:
            memberships: [N, K] fuzzy membership per rule
        """
        widths = torch.exp(self.log_widths).clamp(min=1e-4)  # [K, D]
        # Broadcast: x [N, D] -> [N, 1, D], centers [K, D] -> [1, K, D]
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)    # [N, K, D]
        exponent = -0.5 * (diff / widths.unsqueeze(0)) ** 2  # [N, K, D]
        # Product T-norm over feature dims → firing strength per rule
        memberships = torch.exp(exponent.sum(dim=-1))         # [N, K]
        return memberships  # in (0, 1]


class FuzzyMessagePassingLayer(MessagePassing):
    """
    Fuzzy Message-Passing Layer.
    Aggregates neighbor embeddings via fuzzy rule firing strengths,
    then defuzzifies via weighted centroid.
    """

    def __init__(self, in_channels: int, out_channels: int, n_rules: int = 16,
                 aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_rules = n_rules

        # Membership functions applied to neighbor embeddings
        self.membership = GaussianMembership(in_channels, n_rules)
        # Rule consequent weights: each rule k maps to output embedding
        self.rule_weights = nn.Parameter(torch.randn(n_rules, out_channels) * 0.01)
        # L1 regularization target for rule pruning
        self.bias = nn.Parameter(torch.zeros(out_channels))
        # Linear projection for residual
        self.proj = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Compute memberships for all nodes
        mu = self.membership(x)  # [N, K]
        # Propagate firing strengths
        out = self.propagate(edge_index, x=x, mu=mu)  # [N, out]
        # Residual connection
        out = out + self.proj(x)
        return out + self.bias

    def message(self, x_j: torch.Tensor, mu_j: torch.Tensor) -> torch.Tensor:
        """
        For each edge (i←j): message is rule-weighted x_j.
        Args:
            x_j: [E, in_channels] source node features
            mu_j: [E, K] source node rule memberships
        Returns:
            [E, out_channels] messages
        """
        # Defuzzification: weighted sum over rules per edge
        # mu_j: [E, K], rule_weights: [K, out]
        weighted = torch.einsum('ek,ko->eo', mu_j, self.rule_weights)  # [E, out]
        denom = mu_j.sum(dim=-1, keepdim=True).clamp(min=1e-8)         # [E, 1]
        return weighted / denom  # [E, out]

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor,
                  ptr=None, dim_size=None) -> torch.Tensor:
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def l1_regularization(self) -> torch.Tensor:
        """L1 regularization on rule weights to prune inactive rules."""
        return self.membership.log_widths.abs().mean()


class SugenoFuzzyIntegral(nn.Module):
    """
    Sugeno-type fuzzy integral for nonconformity scoring.
    Computes S_λ(μ_v, y_v) = sup_{A ⊆ C} [min(min_{c∈A} μ_v(c), g_λ(A))]
    where g_λ is a λ-fuzzy measure.

    For efficiency, we use a learned λ parameter and compute the integral
    over sorted membership values.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        # λ parameter of the fuzzy measure (scalar, learnable)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    @property
    def lambda_val(self) -> torch.Tensor:
        return torch.tanh(self.log_lambda) * 9.99  # λ ∈ (-10, 10)

    def _lambda_measure(self, singletons: torch.Tensor) -> torch.Tensor:
        """
        Compute the λ-fuzzy measure g_λ for all subsets implied by sorted ordering.
        singletons: [B, C] per-class measure values (sorted descending in caller)
        Returns: [B, C] cumulative g_λ values
        """
        lam = self.lambda_val
        # g_λ({c1,...,ck}) via recursive formula: g(A∪B) = g(A)+g(B)+λ·g(A)·g(B)
        g = torch.zeros_like(singletons[:, :1])
        measures = []
        for i in range(singletons.size(1)):
            gi = singletons[:, i:i+1]
            g = g + gi + lam * g * gi
            measures.append(g.clone())
        return torch.cat(measures, dim=1)  # [B, C]

    def forward(self, mu: torch.Tensor, target_class: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Sugeno integral confidence score.
        Args:
            mu: [B, C] fuzzy membership distribution
            target_class: [B] target class indices (None = compute for all classes)
        Returns:
            scores: [B] Sugeno integral values (higher = more confident)
        """
        if target_class is not None:
            # For nonconformity: focus on target class ordering
            # Sort memberships descending, compute integral
            B, C = mu.shape
            # Bring target class membership to front for ordering
            target_mu = mu[torch.arange(B), target_class].unsqueeze(1)  # [B,1]
            # Sort all classes by membership descending
            sorted_mu, sorted_idx = mu.sort(dim=1, descending=True)
            # Singletons from normalized memberships
            singletons = sorted_mu.clamp(0, 1)
            g = self._lambda_measure(singletons)  # [B, C]
            # Sugeno integral: sup_k min(μ_(k), g({c_1,...,c_k}))
            min_vals = torch.min(singletons, g)   # [B, C]
            integral = min_vals.max(dim=1).values  # [B]
            # Weight by whether target class is in top sets
            # Use target class rank as confidence adjustment
            target_mu_flat = target_mu.squeeze(1)  # [B]
            return integral * (target_mu_flat / (mu.max(dim=1).values + 1e-8))
        else:
            # Compute for all classes (used in prediction set construction)
            B, C = mu.shape
            scores = []
            for c in range(C):
                target_c = torch.full((B,), c, dtype=torch.long, device=mu.device)
                scores.append(self.forward(mu, target_c))
            return torch.stack(scores, dim=1)  # [B, C]
