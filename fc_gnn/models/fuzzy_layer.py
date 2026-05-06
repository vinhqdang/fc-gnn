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
    S_λ(μ_v, y_v) = sup_{A ⊆ C} [min(min_{c∈A} μ_v(c), g_λ(A))]

    Computed efficiently via sorted membership values:
    1. Sort μ_v descending: μ_(1) ≥ μ_(2) ≥ ... ≥ μ_(C)
    2. Build prefix λ-fuzzy measure g_λ({c_1,...,c_k}) recursively
    3. Integral = max_k min(μ_(k), g_λ({c_1,...,c_k}))

    For per-class nonconformity: S_λ(μ_v, c) uses μ_v[c] as the
    class-specific evidence score, with the integral providing a
    neighbourhood-aware correction.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    @property
    def lambda_val(self) -> torch.Tensor:
        return torch.tanh(self.log_lambda) * 4.0  # λ ∈ (-4, 4)

    def _sugeno_integral(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Core Sugeno integral over sorted membership values.
        Args:
            mu: [B, C] membership values (arbitrary order)
        Returns:
            [B] integral values in [0,1]
        """
        sorted_mu, _ = mu.sort(dim=1, descending=True)  # [B, C]
        lam = self.lambda_val
        # Build cumulative λ-measure
        g = sorted_mu[:, 0:1].clone()  # g({c_1}) = μ_(1) scaled
        min_vals = [torch.min(sorted_mu[:, 0:1], g)]
        for k in range(1, sorted_mu.size(1)):
            mu_k = sorted_mu[:, k:k+1]
            # g_λ(A ∪ {c_{k+1}}) = g(A) + g({c_{k+1}}) + λ·g(A)·g({c_{k+1}})
            g = g + mu_k + lam * g * mu_k
            g = g.clamp(0, 1)
            min_vals.append(torch.min(sorted_mu[:, k:k+1], g))
        min_stack = torch.cat(min_vals, dim=1)  # [B, C]
        return min_stack.max(dim=1).values       # [B]

    def forward(self, mu: torch.Tensor, target_class: torch.Tensor = None) -> torch.Tensor:
        """
        Compute class-specific Sugeno confidence scores.

        Args:
            mu: [B, C] fuzzy membership distribution
            target_class: [B] class indices. If None, compute for all C classes.
        Returns:
            [B] scores if target_class given, else [B, C]
        """
        if target_class is not None:
            B = mu.size(0)
            # S_λ(μ_v, c) = μ_v[c] * S_λ(μ_v) / (μ_v.max() + ε)
            # → scales the global integral by the relative membership of the target class
            target_mu = mu[torch.arange(B), target_class]     # [B]
            global_integral = self._sugeno_integral(mu)        # [B]
            max_mu = mu.max(dim=1).values.clamp(min=1e-8)      # [B]
            return global_integral * (target_mu / max_mu)
        else:
            B, C = mu.shape
            global_integral = self._sugeno_integral(mu)        # [B]
            max_mu = mu.max(dim=1).values.clamp(min=1e-8)      # [B]
            # For each class c: score = integral * mu[:, c] / max_mu
            return global_integral.unsqueeze(1) * (mu / max_mu.unsqueeze(1))  # [B, C]
