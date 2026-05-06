"""Baseline GNN models for comparison with FC-GNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm


class PlainGCN(nn.Module):
    """Standard GCN — no conformal prediction, no fuzzy logic."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return F.softmax(x, dim=-1)

    def get_nonconformity_scores(self, x, edge_index, y):
        mu = self.forward(x, edge_index)
        return 1.0 - mu[torch.arange(len(y)), y]

    def get_prediction_set_scores(self, x, edge_index):
        mu = self.forward(x, edge_index)
        return 1.0 - mu


class PlainSAGE(nn.Module):
    """GraphSAGE — no CP, no fuzzy logic."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return F.softmax(x, dim=-1)

    def get_nonconformity_scores(self, x, edge_index, y):
        mu = self.forward(x, edge_index)
        return 1.0 - mu[torch.arange(len(y)), y]

    def get_prediction_set_scores(self, x, edge_index):
        mu = self.forward(x, edge_index)
        return 1.0 - mu


class CFGNN(PlainSAGE):
    """
    CF-GNN (Huang et al., NeurIPS 2023): GraphSAGE backbone + marginal CP.
    The CP calibration is handled externally by MondrianCP with marginal mode.
    """
    pass


class DAPSGNN(nn.Module):
    """
    DAPS (Zargarbashi et al., ICML 2023): GCN + diffusion-adjusted CP scores.
    Diffuses conformity scores over the graph using adjacency matrix.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return F.softmax(x, dim=-1)

    def _diffuse_scores(self, scores, edge_index, n_nodes, n_steps=2):
        """Propagate CP scores over the graph for DAPS-style adjustment."""
        from torch_geometric.utils import to_dense_adj
        adj = to_dense_adj(edge_index, max_num_nodes=n_nodes).squeeze(0)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        s = scores.clone()
        for _ in range(n_steps):
            s = (1 - self.alpha) * scores + self.alpha * (adj_norm @ s)
        return s

    def get_nonconformity_scores(self, x, edge_index, y):
        mu = self.forward(x, edge_index)
        base = 1.0 - mu[torch.arange(len(y)), y]
        # Diffuse scores
        all_scores_mat = 1.0 - mu  # [N, C]
        target_scores = base.unsqueeze(1)  # [N, 1]
        diffused = self._diffuse_scores(base.unsqueeze(1), edge_index,
                                         x.size(0)).squeeze(1)
        return diffused

    def get_prediction_set_scores(self, x, edge_index):
        mu = self.forward(x, edge_index)
        base = 1.0 - mu  # [N, C]
        # Diffuse per-class scores
        result = []
        for c in range(mu.size(1)):
            s = self._diffuse_scores(base[:, c:c+1], edge_index, x.size(0)).squeeze(1)
            result.append(s)
        return torch.stack(result, dim=1)  # [N, C]


class RRGNN(nn.Module):
    """
    RR-GNN (Zhang et al., UAI 2025): GraphSAGE + Mondrian CP with residual reweighting.
    Mondrian partitioning is handled externally; here we just provide the backbone.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = nn.Dropout(dropout)
        # Residual reweighting: per-community confidence correction
        self.residual_head = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, return_hidden=False):
        h = x
        for conv, bn in zip(self.convs[:-1], self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
        logits = self.convs[-1](h, edge_index)
        mu = F.softmax(logits, dim=-1)
        if return_hidden:
            return mu, h
        return mu

    def get_nonconformity_scores(self, x, edge_index, y):
        mu, h = self.forward(x, edge_index, return_hidden=True)
        # Residual reweighting: subtract learned correction
        correction = torch.sigmoid(self.residual_head(h)).squeeze(-1)
        base = 1.0 - mu[torch.arange(len(y)), y]
        return base * correction

    def get_prediction_set_scores(self, x, edge_index):
        mu, h = self.forward(x, edge_index, return_hidden=True)
        correction = torch.sigmoid(self.residual_head(h))  # [N, 1]
        return (1.0 - mu) * correction


class SNAPSGNN(nn.Module):
    """
    SNAPS (Song et al., NeurIPS 2024): GCN + feature-similar neighbor score aggregation.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, n_neighbors=5):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for conv, bn in zip(self.convs[:-1], self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
        logits = self.convs[-1](h, edge_index)
        return F.softmax(logits, dim=-1), h

    def _snaps_adjust(self, scores, embeddings, edge_index, n_nodes):
        """Adjust scores using feature similarity to neighbors."""
        from torch_geometric.utils import to_dense_adj
        h = F.normalize(embeddings, dim=-1)
        sim = torch.mm(h, h.t())  # [N, N] cosine similarity
        # Zero out non-edges
        adj = to_dense_adj(edge_index, max_num_nodes=n_nodes).squeeze(0)
        sim = sim * adj
        # Top-k neighbor weights
        topk = min(self.n_neighbors, sim.size(1))
        topk_vals, _ = sim.topk(topk, dim=1)
        weights = F.softmax(topk_vals, dim=1)
        # Weighted average of neighbor scores
        neighbor_scores = torch.mm(sim / (sim.sum(dim=1, keepdim=True) + 1e-8), scores)
        return 0.5 * scores + 0.5 * neighbor_scores

    def get_nonconformity_scores(self, x, edge_index, y):
        mu, h = self.forward(x, edge_index)
        base = (1.0 - mu[torch.arange(len(y)), y]).unsqueeze(1)  # [N, 1]
        adjusted = self._snaps_adjust(base, h, edge_index, x.size(0))
        return adjusted.squeeze(1)

    def get_prediction_set_scores(self, x, edge_index):
        mu, h = self.forward(x, edge_index)
        base = 1.0 - mu  # [N, C]
        return self._snaps_adjust(base, h, edge_index, x.size(0))


class FuzzyAttentionLayer(nn.Module):
    """Fuzzy attention: computes attention weights using fuzzy membership."""

    def __init__(self, in_channels, out_channels, n_rules=8):
        super().__init__()
        self.attn = GATConv(in_channels, out_channels, heads=4, concat=False,
                            dropout=0.2)
        # Fuzzy gating
        self.fuzzy_gate = nn.Sequential(
            nn.Linear(in_channels, n_rules),
            nn.Sigmoid(),
            nn.Linear(n_rules, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index):
        attn_out = self.attn(x, edge_index)
        gate = self.fuzzy_gate(x)
        return attn_out * gate + x[:, :attn_out.size(1)] if x.size(1) == attn_out.size(1) else attn_out * gate


class FLGNN(nn.Module):
    """
    FL-GNN (ICLR 2024): Fuzzy-Logic GNN — fuzzy aggregation, no CP.
    Uses simplified Cartesian-product rule base with GCN backbone.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, n_rules=16, dropout=0.3):
        super().__init__()
        from .fuzzy_layer import FuzzyMessagePassingLayer
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.fmpl_layers = nn.ModuleList([
            FuzzyMessagePassingLayer(hidden_channels, hidden_channels, n_rules)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x))
        for fmpl, bn in zip(self.fmpl_layers, self.bns):
            h_new = fmpl(h, edge_index)
            h_new = bn(F.relu(h_new))
            h = h + self.dropout(h_new)
        return F.softmax(self.head(h), dim=-1)

    def get_nonconformity_scores(self, x, edge_index, y):
        mu = self.forward(x, edge_index)
        return 1.0 - mu[torch.arange(len(y)), y]

    def get_prediction_set_scores(self, x, edge_index):
        mu = self.forward(x, edge_index)
        return 1.0 - mu


class FGATGNN(nn.Module):
    """
    FGAT (Xing et al., 2024): Fuzzy Graph Attention Network — no CP.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, n_rules=8, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList([
            FuzzyAttentionLayer(hidden_channels, hidden_channels, n_rules)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x))
        for layer, bn in zip(self.layers, self.bns):
            h_new = layer(h, edge_index)
            h_new = bn(F.relu(h_new))
            h = h + self.dropout(h_new)
        return F.softmax(self.head(h), dim=-1)

    def get_nonconformity_scores(self, x, edge_index, y):
        mu = self.forward(x, edge_index)
        return 1.0 - mu[torch.arange(len(y)), y]

    def get_prediction_set_scores(self, x, edge_index):
        mu = self.forward(x, edge_index)
        return 1.0 - mu
