"""
STGNP: Spatio-Temporal Graph Neural Process for PM2.5 Forecasting.
Reference: inspired by Neural Process literature and graph-based ST forecasting.

Key idea: Frame forecasting as a Neural Process — the model learns a
*distribution* over predictions conditioned on the observed context
(historical data).  This gives calibrated uncertainty estimates alongside
point forecasts, useful for risk-aware decision making.

The NP decomposition:
  • Context encoder  : historical observations → node latent representations r
  • Spatial aggregator: GNN over r to share information between neighbours
  • Cross-attention decoder: node query embeddings attend to context r
  • Output head      : Gaussian (μ, σ²) per node per future step

Training uses Gaussian NLL so the model is penalised for both inaccuracy
and over/under-confidence.  At inference the mean is the point forecast.

Architecture:
    x [B,T,N,F]
    → per-node GRU  (temporal)
    → temporal self-attention
    → spatial graph attention  (propagate to neighbours)
    = context r [B, N, H]
    ↓
    cross-attention( node_query, r )  →  [B, N, H]
    → MLP  →  μ [B,pred_len,N,1]  +  σ² [B,pred_len,N,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── building blocks ───────────────────────────────────────────────────────────

class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention over the time axis for each node."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, T, H]
        h, _ = self.attn(x, x, x)
        return self.norm(x + h)


class SpatialGraphAttention(nn.Module):
    """
    Graph attention where edges exist only between connected stations.
    Attention weights are masked by the adjacency matrix so information
    only flows along observed spatial relationships.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: [B, N, H],  adj: [N, N]
        B, N, H = h.shape
        nh, dh = self.num_heads, self.head_dim

        Q = self.q(h).view(B, N, nh, dh).transpose(1, 2)  # [B, nh, N, dh]
        K = self.k(h).view(B, N, nh, dh).transpose(1, 2)
        V = self.v(h).view(B, N, nh, dh).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, nh, N, N]

        # Mask: only allow attention between connected nodes (adj > 0)
        connected = (adj > 0).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        attn = attn.masked_fill(~connected, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)           # handle all-inf rows
        attn = self.drop(attn)

        out = torch.matmul(attn, V)                      # [B, nh, N, dh]
        out = out.transpose(1, 2).reshape(B, N, H)
        return self.norm(h + self.out(out))


# ── main model ────────────────────────────────────────────────────────────────

class STGNP(nn.Module):
    """
    STGNP — Spatio-Temporal Graph Neural Process.

    Args:
        num_nodes   : N  (monitoring stations)
        in_features : F  (10 in our dataset)
        hidden_dim  : H
        pred_len    : output horizon
        num_layers  : GRU depth
        num_heads   : attention heads
        dropout     : dropout probability
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_dim: int = 64,
        pred_len: int = 24,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len  = pred_len

        # ── context encoder ───────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.temporal_gru  = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.temporal_attn = TemporalSelfAttention(hidden_dim, num_heads, dropout)
        self.spatial_attn  = SpatialGraphAttention(hidden_dim, num_heads, dropout)

        # ── node query embeddings (one per station) ───────────────
        self.node_embed = nn.Embedding(num_nodes, hidden_dim)

        # ── cross-attention: query nodes attend to context ────────
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # ── output heads: mean + log-variance (Gaussian NLL) ──────
        self.head_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )
        self.head_logvar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _encode_context(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode historical features to per-node context representations.

        Returns: [B, N, H]
        """
        B, T, N, F = x.shape
        H = self.temporal_gru.hidden_size

        # Project features: [B, T, N, F] → [B*N, T, H]
        h = self.input_proj(x)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, H)

        # Temporal self-attention then GRU
        h = self.temporal_attn(h)                        # [B*N, T, H]
        _, h_last = self.temporal_gru(h)                 # h_last: [L, B*N, H]
        h_node = h_last[-1].reshape(B, N, H)             # [B, N, H]

        # Spatial attention: share context with neighbours
        h_node = self.spatial_attn(h_node, adj)          # [B, N, H]
        return h_node

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        adj_mx: torch.Tensor,
        **_,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x      : [B, T, N, F]  normalised historical features
            adj_mx : [N, N]         symmetric adjacency

        Returns:
            mean : [B, pred_len, N, 1]  PM2.5 point forecast
            var  : [B, pred_len, N, 1]  predictive variance (uncertainty)
        """
        B  = x.size(0)
        N  = self.num_nodes

        # Encode context
        context = self._encode_context(x, adj_mx)        # [B, N, H]

        # Node query embeddings
        node_ids = torch.arange(N, device=x.device)
        queries  = self.node_embed(node_ids)              # [N, H]
        queries  = queries.unsqueeze(0).expand(B, -1, -1) # [B, N, H]

        # Cross-attention: each node query gathers from all context nodes
        h, _ = self.cross_attn(queries, context, context) # [B, N, H]
        h    = self.cross_norm(queries + h)               # residual

        # Output
        mean    = self.head_mean(h)       # [B, N, pred_len]
        log_var = self.head_logvar(h)     # [B, N, pred_len]
        var     = torch.exp(log_var.clamp(-8.0, 4.0))    # σ² ∈ (e^-8, e^4)

        # Reshape → [B, pred_len, N, 1]
        mean = mean.permute(0, 2, 1).unsqueeze(-1)
        var  = var.permute(0, 2, 1).unsqueeze(-1)
        return mean, var

    def nll_loss(
        self,
        x: torch.Tensor,
        adj_mx: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood:
            L = 0.5 * [log σ² + (y - μ)² / σ²]

        Args:
            target : [B, pred_len, N, 1]  normalised ground-truth PM2.5
        """
        mean, var = self.forward(x, adj_mx)
        return (0.5 * (torch.log(var) + (target - mean) ** 2 / var)).mean()
