"""
AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction.
Reference: Hettige et al., 2024 (WWW)

Key idea: Model two physical transport processes explicitly:
  • Diffusion  – pollutants spread from high- to low-concentration areas
                 modelled via the symmetric graph Laplacian (L_sym = I - D^{-1/2} A D^{-1/2})
  • Advection  – wind carries pollutants from upwind to downwind stations
                 modelled via the directed wind-aware adjacency (adj_wind)

The physics blocks act as inductive bias; a neural residual corrects
for non-physical dynamics the equations cannot capture.

Architecture:
    Encoder GRU  (encode history [B,T,N,F] → node states [B,N,H])
    ↓
    Physics Simulator  (diffusion + advection, K steps, Euler integration)
    ↓
    Decoder GRU cell  (residual correction, unrolled pred_len steps)
    ↓
    Linear output head  → [B, pred_len, N, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── physics blocks ────────────────────────────────────────────────────────────

class DiffusionBlock(nn.Module):
    """
    Graph diffusion:  ΔC = κ · L_sym · C
    L_sym = I − D^{-1/2} A D^{-1/2}  (captures spread from high→low areas)

    κ is a learnable scalar; a sigmoid gate modulates the contribution
    so the model can suppress diffusion when data suggests otherwise.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.log_kappa = nn.Parameter(torch.tensor(-2.0))  # init κ ≈ 0.135
        self.gate      = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        # h: [B, N, H],  laplacian: [N, N]
        kappa = torch.exp(self.log_kappa).clamp(max=1.0)
        B = h.size(0)
        L = laplacian.unsqueeze(0).expand(B, -1, -1)
        diff = kappa * torch.bmm(L, h)                  # [B, N, H]
        return diff * torch.sigmoid(self.gate(h))        # gated output


class AdvectionBlock(nn.Module):
    """
    Wind-driven advection:  ΔC = α · A_wind · (W_v · C)
    A_wind is the directed wind-aware adjacency.

    The current wind vectors (U, V) modulate the transported values
    so advection strength scales with actual wind speed.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(-2.0))   # init α ≈ 0.135
        self.wind_gate  = nn.Linear(2, hidden_dim)           # (U, V) → H
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj   = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        adj_wind: torch.Tensor,
        wind_uv: torch.Tensor,
    ) -> torch.Tensor:
        # h:        [B, N, H]
        # adj_wind: [N, N]
        # wind_uv:  [B, N, 2]  (wind_u, wind_v from feature indices 8 & 9)
        alpha = torch.exp(self.log_alpha).clamp(max=1.0)
        B = h.size(0)
        A = adj_wind.unsqueeze(0).expand(B, -1, -1)

        wind_mod = torch.sigmoid(self.wind_gate(wind_uv))   # [B, N, H]
        v        = self.value_proj(h) * wind_mod             # [B, N, H]
        adv      = alpha * torch.bmm(A, v)                   # [B, N, H]
        return self.out_proj(adv)


# ── main model ────────────────────────────────────────────────────────────────

class AirPhyNet(nn.Module):
    """
    AirPhyNet.

    Args:
        num_nodes         : N  (monitoring stations)
        in_features       : F  (10 in our dataset)
        hidden_dim        : H
        pred_len          : output horizon
        num_physics_steps : Euler integration steps K
        dropout           : dropout probability
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_dim: int = 64,
        pred_len: int = 24,
        num_physics_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len          = pred_len
        self.num_physics_steps = num_physics_steps

        # ── encoder ───────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
        )
        self.encoder_gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )

        # ── physics blocks ────────────────────────────────────────
        self.diffusion = DiffusionBlock(hidden_dim)
        self.advection = AdvectionBlock(hidden_dim)

        # Learned gate: how much physics correction to apply
        self.physics_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.physics_norm = nn.LayerNorm(hidden_dim)

        # ── decoder (residual GRU cell, unrolled) ─────────────────
        self.decoder_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # ── output ────────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _build_laplacian(adj_mx: torch.Tensor) -> torch.Tensor:
        """Symmetric normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}."""
        N = adj_mx.size(0)
        A = adj_mx + torch.eye(N, device=adj_mx.device) * 1e-4
        d = A.sum(-1).clamp(min=1e-6)
        d_inv_sqrt = d.pow(-0.5)
        A_norm = d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)
        return torch.eye(N, device=adj_mx.device) - A_norm      # [N, N]

    def forward(
        self,
        x: torch.Tensor,
        adj_mx: torch.Tensor,
        adj_wind: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        """
        Args:
            x        : [B, T, N, F]   normalised historical features
            adj_mx   : [N, N]          symmetric distance adjacency (diffusion)
            adj_wind : [N, N]          directed wind adjacency (advection)

        Returns:
            [B, pred_len, N, 1]  normalised PM2.5 predictions
        """
        B, T, N, F = x.shape
        H = self.encoder_gru.hidden_size

        laplacian = self._build_laplacian(adj_mx)     # [N, N]

        # Wind vectors at the last observed timestep
        wind_uv = x[:, -1, :, 8:10]                  # [B, N, 2]

        # ── 1. encode history ─────────────────────────────────────
        h = self.input_proj(x)                        # [B, T, N, H]
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, H)
        _, enc_state = self.encoder_gru(h)
        h_node = enc_state[-1].reshape(B, N, H)       # [B, N, H]

        # ── 2. physics-guided simulation (K Euler steps) ──────────
        h_phys = h_node
        for _ in range(self.num_physics_steps):
            diff    = self.diffusion(h_phys, laplacian)
            adv     = self.advection(h_phys, adj_wind, wind_uv)
            physics = diff + adv                          # [B, N, H]
            gate    = self.physics_gate(torch.cat([h_phys, physics], dim=-1))
            h_phys  = self.physics_norm(h_phys + gate * physics)

        # ── 3. residual decoder: unroll pred_len steps ────────────
        preds  = []
        h_dec  = h_phys.reshape(B * N, H)              # [B*N, H]
        for _ in range(self.pred_len):
            h_dec = self.decoder_norm(self.decoder_cell(h_dec, h_dec))
            out   = self.output_head(h_dec)             # [B*N, 1]
            preds.append(out.reshape(B, N, 1))

        return torch.stack(preds, dim=1)                # [B, pred_len, N, 1]
