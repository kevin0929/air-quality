"""
DCRNN — faithful reproduction of Li et al., ICLR 2018.
Adapted for preprocess.py output format (features.npy [T,N,F], adj_mx.npy [N,N]).

Reference architecture: chnsh/DCRNN_PyTorch (widely-used, paper-author-endorsed
PyTorch port; official repo liyaguang/DCRNN is TensorFlow-only).

Key components:
  - DCGRUCell: GRU cell where the linear transforms are replaced by diffusion
    convolution (bidirectional random-walk on the graph), instead of dense
    matrix multiplication or 1D convolution.
  - Encoder-Decoder (Seq2Seq): encode hist_len steps into a hidden state,
    then decode pred_len steps autoregressively.
  - Scheduled sampling (curriculum learning): during decoding, the decoder
    input at each step is the ground truth with probability p (which decays
    over training iterations), or the model's own previous prediction
    otherwise. This is the paper's key trick for stabilizing multi-step
    training and closing the train/inference gap.

Differences from original (document in thesis):
  1. Graph: uses preprocess.py's adj_mx.npy (Gaussian-kernel geographic
     adjacency) instead of a distance-thresholded sensor graph built from
     METR-LA sensor coordinates.
  2. Only PM2.5 (index 0) is treated as the forecasting target; other
     features are used as auxiliary input channels, not forecast outputs
     (original DCRNN forecasts the single traffic-speed channel).
  3. No TensorFlow; this is a PyTorch implementation following chnsh's port.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── graph diffusion supports ─────────────────────────────────────────────────

def calculate_random_walk_matrix(adj_mx: np.ndarray) -> np.ndarray:
    """D^{-1} A : row-normalized random walk transition matrix."""
    d = adj_mx.sum(axis=1)
    d_inv = np.where(d > 0, 1.0 / d, 0.0)
    return (adj_mx * d_inv[:, None]).astype(np.float32)


def build_supports(adj_mx: np.ndarray, filter_type: str = "dual_random_walk"):
    """
    Returns list of [N,N] np.float32 arrays used as diffusion-convolution
    supports, matching the original paper's Eq. 2-3 (bidirectional
    random-walk diffusion process).
    """
    if filter_type == "random_walk":
        return [calculate_random_walk_matrix(adj_mx)]
    elif filter_type == "dual_random_walk":
        return [
            calculate_random_walk_matrix(adj_mx),
            calculate_random_walk_matrix(adj_mx.T),
        ]
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")


# ── DCGRU cell (diffusion convolution replacing GRU's linear layers) ─────────

class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell.

    Each gate's linear transform y = Wx + b is replaced by a diffusion
    convolution: y = sum_k (support_k^p @ x) W_k + b, matching Eq. 3-5 of the
    DCRNN paper (Li et al. 2018). max_diffusion_step controls the order of
    the diffusion polynomial (K in the paper, typically 2).
    """

    def __init__(self, num_units, supports, num_nodes, input_dim,
                 max_diffusion_step=2, use_gc_for_ru=True):
        super().__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.max_diffusion_step = max_diffusion_step
        self.use_gc_for_ru = use_gc_for_ru

        self.register_buffer(
            "supports_stack",
            torch.stack([torch.FloatTensor(s) for s in supports], dim=0),
        )
        self.num_supports = len(supports)
        # +1 for identity (order-0 diffusion term)
        self.num_matrices = self.num_supports * max_diffusion_step + 1

        input_size = input_dim + num_units
        self.gconv_gate = nn.Linear(input_size * self.num_matrices, 2 * num_units)
        self.gconv_candidate = nn.Linear(input_size * self.num_matrices, num_units)

    def _diffusion_conv(self, x, linear_layer):
        """
        x: [B, N, input_size]
        Applies the K-order diffusion polynomial for every support, then a
        shared linear layer, matching the paper's graph_conv() implementation.
        """
        B, N, C = x.shape
        x0 = x.permute(1, 2, 0).reshape(N, -1)          # [N, C*B]
        outputs = [x0]                                   # order-0 term (identity)

        for support in self.supports_stack:
            x1 = torch.mm(support, x0)                    # [N, C*B]
            outputs.append(x1)
            x_prev, x_cur = x0, x1
            for _ in range(2, self.max_diffusion_step + 1):
                x2 = 2 * torch.mm(support, x_cur) - x_prev  # Chebyshev-like recursion
                outputs.append(x2)
                x_prev, x_cur = x_cur, x2

        out = torch.stack(outputs, dim=0)                 # [num_matrices, N, C*B]
        out = out.reshape(self.num_matrices, N, C, B)
        out = out.permute(3, 1, 2, 0).reshape(B, N, C * self.num_matrices)
        return linear_layer(out)

    def forward(self, x, state):
        """
        x: [B, N, input_dim]   state: [B, N, num_units]
        Returns new state [B, N, num_units]
        """
        input_and_state = torch.cat([x, state], dim=-1)   # [B, N, input_dim+num_units]

        if self.use_gc_for_ru:
            value = torch.sigmoid(self._diffusion_conv(input_and_state, self.gconv_gate))
        else:
            value = torch.sigmoid(
                nn.functional.linear(input_and_state, self.gconv_gate.weight, self.gconv_gate.bias)
            )
        r, u = torch.split(value, self.num_units, dim=-1)

        candidate_input = torch.cat([x, r * state], dim=-1)
        c = torch.tanh(self._diffusion_conv(candidate_input, self.gconv_candidate))

        new_state = u * state + (1 - u) * c
        return new_state

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.num_nodes, self.num_units, device=device)


# ── Encoder / Decoder stacks ─────────────────────────────────────────────────

class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, num_nodes, supports, hid_dim, num_layers, max_diffusion_step):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes

        self.cells = nn.ModuleList([
            DCGRUCell(hid_dim, supports, num_nodes,
                     input_dim if i == 0 else hid_dim,
                     max_diffusion_step)
            for i in range(num_layers)
        ])

    def forward(self, x_seq, device):
        """x_seq: [B, T, N, input_dim] -> final hidden states [num_layers, B, N, hid_dim]"""
        B, T, N, _ = x_seq.shape
        hidden = [cell.init_hidden(B, device) for cell in self.cells]

        for t in range(T):
            layer_input = x_seq[:, t]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(layer_input, hidden[i])
                layer_input = hidden[i]

        return hidden   # list of [B, N, hid_dim], one per layer


class DCRNNDecoder(nn.Module):
    """
    Autoregressive decoder with scheduled sampling.
    At each step, output = Linear(hidden_state_of_last_layer) -> [B, N, output_dim].
    """

    def __init__(self, output_dim, num_nodes, supports, hid_dim, num_layers, max_diffusion_step):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes

        self.cells = nn.ModuleList([
            DCGRUCell(hid_dim, supports, num_nodes,
                     output_dim if i == 0 else hid_dim,
                     max_diffusion_step)
            for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hid_dim, output_dim)

    def forward(self, hidden_states, y_true, pred_len, teacher_forcing_ratio):
        """
        hidden_states: list of [B, N, hid_dim], from encoder (one per layer)
        y_true: [B, pred_len, N, output_dim] or None (inference mode)
        Returns: [B, pred_len, N, output_dim]
        """
        B, N = hidden_states[0].shape[0], hidden_states[0].shape[1]
        device = hidden_states[0].device

        # GO symbol: zeros, matches original implementation
        decoder_input = torch.zeros(B, N, self.output_dim, device=device)
        hidden = list(hidden_states)
        outputs = []

        for t in range(pred_len):
            layer_input = decoder_input
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(layer_input, hidden[i])
                layer_input = hidden[i]

            out = self.output_layer(hidden[-1])   # [B, N, output_dim]
            outputs.append(out)

            # Scheduled sampling: use ground truth with prob teacher_forcing_ratio
            if y_true is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y_true[:, t]
            else:
                decoder_input = out

        return torch.stack(outputs, dim=1)   # [B, pred_len, N, output_dim]


# ── Full DCRNN model ──────────────────────────────────────────────────────────

class DCRNN(nn.Module):
    """
    DCRNN (ICLR 2018) faithful reproduction with scheduled sampling.

    Args:
        num_nodes   : N monitoring stations
        input_dim   : F input features (all features fed in, PM2.5 at index 0)
        output_dim  : forecasting target dim (1, PM2.5 only)
        hid_dim     : RNN hidden size (paper default: 64)
        num_layers  : stacked DCGRU layers (paper default: 2)
        max_diffusion_step: K, diffusion polynomial order (paper default: 2)
        pred_len    : output horizon
        adj_mx      : [N, N] np.ndarray, geographic adjacency (from preprocess.py)
        filter_type : "dual_random_walk" (paper default) or "random_walk"
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int = 1,
        hid_dim: int = 64,
        num_layers: int = 2,
        max_diffusion_step: int = 2,
        pred_len: int = 24,
        adj_mx: np.ndarray = None,
        filter_type: str = "dual_random_walk",
    ):
        super().__init__()
        assert adj_mx is not None, "DCRNN requires a predefined adjacency matrix (adj_mx.npy)"

        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.output_dim = output_dim

        supports = build_supports(adj_mx, filter_type)

        self.encoder = DCRNNEncoder(input_dim, num_nodes, supports, hid_dim, num_layers, max_diffusion_step)
        self.decoder = DCRNNDecoder(output_dim, num_nodes, supports, hid_dim, num_layers, max_diffusion_step)

    def forward(self, x, y_true=None, teacher_forcing_ratio=0.0):
        """
        x: [B, T, N, F]
        y_true: [B, pred_len, N, output_dim] (only used during training for
                 scheduled sampling; pass None at eval time)
        Returns: [B, pred_len, N, output_dim]
        """
        device = x.device
        hidden_states = self.encoder(x, device)
        return self.decoder(hidden_states, y_true, self.pred_len, teacher_forcing_ratio)