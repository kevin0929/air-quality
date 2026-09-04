"""
AGCRN — faithful reproduction of Bai et al., NeurIPS 2020.
Adapted for preprocess.py output format (features.npy [T,N,F]).

Original design (traffic forecasting on PEMSD4/PEMSD8):
  - Node Adaptive Parameter Learning (NAPL): each node gets its own GCN
    weight matrix, generated from a learnable node embedding E, instead of
    sharing one weight matrix across all nodes.
  - Data Adaptive Graph Generation (DAGG): the adjacency matrix is NOT
    predefined (no distance/wind matrix needed) -- it's learned end-to-end
    as softmax(relu(E @ E^T)), directly from the node embeddings.
  - AGCRNCell: replaces the standard GRU's linear layers with the above
    adaptive graph convolution (AGCN), so spatial and temporal modelling
    happen jointly at every recurrent step.
  - Encoder-decoder: encode hist_len steps -> single-shot Linear projection
    to pred_len steps (NOT autoregressive, per original architecture).

Differences from official LeiBAI/AGCRN repo (document in thesis):
  1. Input: features.npy [T,N,F] instead of PEMSD4/8's traffic csv format.
  2. No predefined adjacency at all (matches original -- this model doesn't
     use adj_mx/adj_wind from preprocess.py; the whole point is not needing one).
  3. Originally single-feature (traffic flow); here F=10 (PM2.5 + met).
  4. Default embed_dim=10, rnn_units=64, cheb_k=2 (as in original PEMSD4 config).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── AGCN: Adaptive Graph Convolution (NAPL + DAGG combined) ───────────────────

class AVWGCN(nn.Module):
    """
    Adaptive Vertex-Weighted Graph Convolution.

    Combines:
      DAGG: adaptive adjacency  A = softmax(ReLU(E @ E^T))   [N, N]
      NAPL: node-specific weights, generated from E via low-rank factors,
            instead of one shared weight matrix for all nodes.

    Uses Chebyshev polynomial approximation (order cheb_k) of the graph
    convolution, matching the original AGCN implementation.
    """

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = cheb_k
        # Weights pool: [embed_dim, cheb_k, dim_in, dim_out]
        # Node-specific weight = node_embeddings @ weights_pool
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool    = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        nn.init.xavier_normal_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, x, node_embeddings):
        """
        x: [B, N, dim_in]
        node_embeddings: [N, embed_dim]  (learnable, shared across all layers/steps)
        Returns: [B, N, dim_out]
        """
        N = node_embeddings.shape[0]

        # DAGG: data-adaptive adjacency matrix
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)

        # Chebyshev polynomial basis: [I, A, 2*A@T_{k-1} - T_{k-2}, ...]
        support_set = [torch.eye(N, device=x.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports_stacked = torch.stack(support_set, dim=0)   # [cheb_k, N, N]

        # NAPL: node-specific weights/bias from node embeddings
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # [N, cheb_k, dim_in, dim_out]
        bias    = torch.matmul(node_embeddings, self.bias_pool)                       # [N, dim_out]

        x_g = torch.einsum("knm,bmc->bknc", supports_stacked, x)      # [B, cheb_k, N, dim_in]
        x_g = x_g.permute(0, 2, 1, 3)                                  # [B, N, cheb_k, dim_in]
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # [B, N, dim_out]
        return x_gconv


# ── AGCRNCell: GRU cell with AGCN replacing the linear layers ─────────────────

class AGCRNCell(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = dim_out
        self.gate   = AVWGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + dim_out, dim_out,     cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        """
        x: [B, N, dim_in]  state: [B, N, dim_out]
        Returns new state [B, N, dim_out]
        """
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, r * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = z * state + (1 - z) * hc
        return h

    def init_hidden_state(self, batch_size, device):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)


# ── AVWDCRNN: stacked AGCRNCell encoder ───────────────────────────────────────

class AVWDCRNN(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super().__init__()
        assert num_layers >= 1
        self.num_nodes  = num_nodes
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(num_nodes, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(num_nodes, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        """
        x: [B, T, N, dim_in]
        init_state: [num_layers, B, N, dim_out]
        Returns output_hidden (last layer's full sequence) [B, T, N, dim_out]
                and final states of each layer [num_layers, B, N, dim_out]
        """
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size, device):
        init_states = [cell.init_hidden_state(batch_size, device) for cell in self.dcrnn_cells]
        return torch.stack(init_states, dim=0)   # [num_layers, B, N, dim_out]


# ── AGCRN main model ──────────────────────────────────────────────────────────

class AGCRN(nn.Module):
    """
    AGCRN (NeurIPS 2020) faithful reproduction.

    Args:
        num_nodes  : N monitoring stations
        in_features: F input features (all fed to encoder, PM2.5 at index 0)
        out_dim    : output feature count (1, PM2.5 only)
        hidden_dim : RNN hidden size (paper default: 64)
        embed_dim  : node embedding dimension (paper default: 10)
        cheb_k     : Chebyshev polynomial order (paper default: 2)
        num_layers : stacked AGCRNCell layers (paper default: 2)
        pred_len   : output horizon
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        out_dim: int = 1,
        hidden_dim: int = 64,
        embed_dim: int = 10,
        cheb_k: int = 2,
        num_layers: int = 2,
        pred_len: int = 24,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_dim    = out_dim
        self.hidden_dim = hidden_dim
        self.pred_len   = pred_len

        # Learnable node embeddings -- this IS the graph (DAGG + NAPL source)
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embed_dim), requires_grad=True
        )

        self.encoder = AVWDCRNN(
            num_nodes, in_features, hidden_dim, cheb_k, embed_dim, num_layers
        )

        # Single-shot projection: hidden_dim -> pred_len * out_dim
        # (matches original: no autoregressive decoder, predicts all steps at once)
        self.end_conv = nn.Conv2d(
            in_channels=1, out_channels=pred_len * out_dim,
            kernel_size=(1, hidden_dim), bias=True,
        )

    def forward(self, x, *_unused):
        """
        x: [B, T, N, F]  (extra args accepted & ignored for interface
                          compatibility with adj_mx/adj_wind used by other models)
        Returns: [B, pred_len, N, out_dim]
        """
        B = x.shape[0]
        init_state = self.encoder.init_hidden(B, x.device)
        output, _ = self.encoder(x, init_state, self.node_embeddings)   # [B, T, N, hidden_dim]

        # Take last timestep's hidden state, project to all future steps at once
        output = output[:, -1:, :, :]                                    # [B, 1, N, hidden_dim]
        output = self.end_conv(output)                                   # [B, pred_len*out_dim, N, 1]
        output = output.squeeze(-1).reshape(B, self.pred_len, self.out_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)                              # [B, pred_len, N, out_dim]
        return output