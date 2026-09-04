"""
GTS — faithful reproduction of Shang, Chen & Bi, ICLR 2021.
"Discrete Graph Structure Learning for Forecasting Multiple Time Series"
Adapted for preprocess.py output format.

Core idea: instead of using a predefined adjacency matrix (like DCRNN),
GTS LEARNS the graph structure jointly with the forecasting model.
  1. A feature extractor (1D-conv) encodes each node's full training-set time
     series into a fixed-size embedding.
  2. A link predictor (MLP) scores every node pair (i,j) using the
     concatenation of their embeddings, producing a [N,N] logit matrix theta.
  3. The Gumbel-Softmax reparameterization trick samples a DISCRETE (0/1)
     adjacency matrix from theta in a differentiable way (so gradients can
     flow back into the feature extractor and link predictor).
  4. The sampled graph is fed into a DCRNN-style encoder-decoder (same
     diffusion convolution GRU cells as DCRNN) for the actual forecasting.
  5. A KL-regularization term encourages the learned graph towards a sparse
     Erdos-Renyi prior, matching Eq. 6 of the paper.

Differences from official chaoshangcs/GTS repo (document in thesis):
  1. Input: features.npy [T,N,F] instead of METR-LA/PEMS-BAY sensor h5 files.
  2. Node embeddings for the feature extractor are computed over the
     Z-score normalized PM2.5 series (index 0) from the training split only,
     to avoid test-set leakage into the learned graph.
  3. No TensorFlow dependency check (original requirements.txt lists it but
     the PyTorch model doesn't use it).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dcrnn_faithful import DCGRUCell, calculate_random_walk_matrix


# ── Feature extractor: encodes each node's full history into an embedding ────

class FeatureExtractor(nn.Module):
    """
    1D-CNN over each node's PM2.5 training series, producing a fixed-size
    node embedding used by the link predictor. Matches the paper's Section
    3.1 "graph structure parameterization" step.
    """

    def __init__(self, seq_len, embed_dim=100):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=10, stride=1, padding=4)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=10, stride=1, padding=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, embed_dim)

    def forward(self, x):
        """x: [N, 1, seq_len] -> [N, embed_dim]"""
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)   # [N, 16]
        return self.fc(h)              # [N, embed_dim]


# ── Link predictor + Gumbel-Softmax discrete graph sampling ──────────────────

class GraphLearner(nn.Module):
    """
    Learns a discrete [N,N] adjacency matrix from node embeddings.

    theta_{ij} = MLP([e_i, e_j])   (link predictor logits)
    A ~ Gumbel-Softmax(theta)      (differentiable discrete sampling)
    """

    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.link_pred = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),   # binary logits: [no-edge, edge]
        )

    def forward(self, embeddings, temperature=0.5, hard=True):
        """
        embeddings: [N, embed_dim]
        Returns: adj [N, N] (soft or hard, differentiable), theta [N,N,2] logits
        """
        N = embeddings.shape[0]
        ei = embeddings.unsqueeze(1).expand(N, N, -1)   # [N, N, embed_dim]
        ej = embeddings.unsqueeze(0).expand(N, N, -1)   # [N, N, embed_dim]
        pair_feat = torch.cat([ei, ej], dim=-1)          # [N, N, 2*embed_dim]

        theta = self.link_pred(pair_feat)                # [N, N, 2]
        adj_soft = F.gumbel_softmax(theta, tau=temperature, hard=hard, dim=-1)
        adj = adj_soft[..., 1]                            # take "edge" channel -> [N, N]

        # zero out self-loops (paper masks diagonal before use)
        adj = adj * (1 - torch.eye(N, device=adj.device))
        return adj, theta


def graph_kl_loss(theta, prior_p=0.05):
    """
    KL divergence between the learned edge distribution and a sparse
    Bernoulli(prior_p) prior, matching Eq. 6 of the paper. Encourages sparsity.
    theta: [N,N,2] logits (pre-softmax)
    """
    probs = F.softmax(theta, dim=-1)          # [N,N,2]
    p_edge = probs[..., 1].clamp(min=1e-8, max=1 - 1e-8)
    prior = torch.full_like(p_edge, prior_p)
    kl = p_edge * torch.log(p_edge / prior) + (1 - p_edge) * torch.log((1 - p_edge) / (1 - prior))
    return kl.mean()


# ── GTS forecasting backbone (DCRNN-style, using the LEARNED graph) ──────────

class GTSEncoder(nn.Module):
    def __init__(self, input_dim, num_nodes, num_supports, hid_dim, num_layers, max_diffusion_step):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.num_supports = num_supports
        self.max_diffusion_step = max_diffusion_step

        self.cells = nn.ModuleList([
            _DynamicDCGRUCell(hid_dim, num_nodes,
                              input_dim if i == 0 else hid_dim,
                              max_diffusion_step, num_supports)
            for i in range(num_layers)
        ])

    def forward(self, x_seq, supports, device):
        B, T, N, _ = x_seq.shape
        hidden = [cell.init_hidden(B, device) for cell in self.cells]
        for t in range(T):
            layer_input = x_seq[:, t]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(layer_input, hidden[i], supports)
                layer_input = hidden[i]
        return hidden


class GTSDecoder(nn.Module):
    def __init__(self, output_dim, num_nodes, num_supports, hid_dim, num_layers, max_diffusion_step):
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_nodes = num_nodes

        self.cells = nn.ModuleList([
            _DynamicDCGRUCell(hid_dim, num_nodes,
                              output_dim if i == 0 else hid_dim,
                              max_diffusion_step, num_supports)
            for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hid_dim, output_dim)

    def forward(self, hidden_states, supports, y_true, pred_len, teacher_forcing_ratio):
        B, N = hidden_states[0].shape[0], hidden_states[0].shape[1]
        device = hidden_states[0].device
        decoder_input = torch.zeros(B, N, self.output_dim, device=device)
        hidden = list(hidden_states)
        outputs = []

        for t in range(pred_len):
            layer_input = decoder_input
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(layer_input, hidden[i], supports)
                layer_input = hidden[i]
            out = self.output_layer(hidden[-1])
            outputs.append(out)

            if y_true is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y_true[:, t]
            else:
                decoder_input = out

        return torch.stack(outputs, dim=1)


class _DynamicDCGRUCell(nn.Module):
    """
    Same diffusion-convolution GRU cell as DCRNN's DCGRUCell, but the graph
    support is passed in at forward() time instead of being fixed at
    construction -- because GTS's graph is learned fresh every forward pass
    (it depends on the current node embeddings), not precomputed once.
    """

    def __init__(self, num_units, num_nodes, input_dim, max_diffusion_step, num_supports):
        super().__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.max_diffusion_step = max_diffusion_step
        self.num_matrices = num_supports * max_diffusion_step + 1

        input_size = input_dim + num_units
        self.gconv_gate = nn.Linear(input_size * self.num_matrices, 2 * num_units)
        self.gconv_candidate = nn.Linear(input_size * self.num_matrices, num_units)

    def _diffusion_conv(self, x, linear_layer, supports):
        B, N, C = x.shape
        x0 = x.permute(1, 2, 0).reshape(N, -1)
        outputs = [x0]

        for support in supports:
            x1 = torch.mm(support, x0)
            outputs.append(x1)
            x_prev, x_cur = x0, x1
            for _ in range(2, self.max_diffusion_step + 1):
                x2 = 2 * torch.mm(support, x_cur) - x_prev
                outputs.append(x2)
                x_prev, x_cur = x_cur, x2

        out = torch.stack(outputs, dim=0)
        out = out.reshape(self.num_matrices, N, C, B)
        out = out.permute(3, 1, 2, 0).reshape(B, N, C * self.num_matrices)
        return linear_layer(out)

    def forward(self, x, state, supports):
        input_and_state = torch.cat([x, state], dim=-1)
        value = torch.sigmoid(self._diffusion_conv(input_and_state, self.gconv_gate, supports))
        r, u = torch.split(value, self.num_units, dim=-1)
        candidate_input = torch.cat([x, r * state], dim=-1)
        c = torch.tanh(self._diffusion_conv(candidate_input, self.gconv_candidate, supports))
        return u * state + (1 - u) * c

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.num_nodes, self.num_units, device=device)


# ── Full GTS model ────────────────────────────────────────────────────────────

class GTS(nn.Module):
    """
    GTS (ICLR 2021) faithful reproduction.

    Args:
        num_nodes    : N monitoring stations
        input_dim    : F input features (PM2.5 at index 0)
        output_dim   : 1 (PM2.5 forecast)
        hid_dim      : DCGRU hidden size (paper default: 64)
        num_layers   : stacked DCGRU layers (paper default: 2)
        max_diffusion_step: K (paper default: 2)
        pred_len     : output horizon
        train_seq_len: length of each node's history series used by the
                       feature extractor to learn the graph (paper uses the
                       full training-set series; here we use hist_len window)
        embed_dim    : node embedding dimension for the link predictor
        temperature  : Gumbel-Softmax temperature (paper sweeps 0.1-1.0)
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
        train_seq_len: int = 24,
        embed_dim: int = 100,
        temperature: float = 0.5,
        filter_type: str = "dual_random_walk",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.temperature = temperature
        self.filter_type = filter_type
        num_supports = 2 if filter_type == "dual_random_walk" else 1

        self.feature_extractor = FeatureExtractor(train_seq_len, embed_dim)
        self.graph_learner = GraphLearner(embed_dim)

        self.encoder = GTSEncoder(input_dim, num_nodes, num_supports, hid_dim, num_layers, max_diffusion_step)
        self.decoder = GTSDecoder(output_dim, num_nodes, num_supports, hid_dim, num_layers, max_diffusion_step)

    def _build_supports(self, adj):
        """Convert a learned [N,N] soft/hard adjacency into diffusion supports."""
        # 加微小 epsilon：Gumbel-Softmax 硬取樣（hard=True）在節點數較多時
        # (例如 127 個測站) 理論上有極小機率讓某一列剛好全部取樣成 0，
        # 造成分子分母同時逼近 0 的邊界情況，數值上不穩定。加一個均勻的
        # 微小偏移量可以避免這個退化情況，且對圖結構本身影響可忽略不計。
        adj = adj + 1e-8

        if self.filter_type == "dual_random_walk":
            d_fwd = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
            d_bwd = adj.sum(dim=0, keepdim=True).clamp(min=1e-6)
            support_fwd = adj / d_fwd
            support_bwd = adj.t() / d_bwd.t()
            return [support_fwd, support_bwd]
        else:
            d = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return [adj / d]

    def forward(self, x, node_series, y_true=None, teacher_forcing_ratio=0.0):
        """
        x: [B, T, N, F]  current batch window (fed to DCRNN encoder-decoder)
        node_series: [N, 1, train_seq_len]  PM2.5 history used ONLY to learn
                     the graph structure (typically a fixed sample from the
                     training split, same series reused across all batches
                     as in the original paper -- not the current batch's x).
        y_true: [B, pred_len, N, output_dim] or None
        Returns: preds [B, pred_len, N, output_dim], kl_loss (scalar)
        """
        device = x.device

        # 1. Learn the graph structure from node embeddings
        embeddings = self.feature_extractor(node_series)          # [N, embed_dim]
        adj, theta = self.graph_learner(embeddings, self.temperature, hard=self.training)
        supports = self._build_supports(adj)

        # 2. Forecast using DCRNN-style encoder-decoder with the learned graph
        hidden_states = self.encoder(x, supports, device)
        preds = self.decoder(hidden_states, supports, y_true, self.pred_len, teacher_forcing_ratio)

        kl = graph_kl_loss(theta)
        return preds, kl, adj