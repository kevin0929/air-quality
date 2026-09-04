"""
PM2.5-GNN — faithful reproduction of Song et al., SIGSPATIAL 2020.
Adapted for preprocess.py output (features.npy / adj_wind.npy / scalers.pkl).

Differences from original KnowAir version (document in thesis):
  1. Teacher-forcing: future meteorology uses ground-truth observations
     (feature[:, hist_len+i]) instead of NWP forecasts (closed-loop setting).
  2. Edge format: converted from dense adj_wind [N,N] → sparse edge_index/edge_attr.
  3. torch_scatter replaced with pure PyTorch for broader compatibility.
  4. wind_mean/wind_std from scalers.pkl instead of KnowAir dataset stats.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter, Sequential, Sigmoid


# ── scatter helpers (replaces torch_scatter) ─────────────────────────────────

def scatter_add_pt(src, index, dim, dim_size):
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    idx = index.view(*([1] * dim), -1, *([1] * (src.dim() - dim - 1)))
    idx = idx.expand_as(src)
    out.scatter_add_(dim, idx, src)
    return out

def scatter_sub_pt(src, index, dim, dim_size):
    return scatter_add_pt(-src, index, dim, dim_size)


# ── edge construction from preprocess.py output ───────────────────────────────

def build_edge_index_attr(adj_wind: np.ndarray,
                           dist_mx: np.ndarray,
                           station_lons: np.ndarray,
                           station_lats: np.ndarray):
    """
    Convert dense adj_wind [N,N] → edge_index [2,M], edge_attr [M,2].

    edge_attr[:,0] = city_dist  (km)
    edge_attr[:,1] = city_direc (radians, clockwise from north, i→j direction)
    """
    N = adj_wind.shape[0]
    src_list, tgt_list, dist_list, direc_list = [], [], [], []

    for i in range(N):
        for j in range(N):
            if i != j and adj_wind[i, j] > 0:
                src_list.append(i)
                tgt_list.append(j)
                dist_list.append(float(dist_mx[i, j]))
                # Geographic bearing from i to j (CW from North, radians)
                dlon = station_lons[j] - station_lons[i]
                dlat = station_lats[j] - station_lats[i]
                bearing = np.arctan2(dlon, dlat)   # CW from N
                direc_list.append(float(bearing))

    edge_index = np.array([src_list, tgt_list], dtype=np.int64)
    edge_attr  = np.stack([dist_list, direc_list], axis=1).astype(np.float32)
    return edge_index, edge_attr


# ── GRU Cell (identical to original cells.py) ────────────────────────────────

class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.x2h = nn.Linear(input_size,  3 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x, h):
        gx = self.x2h(x)
        gh = self.h2h(h)
        xr, xz, xn = gx.chunk(3, dim=-1)
        hr, hz, hn_ = gh.chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn_)
        return (1 - z) * h + z * n


# ── GraphGNN (faithful to original) ──────────────────────────────────────────

class GraphGNN(nn.Module):
    """
    edge_weight = ReLU(3 * wind_speed * cos(theta) / city_dist)
    theta = |city_direc - wind_direc|

    scatter_add (import) + scatter_sub (export) directed aggregation.
    """

    def __init__(self, device, edge_index, edge_attr,
                 in_dim, out_dim, wind_mean, wind_std):
        super().__init__()
        self.device = device

        self.register_buffer("edge_index",
                             torch.LongTensor(edge_index))
        self.register_buffer("edge_attr",
                             torch.FloatTensor(edge_attr))

        ea = torch.FloatTensor(edge_attr)
        ea_std = ea.std(dim=0).clamp(min=1e-6)
        self.register_buffer("edge_attr_norm", (ea - ea.mean(dim=0)) / ea_std)

        self.register_buffer("wind_mean",
                             torch.FloatTensor(wind_mean.astype(np.float32)))
        self.register_buffer("wind_std",
                             torch.FloatTensor(wind_std.astype(np.float32)))

        self.w = Parameter(torch.rand([1]))
        self.b = Parameter(torch.rand([1]))

        e_h, e_out, n_out = 32, 30, out_dim

        self.edge_mlp = Sequential(
            Linear(in_dim * 2 + 2 + 1, e_h), Sigmoid(),
            Linear(e_h, e_out),              Sigmoid(),
        )
        self.node_mlp = Sequential(
            Linear(e_out, n_out), Sigmoid(),
        )

    def forward(self, x):
        """x: [B, N, in_dim]  →  [B, N, out_dim]"""
        B, N, _ = x.shape
        M = self.edge_index.shape[1]

        edge_src, edge_target = self.edge_index     # [M]
        node_src    = x[:, edge_src]                # [B, M, in_dim]
        node_target = x[:, edge_target]             # [B, M, in_dim]

        # Denormalise wind (last 2 features: windspeed, winddirec)
        src_wind       = node_src[:, :, -2:] * self.wind_std + self.wind_mean
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:, :, 1]

        edge_attr_b = self.edge_attr.unsqueeze(0).expand(B, M, 2)
        city_dist   = edge_attr_b[:, :, 0].clamp(min=1e-6)
        city_direc  = edge_attr_b[:, :, 1]

        theta       = torch.abs(city_direc - src_wind_direc)
        edge_weight = F.relu(3.0 * src_wind_speed * torch.cos(theta) / city_dist)

        edge_attr_norm_b = self.edge_attr_norm.unsqueeze(0).expand(B, M, 2)
        out = torch.cat([
            node_src, node_target,
            edge_attr_norm_b,
            edge_weight.unsqueeze(-1),
        ], dim=-1)

        out     = self.edge_mlp(out)
        out_add = scatter_add_pt(out, edge_target, dim=1, dim_size=N)
        out_sub = scatter_sub_pt(out, edge_src,    dim=1, dim_size=N)
        out     = out_add + out_sub

        return self.node_mlp(out)


# ── PM25GNN (faithful to original, adapted interface) ─────────────────────────

class PM25GNN(nn.Module):
    """
    PM2.5-GNN (SIGSPATIAL 2020).

    Teacher-forcing (Option B):
        feature[:, hist_len+i] = ground-truth observations at future step i,
        not NWP forecast. State this clearly in your thesis.

    Args:
        hist_len     : input sequence length
        pred_len     : prediction horizon
        in_dim       : F (node features, PM2.5 NOT included here)
        city_num     : N stations
        batch_size   : B
        device       : torch.device
        edge_index   : [2, M]  (from build_edge_index_attr)
        edge_attr    : [M, 2]  (city_dist, city_direc)
        wind_mean    : [2]  physical mean of (windspeed, winddirec)
        wind_std     : [2]  physical std  of (windspeed, winddirec)
    """

    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size,
                 device, edge_index, edge_attr, wind_mean, wind_std):
        super().__init__()
        self.device     = device
        self.hist_len   = hist_len
        self.pred_len   = pred_len
        self.city_num   = city_num
        self.batch_size = batch_size
        self.in_dim     = in_dim
        self.hid_dim    = 64
        self.out_dim    = 1
        self.gnn_out    = 13

        self.fc_in = nn.Linear(in_dim, self.hid_dim)

        # x passed to GraphGNN = [pm25(1), feature(F)] → actual in_dim = 1 + F
        gnn_in_dim = 1 + in_dim

        self.graph_gnn = GraphGNN(
            device, edge_index, edge_attr,
            gnn_in_dim, self.gnn_out,
            wind_mean, wind_std,
        )

        # input to GRUCell = gnn_out + pm25(1) + feature(F)
        self.gru_cell = GRUCell(self.gnn_out + gnn_in_dim, self.hid_dim)
        self.fc_out   = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, feature):
        """
        pm25_hist : [B, hist_len, N, 1]
        feature   : [B, hist_len+pred_len, N, F]
        Returns   : [B, pred_len, N, 1]
        """
        B = pm25_hist.size(0)
        pm25_pred = []

        hn = torch.zeros(B * self.city_num, self.hid_dim, device=self.device)
        xn = pm25_hist[:, -1]   # [B, N, 1]

        for i in range(self.pred_len):
            # Concat PM2.5 + meteo at step hist_len+i (teacher-forcing)
            x = torch.cat([xn, feature[:, self.hist_len + i]], dim=-1)  # [B, N, 1+F]

            xn_gnn = self.graph_gnn(x.contiguous())   # [B, N, gnn_out]

            x = torch.cat([xn_gnn, x], dim=-1)        # [B, N, gnn_out+1+F]

            hn  = self.gru_cell(x.reshape(B * self.city_num, -1), hn)
            xn  = self.fc_out(hn.view(B, self.city_num, self.hid_dim))  # [B, N, 1]
            pm25_pred.append(xn)

        return torch.stack(pm25_pred, dim=1)   # [B, pred_len, N, 1]