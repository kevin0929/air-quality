"""
AirPhyNet — faithful reproduction of Hettige et al., ICLR 2024.
Adapted for preprocess.py output format.

FIX (v2): ODEFunc is now built ONCE in AirPhyNetModel.__init__ and kept as
a persistent submodule, so its parameters (log_kappa, log_alpha, diff_gcn,
adv_gcn, wind_proj, out_proj) are tracked by the optimizer and actually learn.

Previously ODEFunc was re-instantiated inside forward() every batch, which
meant its weights were random noise on every call and never trained --
this was the root cause of Val-MAE stalling around 6.2.

Dynamic per-batch data (current wind vectors, current graphs) are now
pushed into the persistent ODEFunc via `set_context()` before each
odeint() call, instead of rebuilding the whole module.

Install dependency:
    pip install torchdiffeq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import GRU
from torchdiffeq import odeint


# ── utils ─────────────────────────────────────────────────────────────────────

def split_last_dim(data):
    d = data.size(-1) // 2
    return data[..., :d], data[..., d:]


def sample_standard_gaussian(mu, sigma):
    eps = torch.empty_like(mu).normal_()
    return mu + eps * sigma


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


# ── graph helpers ─────────────────────────────────────────────────────────────

def build_transition_matrices(adj, filter_type='dual_random_walk'):
    """Returns list of [N, N] transition matrices (not parameters, just tensors)."""
    N = adj.shape[0]
    adj = adj + torch.eye(N, device=adj.device) * 1e-4

    d_out = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
    d_in  = adj.sum(dim=0, keepdim=True).clamp(min=1e-6)

    rw   = adj / d_out
    rw_T = adj.T / d_in.T

    if filter_type == 'random_walk':
        return [rw]
    elif filter_type == 'dual_random_walk':
        return [rw, rw_T]
    elif filter_type == 'laplacian':
        d_inv_sqrt = d_out.pow(-0.5)
        sym = d_inv_sqrt * adj * d_inv_sqrt.T
        return [sym]
    raise ValueError(f"Unknown filter_type: {filter_type}")


def graph_conv(x, supports):
    """x: [B, N, F], supports: list of [N, N] -> [B, N, len(supports)*F]"""
    return torch.cat(
        [torch.einsum('nm,bmf->bnf', S, x) for S in supports], dim=-1
    )


# ── ODEFunc (now a persistent, trainable submodule) ───────────────────────────

class ODEFunc(nn.Module):
    """
    RHS of dz/dt = f(z, t).

    Diffusion:  kappa * GCN_diff(z)                  (distance-based graph)
    Advection:  alpha * GCN_adv(z) * wind_gate(wind)  (wind-based graph)

    IMPORTANT: this module is created ONCE (in AirPhyNetModel.__init__) and
    reused across all batches/epochs. Per-batch data (current wind vectors,
    current adjacency) are injected via set_context() as buffers, not
    re-initialised parameters -- this is what lets kappa/alpha/GCN weights
    actually train instead of being random every forward pass.
    """

    def __init__(self, latent_dim, ode_gen_dim, num_nodes,
                 n_diff_supports=2):
        super().__init__()
        self.latent_dim  = latent_dim
        self.ode_gen_dim = ode_gen_dim
        self.num_nodes   = num_nodes
        self.nfe         = 0

        # Diffusion GCN: latent_dim * n_diff_supports -> ode_gen_dim
        self.diff_gcn = nn.Sequential(
            nn.Linear(latent_dim * n_diff_supports, ode_gen_dim),
            nn.Tanh(),
        )
        # Advection GCN: single wind-adjacency support -> ode_gen_dim
        self.adv_gcn = nn.Sequential(
            nn.Linear(latent_dim * 1, ode_gen_dim),
            nn.Tanh(),
        )
        # Wind modulation gate: (wind_u, wind_v) -> ode_gen_dim
        self.wind_proj = nn.Sequential(
            nn.Linear(2, ode_gen_dim),
            nn.Sigmoid(),
        )
        # Combine diffusion + advection -> back to latent_dim
        self.out_proj = nn.Linear(2 * ode_gen_dim, latent_dim)

        # Learnable coefficients
        self.log_kappa = nn.Parameter(torch.tensor(-3.0))
        self.log_alpha = nn.Parameter(torch.tensor(-3.0))

        # Context placeholders (set per-batch via set_context, not parameters)
        # persistent=False: these are per-batch scratch buffers overwritten by
        # set_context() before every forward call. They must NOT be saved to
        # the checkpoint's state_dict, otherwise loading a checkpoint saved
        # with a different batch_size (e.g. last partial batch) will fail
        # with a shape mismatch.
        self.register_buffer('diff_supports_stack', torch.zeros(2, num_nodes, num_nodes), persistent=False)
        self.register_buffer('adv_support',        torch.zeros(num_nodes, num_nodes), persistent=False)
        self.register_buffer('wind_vars',          torch.zeros(1, num_nodes, 2), persistent=False)
        self._n_diff_supports = n_diff_supports

    def set_context(self, diff_supports, adv_support, wind_vars):
        """
        Called once per forward() before odeint(), NOT re-initialising any
        trainable weights -- only updates the graph/wind data for this batch.

        diff_supports : list of [N, N] tensors (from build_transition_matrices)
        adv_support   : [N, N] wind-aware adjacency
        wind_vars     : [B, N, 2]  current wind_u, wind_v
        """
        self.diff_supports_stack = torch.stack(diff_supports, dim=0)   # [n_supports, N, N]
        self.adv_support         = adv_support                          # [N, N]
        self.wind_vars           = wind_vars                            # [B, N, 2]

    def forward(self, t, z):
        """
        z: [B_flat, N * latent_dim]  (B_flat = n_traj * B, flattened for odeint)
        Returns dz/dt of same shape.
        """
        self.nfe += 1
        B_flat = z.shape[0]
        N = self.num_nodes
        D = self.latent_dim

        z_node = z.reshape(B_flat, N, D)

        kappa = self.log_kappa.exp().clamp(max=0.3)
        alpha = self.log_alpha.exp().clamp(max=0.3)

        # ── Diffusion ──────────────────────────────────────────────
        diff_supports = [self.diff_supports_stack[i] for i in range(self._n_diff_supports)]
        diff_in  = graph_conv(z_node, diff_supports)          # [B_flat, N, D*n_supports]
        diff_out = kappa * self.diff_gcn(diff_in)             # [B_flat, N, ode_gen_dim]

        # ── Advection ──────────────────────────────────────────────
        adv_in   = graph_conv(z_node, [self.adv_support])     # [B_flat, N, D]
        adv_feat = self.adv_gcn(adv_in)                       # [B_flat, N, ode_gen_dim]

        # wind_vars is [B, N, 2]; broadcast to B_flat (= n_traj * B)
        B = self.wind_vars.shape[0]
        n_traj = B_flat // B
        wind_expanded = self.wind_vars.repeat(n_traj, 1, 1)   # [B_flat, N, 2]
        wind_gate = self.wind_proj(wind_expanded)             # [B_flat, N, ode_gen_dim]

        adv_out = alpha * adv_feat * wind_gate                # [B_flat, N, ode_gen_dim]

        # ── Combine ────────────────────────────────────────────────
        combined = torch.cat([diff_out, adv_out], dim=-1)     # [B_flat, N, 2*ode_gen_dim]
        dzdt = self.out_proj(combined)                        # [B_flat, N, D]

        return dzdt.reshape(B_flat, N * D)


# ── DiffeqSolver ─────────────────────────────────────────────────────────────

class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, latent_dim,
                 odeint_rtol=1e-3, odeint_atol=1e-4):
        super().__init__()
        self.ode_func    = ode_func
        self.method      = method
        self.latent_dim  = latent_dim
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict):
        """
        first_point : [n_traj, B, N * latent_dim]
        Returns sol_ys : [horizon, n_traj, B, N * latent_dim]
        """
        n_traj = first_point.shape[0]
        B      = first_point.shape[1]
        fp_flat = first_point.reshape(n_traj * B, -1)

        self.ode_func.nfe = 0

        # Fixed-step solvers (rk4, euler, midpoint) need an explicit step_size,
        # otherwise torchdiffeq errors out. Adaptive solvers (dopri5) ignore it.
        solve_kwargs = dict(method=self.method, rtol=self.odeint_rtol, atol=self.odeint_atol)
        if self.method in ("rk4", "euler", "midpoint"):
            time_span = (time_steps_to_predict[-1] - time_steps_to_predict[0]).item()
            n_steps   = max(len(time_steps_to_predict) - 1, 1)
            solve_kwargs["options"] = {"step_size": time_span / n_steps}

        sol = odeint(self.ode_func, fp_flat, time_steps_to_predict, **solve_kwargs)
        # [horizon, n_traj*B, N*D]

        horizon = sol.shape[0]
        return sol.reshape(horizon, n_traj, B, -1), self.ode_func.nfe


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder_z0_RNN(nn.Module):
    """GRU encoder: PM2.5 history -> (mu, std) of latent z0, plus current wind."""

    def __init__(self, num_nodes, rnn_units, latent_dim, input_dim=1):
        super().__init__()
        self.num_nodes  = num_nodes
        self.rnn_units  = rnn_units
        self.latent_dim = latent_dim

        self.gru_rnn = GRU(input_dim, rnn_units)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(rnn_units, 50), nn.Tanh(),
            nn.Linear(50, latent_dim * 2),
        )
        init_network_weights(self.hiddens_to_z0)

    def forward(self, x):
        """
        x: [B, T, N, F]
        Returns mean, std: [1, B, N*latent_dim]; wind_uv: [B, N, 2]
        """
        B, T, N, Fdim = x.shape

        pm25    = x[:, :, :, 0:1]      # [B, T, N, 1]
        wind_uv = x[:, -1, :, 8:10]    # [B, N, 2]

        pm25_flat  = pm25.permute(1, 0, 2, 3).reshape(T, B * N, 1)
        outputs, _ = self.gru_rnn(pm25_flat)
        last_out   = outputs[-1].reshape(B, N, self.rnn_units)

        z0_params = self.hiddens_to_z0(last_out)
        mean, std = split_last_dim(z0_params)

        mean = mean.reshape(B, N * self.latent_dim).unsqueeze(0)
        std  = std.reshape(B, N * self.latent_dim).unsqueeze(0).abs()

        return mean, std, wind_uv


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, output_dim, num_nodes, latent_dim):
        super().__init__()
        self.num_nodes  = num_nodes
        self.output_dim = output_dim
        self.latent_dim = latent_dim

    def forward(self, sol_ys):
        """sol_ys: [horizon, n_traj, B, N*latent_dim] -> [horizon, B, N*output_dim]"""
        horizon, n_traj, B, _ = sol_ys.shape
        N, D = self.num_nodes, self.latent_dim
        sol = sol_ys.reshape(horizon, n_traj, B, N, D)
        sol = sol.mean(dim=-1, keepdim=True)   # [horizon, n_traj, B, N, 1]
        sol = sol.mean(dim=1)                  # [horizon, B, N, 1]
        return sol.reshape(horizon, B, N * self.output_dim)


# ── AirPhyNetModel ────────────────────────────────────────────────────────────

class AirPhyNetModel(nn.Module):
    """
    AirPhyNet (ICLR 2024) faithful reproduction, v2.

    Key fix: ODEFunc + DiffeqSolver are built ONCE here (persistent
    submodules), so their weights are tracked by the optimizer and learn
    across all batches -- not rebuilt from scratch on every forward call.
    """

    def __init__(
        self,
        num_nodes:    int,
        in_features:  int,
        hidden_dim:   int   = 64,
        pred_len:     int   = 24,
        latent_dim:   int   = 4,
        n_traj:       int   = 1,
        ode_method:   str   = 'dopri5',
        ode_rtol:     float = 1e-3,
        ode_atol:     float = 1e-4,
        gen_dim:      int   = 64,
        filter_type:  str   = 'dual_random_walk',
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.num_nodes   = num_nodes
        self.pred_len    = pred_len
        self.latent_dim  = latent_dim
        self.n_traj      = n_traj
        self.filter_type = filter_type

        n_diff_supports = 2 if filter_type == 'dual_random_walk' else 1

        self.encoder_z0 = Encoder_z0_RNN(
            num_nodes=num_nodes, rnn_units=hidden_dim,
            latent_dim=latent_dim, input_dim=1,
        )
        self.decoder = Decoder(
            output_dim=1, num_nodes=num_nodes, latent_dim=latent_dim,
        )

        # ── PERSISTENT ODEFunc (built once, trains properly) ──────
        self.ode_func = ODEFunc(
            latent_dim=latent_dim, ode_gen_dim=gen_dim,
            num_nodes=num_nodes, n_diff_supports=n_diff_supports,
        )
        self.diffeq_solver = DiffeqSolver(
            self.ode_func, ode_method, latent_dim,
            odeint_rtol=ode_rtol, odeint_atol=ode_atol,
        )

        t = torch.arange(0, pred_len, step=1).float() / pred_len
        self.register_buffer('time_steps', t)

    def forward(self, x, adj_mx, adj_wind, return_kl=False):
        """
        x        : [B, T, N, F]
        adj_mx   : [N, N]  symmetric adjacency (diffusion)
        adj_wind : [N, N]  directed adjacency  (advection)
        return_kl: if True, also return KL(q(z0|x) || N(0,1)) for regularisation
        Returns  : [B, pred_len, N, 1]   (and kl_loss scalar if return_kl=True)
        """
        # 1. Encode -> z0 distribution + current wind
        first_point_mu, first_point_std, last_wind = self.encoder_z0(x)

        # 2. Sample n_traj trajectories (TRAIN) or use mean directly (EVAL)
        #
        # Bug fix: previously this always sampled stochastically, even during
        # evaluate(). With n_traj=1 that means every validation pass used a
        # fresh random noise draw, so Val-MAE bounced around from pure sampling
        # variance instead of reflecting real model improvement (e.g. observed
        # 6.253 -> 6.176 -> 6.207 -> 6.177, non-monotonic noise, not a trend).
        # Standard VAE practice: sample during training (regularised by KL),
        # but decode from the mean at eval time for a deterministic, repeatable
        # prediction.
        if self.training:
            means_z0 = first_point_mu.repeat(self.n_traj, 1, 1)
            sigma_z0 = first_point_std.repeat(self.n_traj, 1, 1)
            first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
        else:
            first_point_enc = first_point_mu   # [1, B, N*latent_dim], deterministic

        # 3. Push this batch's graph + wind into the PERSISTENT ODEFunc
        diff_supports = build_transition_matrices(adj_mx, self.filter_type)
        self.ode_func.set_context(
            diff_supports=diff_supports,
            adv_support=adj_wind,
            wind_vars=last_wind,
        )

        # 4. Solve ODE (ode_func weights are trained, not random)
        sol_ys, _ = self.diffeq_solver(first_point_enc, self.time_steps)

        # 5. Decode -> PM2.5 predictions
        outputs = self.decoder(sol_ys)       # [pred_len, B, N]
        outputs = outputs.permute(1, 0, 2)   # [B, pred_len, N]
        pm25_pred = outputs.unsqueeze(-1)    # [B, pred_len, N, 1]

        if return_kl:
            # KL( N(mu, std) || N(0, 1) ), standard VAE regulariser.
            # Without this, std can grow unbounded and the sampled first_point
            # becomes pure noise, drowning out the encoder's learned signal --
            # this was the root cause of the loss oscillating instead of
            # decreasing during training.
            kl = -0.5 * torch.mean(
                1 + torch.log(first_point_std.pow(2) + 1e-6)
                  - first_point_mu.pow(2)
                  - first_point_std.pow(2)
            )
            return pm25_pred, kl

        return pm25_pred