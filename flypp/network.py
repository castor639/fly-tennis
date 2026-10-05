"""Connectome-constrained rate network over the MaleCNS graph, in PyTorch.

Implements the flyvis formulation (Lappalainen et al., Nature 2024), verified against
`flyvis/network/dynamics.py::PPNeuronIGRSynapses` and
`flyvis/network/initialization.py`:

    tau_i dV_i/dt = -V_i + b_i + sum_j w_ij ReLU(V_j) + x_i(t)
    w_ij          = sign_ij * N_syn,ij * alpha_g(i,j)

`sign` and `N_syn` are frozen from the connectome, so Dale's law holds by construction
and training can never flip an excitatory synapse to inhibitory. Only `alpha` (per
parameter group), `tau` and `b` (per cell type) carry gradients. `alpha` is kept
non-negative by construction so the sign cannot be smuggled around.

Everything is a plain nn.Module, so a task loss - including an RL objective - can be
backpropagated straight through the dynamics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from flypp.connectome import Connectome


class ConnectomeNetwork(nn.Module):
    def __init__(
        self,
        c: Connectome,
        *,
        alpha_scale: float = 8.0,
        tau_init_ms: float = 20.0,
        bias_init: float = 0.5,
        normalize_input: bool = True,
        rate_max: float | None = 5.0,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_nodes = c.n_nodes
        self.n_edges = c.n_edges
        self.device = torch.device(device)

        # --- frozen connectome structure -------------------------------------
        self.register_buffer("src", torch.as_tensor(c.src, dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(c.dst, dtype=torch.long))
        self.register_buffer("n_syn", torch.as_tensor(c.n_syn, dtype=dtype))
        self.register_buffer("edge_sign", torch.as_tensor(c.edge_sign, dtype=dtype))
        self.register_buffer("group_id", torch.as_tensor(c.group_id, dtype=torch.long))

        # Per-neuron input normalisation. In-degree in the whole CNS spans three
        # orders of magnitude (median ~200 input synapses, max ~119,000), so a single
        # scalar gain cannot work: whatever keeps the top 1% from diverging starves
        # everyone else, and signal died ~100x per synapse at the stable scale. Dividing
        # each neuron's incoming weights by its total input synapse count makes the
        # drive a weighted average of presynaptic rates - the synaptic-scaling
        # homeostasis real neurons perform. Alpha then lives at O(1) instead of O(1e-4).
        # Saturating rate. A pure ReLU rate network with 65% excitatory synapses has a
        # global positive-feedback loop and a hard instability edge (alpha 0.5 settles,
        # 1.0 diverges to 1e12). Real neurons have a maximum firing rate; bounding the
        # rate makes the dynamics unconditionally bounded, so gain can be set high
        # enough to actually transmit localised visual signals instead of being pinned
        # just below the edge of chaos. rate_max=None recovers plain ReLU.
        self.rate_max = rate_max
        self.normalize_input = normalize_input
        if normalize_input:
            total_in = np.bincount(c.dst, weights=c.n_syn, minlength=c.n_nodes)
            inv = 1.0 / np.maximum(total_in, 1.0)
            self.register_buffer("in_norm", torch.as_tensor(inv[c.dst], dtype=dtype))
        else:
            self.register_buffer("in_norm", torch.ones(c.n_edges, dtype=dtype))

        # --- cell-type index for per-type node parameters ---------------------
        types, type_idx = np.unique(c.cell_type, return_inverse=True)
        self.cell_types = types
        self.register_buffer("type_idx", torch.as_tensor(type_idx, dtype=torch.long))

        # --- trainable parameters --------------------------------------------
        # alpha initialised as flyvis does: scale / <N_syn> for the group
        n_groups = c.n_groups
        if normalize_input:
            # weights are already scale-free after normalisation; alpha is a plain gain
            alpha0 = np.full(n_groups, alpha_scale)
        else:
            # flyvis init: scale / <N_syn> for the group
            syn_sum = np.bincount(c.group_id, weights=c.n_syn, minlength=n_groups)
            syn_cnt = np.bincount(c.group_id, minlength=n_groups).clip(min=1)
            alpha0 = alpha_scale / np.maximum(syn_sum / syn_cnt, 1.0)
        self.log_alpha = nn.Parameter(torch.log(torch.as_tensor(alpha0, dtype=dtype)))
        self.log_tau = nn.Parameter(
            torch.full((len(types),), float(np.log(tau_init_ms * 1e-3)), dtype=dtype)
        )
        # Resting potential must be POSITIVE, not zero. With bias=0 and a ReLU
        # nonlinearity every neuron sits at the rectification boundary, so
        # inhibitory input has nothing to subtract from and the network is silent.
        # That matters here because photoreceptors are histaminergic (inhibitory):
        # with a zero baseline, driving the eye propagates literally nothing. Shiu
        # et al. flag the same failure mode - "inhibitory connections to an inactive
        # neuron have no effect".
        self.bias = nn.Parameter(torch.full((len(types),), float(bias_init),
                                            dtype=dtype))
        self.to(self.device)

    # -- derived quantities ---------------------------------------------------

    def edge_weights(self) -> torch.Tensor:
        """w_ij = sign * N_syn * alpha_group. alpha > 0 always, so sign is preserved."""
        alpha = torch.exp(self.log_alpha)[self.group_id]
        return self.edge_sign * self.n_syn * self.in_norm * alpha

    def node_tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)[self.type_idx]

    def node_bias(self) -> torch.Tensor:
        return self.bias[self.type_idx]

    # -- dynamics -------------------------------------------------------------

    def initial_state(self, batch: int = 1) -> torch.Tensor:
        return self.node_bias().expand(batch, self.n_nodes).clone()

    def _weight_matrix(self) -> torch.Tensor:
        """Sparse CSR (n_nodes x n_nodes) weight matrix, cached while the parameters are unchanged.

        Used for inference only: a sparse matmul is ~10x faster than the per-edge gather +
        index_add_ on CPU. The gather path stays for training (it is differentiable
        through alpha)."""
        key = (self.log_alpha._version, self.log_alpha.data_ptr())
        if getattr(self, "_W_key", None) != key:
            w = self.edge_weights().detach()
            W = torch.sparse_coo_tensor(torch.stack([self.dst, self.src]), w, (self.n_nodes, self.n_nodes)).coalesce()
            self._W = W.to_sparse_csr(); self._W_key = key
        return self._W

    def step(self, v: torch.Tensor, x: torch.Tensor, dt: float) -> torch.Tensor:
        """One explicit Euler step of the rate ODE. `v`, `x` are (batch, n_nodes)."""
        rate = self.firing_rate(v)
        if torch.is_grad_enabled():
            w = self.edge_weights()
            current = torch.zeros_like(v)
            current.index_add_(1, self.dst, rate[:, self.src] * w)
        else:
            current = torch.sparse.mm(self._weight_matrix(), rate.T).T
        tau = torch.clamp(self.node_tau(), min=dt)
        dv = (-v + self.node_bias() + current + x) / tau
        return v + dt * dv

    def run(
        self,
        x: torch.Tensor,
        n_steps: int,
        dt: float = 1e-3,
        v0: torch.Tensor | None = None,
        record: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate for `n_steps`. Returns the final state, or the recorded traces
        for the node indices in `record` with shape (batch, n_steps, len(record))."""
        v = self.initial_state(x.shape[0]) if v0 is None else v0
        if record is None:
            for _ in range(n_steps):
                v = self.step(v, x, dt)
            return v
        out = torch.empty(x.shape[0], n_steps, len(record), device=v.device,
                          dtype=v.dtype)
        for t in range(n_steps):
            v = self.step(v, x, dt)
            out[:, t] = v[:, record]
        return out

    def firing_rate(self, v: torch.Tensor) -> torch.Tensor:
        r = torch.relu(v)
        if self.rate_max is None:
            return r
        return self.rate_max * torch.tanh(r / self.rate_max)   # smooth, bounded, ~linear near 0

    # -- helpers --------------------------------------------------------------

    def drive(self, node_indices, amplitude: float, batch: int = 1) -> torch.Tensor:
        x = torch.zeros(batch, self.n_nodes, device=self.device)
        x[:, torch.as_tensor(np.asarray(node_indices), dtype=torch.long,
                             device=self.device)] = amplitude
        return x

    def shuffle_edges_(self, seed: int = 0) -> None:
        """Destroy the wiring while preserving every edge's weight and sign.

        The control from Shiu et al.: if a result survives this, it came from the
        parameters or the drive, not from the connectome.
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(self.n_edges, generator=g).to(self.dst.device)
        self.dst.copy_(self.dst[perm])
        self._W_key = None                       # drop the cached sparse weight matrix
