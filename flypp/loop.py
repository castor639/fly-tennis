"""Close the loop: analytic vision -> connectome -> motor neurons -> joint targets.

Two interfaces have to be built, and both are grounded in the MaleCNS annotations
rather than invented:

INPUT. MaleCNS assigns hex coordinates (assignedOlHex1/2) to optic-lobe intrinsic
neurons but not to photoreceptors, so `build_retino.py` assigns each photoreceptor the
column of its strongest hex-labelled partner. FlyGym's 721 ommatidia per eye and
MaleCNS's ~890 columns per eye are then matched by nearest neighbour in a normalised 2D
lattice embedding.

  CAVEAT, stated plainly: the absolute rotational alignment between FlyGym's eye and
  the MaleCNS hex lattice is NOT established by anything in either dataset. The map is
  retinotopically smooth and consistent, which is what the model needs to learn spatial
  vision, but "column (3,7) is this specific ommatidium" is not claimed.

OUTPUT. Leg motor neurons are annotated by side (somaSide L/R) and by leg
(subclass fl/ml/hl = front/middle/hind), which maps one-to-one onto FlyGym's six legs.
Within a leg, the mapping from motor pool to the 7 actuated DoFs is a trainable linear
readout - the anatomical motor-neuron-to-muscle assignment exists in principle but is
not built here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from flypp.connectome import Connectome

# FlyGym leg order is lf, lm, lh, rf, rm, rh; MaleCNS uses somaSide + subclass.
LEG_ORDER = [("L", "fl"), ("L", "ml"), ("L", "hl"),
             ("R", "fl"), ("R", "ml"), ("R", "hl")]


def _hex_to_xy(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Axial hex coordinates to cartesian."""
    return np.stack([h1 + 0.5 * h2, h2 * np.sqrt(3) / 2], axis=1)


def _normalise(p: np.ndarray) -> np.ndarray:
    """Centre and scale EACH axis to [-1, 1].

    Per-axis rather than global, because the hex lattice in axial coordinates is a
    sheared parallelogram (normalised y extent ~0.70) while the ommatidia form a
    near-circular disc (~0.98). Scaling globally squashes the lattice against one
    edge of the disc and collapses most ommatidia onto a few columns.
    """
    p = p - p.mean(axis=0)
    scale = np.abs(p).max(axis=0)
    return p / np.where(scale > 0, scale, 1.0)


class RetinotopicMap:
    """Maps FlyGym's 721 ommatidia per eye onto MaleCNS photoreceptor node indices.

    Both sides are 60-degree hex lattices: FlyGym's ommatidia form a radius-15 hexagon
    (721 = 1 + 3*15*16) and MaleCNS gives every optic-lobe column axial hex coordinates
    (assignedOlHex1/2, ~880 columns per side). So the map is a lattice-to-lattice
    alignment - one of the 12 hex symmetries plus an integer offset - not a continuous
    fit. The earlier version stretched the sheared hex lattice onto the ommatidia disc
    axis by axis, which crowded the FRONTAL field (where a ball is) onto edge columns at
    ~1/5 weight per ommatidium; that was the main reason the ball barely registered.

    Orientation: the dorsal-rim photoreceptors (R7d/R8d) sit at large hex1 AND hex2, so
    hex1+hex2 runs ventral->dorsal. Which sign of hex1-hex2 is anterior is not pinned
    by the annotations; `anterior_sign` picks it (it does not affect steering, only
    which named columns fire for a frontal object).
    """

    def __init__(self, c: Connectome, csv_path: str, omm_dirs: np.ndarray, anterior_sign: int = +1):
        from flypp.vision import ommatidia_centroids
        tbl = pd.read_csv(csv_path)
        body_to_node = pd.Series(np.arange(c.n_nodes), index=c.body_ids)
        tbl["node"] = body_to_node.reindex(tbl.bodyId).to_numpy()
        tbl = tbl[np.isfinite(tbl.node)].copy()
        tbl["node"] = tbl.node.astype(int)

        # --- ommatidia: integer axial coords from the image lattice ---
        cen, _ = ommatidia_centroids()
        d2 = ((cen[:, None] - cen[None]) ** 2).sum(-1); np.fill_diagonal(d2, np.inf)
        c0 = int(np.argmin(((cen - cen.mean(0)) ** 2).sum(1)))
        nb = cen[np.argsort(d2[c0])[:6]] - cen[c0]
        v1 = nb[np.argmax(nb[:, 0])]                                   # the two basis vectors at 60 deg
        cands = [v for v in nb if abs(np.degrees(np.arccos(v @ v1 / np.linalg.norm(v) / np.linalg.norm(v1))) - 60) < 5]
        v2 = cands[0]
        ij = np.rint(np.linalg.solve(np.stack([v1, v2], 1), (cen - cen[c0]).T).T).astype(int)   # (721, 2)
        az = np.arctan2(omm_dirs[:, 0], -omm_dirs[:, 2]); el = np.arcsin(np.clip(omm_dirs[:, 1], -1, 1))
        # dorsal / anterior directions of the ommatidia lattice, from the eye geometry
        g_el = np.linalg.lstsq(np.c_[ij, np.ones(len(ij))], el, rcond=None)[0][:2]
        g_az = np.linalg.lstsq(np.c_[ij, np.ones(len(ij))], az, rcond=None)[0][:2]
        omm_of = {(int(a), int(b)): k for k, (a, b) in enumerate(ij)}

        # the 12 symmetries of the hex lattice in axial coordinates
        def q(u, v): return u * u + u * v + v * v
        syms = []
        for m in np.array(np.meshgrid(*[[-1, 0, 1]] * 4)).reshape(4, -1).T:
            M = m.reshape(2, 2)
            if abs(round(np.linalg.det(M))) != 1: continue
            ok = all(q(*(M @ np.array(p))) == q(*p) for p in [(1, 0), (0, 1), (1, 1), (2, -1), (1, -2)])
            if ok: syms.append(M)
        assert len(syms) == 12, len(syms)

        self.n_omm = len(omm_dirs)
        self.flat_nodes: dict[str, np.ndarray] = {}; self.flat_omm: dict[str, np.ndarray] = {}
        self.flat_w: dict[str, np.ndarray] = {}; self.n_columns: dict[str, int] = {}
        self.column_omm: dict[str, dict] = {}; self.n_mapped: dict[str, int] = {}
        for side in ("L", "R"):
            s_ = tbl[tbl.somaSide == side]
            groups = s_.groupby(["hex1", "hex2"]).indices
            keys = np.array(list(groups.keys()), int)
            # MaleCNS hex1/hex2 use a 120-degree basis ((1,1) is a neighbour step: the medulla
            # spans 65 along hex1+hex2 but only 35 along hex1-hex2). Flip hex2 to get the
            # same 60-degree axial system as the ommatidia lattice.
            kax = keys * np.array([1, -1])
            # anterior is +az in the left eye camera and -az in the right (mirrored cameras)
            ant = g_az * (1 if side == "L" else -1) * anterior_sign
            best = None
            for M in syms:
                dv = M @ np.array([1, -1]); ap = M @ np.array([1, 1])
                if dv @ g_el <= 0 or ap @ ant <= 0: continue          # must send dorsal up and anterior forward
                hk = kax @ M.T
                t0 = np.rint(ij.mean(0) - hk.mean(0)).astype(int)
                for dt_ in np.array(np.meshgrid(range(-4, 5), range(-4, 5))).reshape(2, -1).T:
                    t = t0 + dt_
                    hit = sum((int(a), int(b)) in omm_of for a, b in hk + t)
                    if best is None or hit > best[0]: best = (hit, M, t)
            hit, M, t = best
            hk = kax @ M.T + t
            col_omm = {}
            for k, (a, b) in zip(map(tuple, keys), hk):
                o = omm_of.get((int(a), int(b)))
                if o is None:                                          # column beyond FlyGym's 721: nearest ommatidium
                    o = int(np.argmin(q(*(ij - np.array([a, b])).T)))
                col_omm[k] = o
            nodes_all, omm_all = [], []
            for k, idx in groups.items():
                nodes_all.append(s_.node.to_numpy()[idx]); omm_all.append(np.full(len(idx), col_omm[k]))
            self.flat_nodes[side] = np.concatenate(nodes_all); self.flat_omm[side] = np.concatenate(omm_all)
            self.flat_w[side] = np.ones(len(self.flat_nodes[side]), np.float32)
            self.n_columns[side] = len(keys); self.n_mapped[side] = hit; self.column_omm[side] = col_omm

    def coverage(self) -> dict:
        return {side: (self.n_columns[side], self.n_mapped[side],
                       len(np.unique(self.flat_omm[side])), len(self.flat_nodes[side]))
                for side in ("L", "R")}

    def scatter(self, intensity_l: np.ndarray, intensity_r: np.ndarray,
                n_nodes: int) -> np.ndarray:
        """Ommatidia intensities -> a drive vector over all connectome nodes."""
        x = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(x, self.flat_nodes["L"], intensity_l[self.flat_omm["L"]] * self.flat_w["L"])
        np.add.at(x, self.flat_nodes["R"], intensity_r[self.flat_omm["R"]] * self.flat_w["R"])
        return x


class MotorDecoder(nn.Module):
    """Leg motor-neuron pools -> joint targets, one trainable readout per leg."""

    def __init__(self, c: Connectome, n_dofs_per_leg: int = 7,
                 device="cuda", neutral: np.ndarray | None = None):
        super().__init__()
        import pyarrow.feather as feather
        a = feather.read_table(
            "data/body-annotations-male-cns-v1.0-minconf-0.5.feather",
            columns=["bodyId", "superclass", "somaSide", "subclass"]).to_pandas()
        a = a.drop_duplicates("bodyId").set_index("bodyId")
        node_of = pd.Series(np.arange(c.n_nodes), index=c.body_ids)
        ann = a.reindex(c.body_ids)

        self.pools = []
        for side, leg in LEG_ORDER:
            m = ((ann.superclass == "vnc_motor").to_numpy()
                 & (ann.somaSide == side).to_numpy()
                 & (ann.subclass == leg).to_numpy())
            self.pools.append(torch.as_tensor(np.flatnonzero(m), dtype=torch.long,
                                              device=device))
        self.readouts = nn.ModuleList([
            nn.Linear(len(p), n_dofs_per_leg) for p in self.pools])
        if neutral is not None:
            n = torch.as_tensor(neutral, dtype=torch.float32).view(6, n_dofs_per_leg)
            for i, r in enumerate(self.readouts):
                nn.init.zeros_(r.weight)
                r.bias.data.copy_(n[i])
        self.to(device)

    @property
    def pool_sizes(self):
        return [len(p) for p in self.pools]

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """(batch, n_nodes) -> (batch, 42) joint targets."""
        return torch.cat([r(v[:, p]) for r, p in zip(self.readouts, self.pools)], dim=1)


def retinotopic_azimuth(c: Connectome, rmap: "RetinotopicMap", omm_dirs: np.ndarray, eyes: dict,
                        annot_path: str = "data/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                        rounds: int = 3) -> np.ndarray:
    """Fly-centred azimuth (radians, +left) of every node that has a retinotopic position.

    Hex-labelled cells take their column's ommatidium; other cells (T2/T3/Tm3/T4, LC
    types, ...) take the column of their strongest labelled presynaptic partner,
    propagated a few rounds. Each eye camera's azimuth is converted to a fly-centred
    one using the camera's straight-ahead direction, so both eyes share one axis.
    """
    import pyarrow.feather as feather
    a = feather.read_table(annot_path, columns=["bodyId", "somaSide", "assignedOlHex1", "assignedOlHex2"]).to_pandas()
    a = a.drop_duplicates("bodyId").set_index("bodyId").reindex(c.body_ids)
    side = a.somaSide.to_numpy(); h1 = a.assignedOlHex1.to_numpy(float); h2 = a.assignedOlHex2.to_numpy(float)
    cam_az = np.arctan2(omm_dirs[:, 0], -omm_dirs[:, 2])
    phi = np.full(c.n_nodes, np.nan)
    for s, key in (("L", "l"), ("R", "r")):
        fwd = eyes[key][1].T @ np.array([1.0, 0.0, 0.0])            # body forward in this eye camera's frame
        ahead = np.arctan2(fwd[0], -fwd[2])
        # in both mirrored camera frames, azimuth grows toward the fly's right (checked by sweep)
        col_phi = {k: ahead - cam_az[o] for k, o in rmap.column_omm[s].items()}
        m = (side == s) & np.isfinite(h1)
        for i in np.flatnonzero(m):
            k = (int(h1[i]), int(h2[i]))
            if k in col_phi: phi[i] = col_phi[k]
    for _ in range(rounds):
        m = np.isfinite(phi[c.src]) & ~np.isfinite(phi[c.dst])
        e = np.flatnonzero(m)[np.argsort(c.n_syn[m])]                    # ascending: strongest partner writes last
        phi[c.dst[e]] = phi[c.src[e]]
    return phi


class BumpDecoder:
    """Where is the ball, as the connectome's own columnar cells see it.

    Reads the small-object pathway of the medulla (Mi1, Tm3, T2, T3 by default): the
    deviation of every cell from a ball-free twin network, minus the cell type's own mean
    (cancels shared drift), and returns the population-vector azimuth of the strongest
    deviating cells plus that bump's amplitude. No trained parameters.
    """

    def __init__(self, c: Connectome, phi: np.ndarray, types=("Mi1", "Tm3", "T2", "T3"), top: int = 16):
        idx = np.concatenate([c.index_of_types(t) for t in types])
        idx = idx[np.isfinite(phi[idx])]
        self.idx = torch.as_tensor(idx, dtype=torch.long)
        self.phi = torch.as_tensor(phi[idx], dtype=torch.float32)
        tid = np.concatenate([[i] * len(c.index_of_types(t)) for i, t in enumerate(types)])
        self.tid = torch.as_tensor(np.concatenate([np.full(int(np.isfinite(phi[c.index_of_types(t)]).sum()), i) for i, t in enumerate(types)]), dtype=torch.long)
        self.n_types = len(types); self.top = top

    def reset(self):
        self.base = None

    def __call__(self, v: torch.Tensor, v_twin: torch.Tensor, dt: float | None = None, tau_adapt: float = 0.5):
        """v: (B, N) fly states, v_twin: (N,) ball-free state. Returns (azimuth[B], amplitude[B]).

        With `dt`, the deviation is high-passed by an exponential baseline (adaptation,
        `tau_adapt` seconds): over a long match the fly's recurrent state drifts away
        from the twin's, and without this the decode locks onto stale history instead of
        the moving ball."""
        d = (v[:, self.idx] - v_twin[self.idx]).to(self.phi.device)
        if dt is not None:
            if getattr(self, "base", None) is None: self.base = d.clone()
            self.base += (d - self.base) * min(1.0, dt / tau_adapt)
            d = d - self.base
        means = torch.zeros(d.shape[0], self.n_types).index_add_(1, self.tid, d) / torch.bincount(self.tid, minlength=self.n_types).float()
        w = (d - means[:, self.tid]).abs()
        top_w, top_i = w.topk(self.top, dim=1)
        az = (top_w * self.phi[top_i]).sum(1) / top_w.sum(1).clamp_min(1e-9)
        return az.numpy(), top_w[:, 0].numpy()
