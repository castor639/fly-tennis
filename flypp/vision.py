"""Analytic visual drive: photoreceptor input without rasterisation.

Phase 0 measured that rendering the ommatidia caps the whole simulation at ~26
control steps/s and that the cap does not move with CPU workers or MJWarp batching
(see PHASE0-RESULTS.md). For a ball-and-one-other-fly arena on a plain background,
the scene is simple enough to compute the retinal image in closed form from MuJoCo
state instead of rendering it.

The mapping from ommatidium to viewing direction is recovered by inverting FlyGym's
own pipeline:

  1. `Retina.ommatidia_id_map` labels each pixel of the CORRECTED (fisheye-undone)
     512x450 image with its ommatidium, so each ommatidium has a centroid there.
  2. `Retina._correct_fisheye` maps corrected -> raw by
        norm = ((2*i - n) / n) / zoom,   src_norm = norm / (1 - k*|norm|^2 + 1e-6)
  3. The raw image is an ordinary MuJoCo perspective camera with fovy 157 deg,
     so a raw pixel maps to a camera-frame ray in the usual way.

Composing those gives a unit viewing direction per ommatidium. `validate_vision.py`
checks the result against actual renders rather than trusting the algebra.
"""

from __future__ import annotations

import numpy as np
import yaml

from flygym import assets_dir


def _load_config() -> dict:
    with open(assets_dir / "model/neuromechfly/vision.yaml") as f:
        return yaml.safe_load(f)


def ommatidia_centroids() -> tuple[np.ndarray, np.ndarray]:
    """Centroid (row, col) of each ommatidium in the corrected image, and pale mask."""
    with np.load(assets_dir / "model/neuromechfly/compound_eye.npz") as d:
        id_map = d["ommatidia_id_map"].astype(np.int32)
        pale = d["pale_mask"].astype(bool)
    n = int(id_map.max())
    rows, cols = np.indices(id_map.shape)
    flat = id_map.ravel()
    counts = np.bincount(flat, minlength=n + 1)[1:]
    r = np.bincount(flat, weights=rows.ravel(), minlength=n + 1)[1:] / counts
    c = np.bincount(flat, weights=cols.ravel(), minlength=n + 1)[1:] / counts
    return np.stack([r, c], axis=1), pale


def ommatidia_directions() -> np.ndarray:
    """Unit viewing direction per ommatidium in the eye-camera frame, shape (721, 3).

    Camera frame is MuJoCo/OpenGL convention: +x right, +y up, looking down -z.
    """
    cfg = _load_config()
    nrows, ncols = cfg["raw_img_height_px"], cfg["raw_img_width_px"]
    zoom = cfg["fisheye_zoom"]
    k = cfg["fisheye_distortion_coefficient"]
    fovy = np.deg2rad(cfg["fovy_per_eye"])

    centroids, _ = ommatidia_centroids()
    dst_row, dst_col = centroids[:, 0], centroids[:, 1]

    # corrected -> raw (the forward map used by _correct_fisheye)
    row_norm = ((2 * dst_row - nrows) / nrows) / zoom
    col_norm = ((2 * dst_col - ncols) / ncols) / zoom
    denom = 1.0 - k * (row_norm**2 + col_norm**2) + 1e-6
    src_row_norm = row_norm / denom
    src_col_norm = col_norm / denom
    src_row = ((src_row_norm + 1) * nrows) / 2
    src_col = ((src_col_norm + 1) * ncols) / 2

    # raw pixel -> camera ray (MuJoCo perspective camera, fovy is vertical)
    half_h = np.tan(fovy / 2)
    half_w = half_h * (ncols / nrows)
    x = (2 * (src_col + 0.5) / ncols - 1) * half_w
    y = (1 - 2 * (src_row + 0.5) / nrows) * half_h
    z = -np.ones_like(x)

    d = np.stack([x, y, z], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


class AnalyticRetina:
    """Closed-form ommatidia response to spherical targets on a plain background.

    Each ommatidium integrates a cosine-falloff acceptance function centred on its
    viewing direction. A target of angular radius `rho` centred at angular distance
    `theta` from an ommatidium occludes it by roughly the overlap of the two caps;
    we use a smooth approximation, which is what a graded photoreceptor sees anyway.
    """

    def __init__(self, acceptance_deg: float = 4.6):
        # ~4.6 deg interommatidial acceptance angle for Drosophila (Gonzalez-Bellido
        # et al. 2011); the exact value only sets the blur scale.
        self.dirs = ommatidia_directions()
        _, self.pale = ommatidia_centroids()
        self.sigma = np.deg2rad(acceptance_deg) / 2.355  # FWHM -> sigma

    @property
    def n_ommatidia(self) -> int:
        return len(self.dirs)

    def response(
        self,
        target_dirs: np.ndarray,
        target_radii: np.ndarray,
        background: float = 1.0,
    ) -> np.ndarray:
        """Per-ommatidium intensity for one eye.

        Args:
            target_dirs: (T, 3) unit vectors to each target, in the eye-camera frame.
            target_radii: (T,) angular radius of each target in radians.
            background: intensity of the empty background.

        Returns:
            (n_ommatidia,) intensity in [0, background]. Targets are dark against a
            bright background, matching a ball seen against lit space.
        """
        if len(target_dirs) == 0:
            return np.full(self.n_ommatidia, background, dtype=np.float32)
        cos = np.clip(self.dirs @ np.asarray(target_dirs).T, -1.0, 1.0)  # (N, T)
        theta = np.arccos(cos)
        # smooth occlusion: 1 inside the target disc, falling off over sigma
        occ = 0.5 * (1.0 - np.tanh((theta - target_radii[None, :]) / self.sigma))
        total = np.clip(occ.sum(axis=1), 0.0, 1.0)
        return (background * (1.0 - total)).astype(np.float32)

    def two_channel(self, intensity: np.ndarray) -> np.ndarray:
        """Expand to the (n_ommatidia, 2) pale/yellow layout FlyGym returns."""
        out = np.zeros((self.n_ommatidia, 2), dtype=np.float32)
        out[self.pale, 0] = intensity[self.pale]
        out[~self.pale, 1] = intensity[~self.pale]
        return out


def angular_radius(distance: float, radius: float) -> float:
    """Angular radius subtended by a sphere of `radius` at `distance` from the eye."""
    return float(np.arcsin(np.clip(radius / max(distance, radius + 1e-9), -1.0, 1.0)))


def world_to_eye(
    target_world: np.ndarray, eye_pos: np.ndarray, eye_rot: np.ndarray
) -> tuple[np.ndarray, float]:
    """Direction to a world-frame point in the eye-camera frame, plus the distance."""
    rel = np.asarray(target_world, float) - np.asarray(eye_pos, float)
    dist = float(np.linalg.norm(rel))
    local = np.asarray(eye_rot, float).T @ rel
    return local / max(dist, 1e-9), dist
