# Phase 1 + analytic visual drive: 2026-09-09

Code: `flypp/connectome.py`, `flypp/vision.py`. Benchmarks and the validation harness
are in `phase0/`. Everything below was measured on the A10G machine.

## 1. Connectome loader

`flypp/connectome.py` turns the three MaleCNS feather files into signed sparse tensors
in the flyvis formulation: `w_ij = sign_ij * N_syn,ij * alpha_(type_i, type_j)`, where
sign and synapse count are frozen and only alpha trains.

Built at the standard ≥5-synapse threshold:

```
nodes            163,903
edges            6,235,682
synapses         89,731,552
cell-type pairs  22,165        (= trainable alpha count)
excitatory       106,761 (65.1%)
sign defaulted   2,476 (1.51%)
```

The edge count reproduces my independent pre-flight measurement (6,235,682) exactly,
and matches what Codex reports for MCNS v1.0. Cached to `cache/connectome_thr5.npz`,
41.8 MB.

### Signs

98.1% of graph neurons get a transmitter directly. Resolution order is `consensus_nt`,
then `celltype_predicted_nt`, then excitatory as a last resort: 2,476 neurons (1.51%)
hit that default, and `sign_known` flags every one of them so the assumption stays
auditable rather than silent.

Composition of the graph: 63% acetylcholine, 17.8% glutamate, 13.4% GABA, 3.6%
histamine. That tracks Shiu et al.'s brain-wide "≈55% cholinergic, 24% glutamatergic,
14% GABAergic" closely enough to believe the import.

Convention (Shiu et al. Methods): ACh, dopamine, octopamine, serotonin → +1;
GABA, glutamate, histamine → −1. One sign per presynaptic neuron (Dale's law).

### Parameter grouping: a real departure from flyvis

Grouping strictly by ordered cell-type pair, as flyvis does, gives **947,555 groups**
for 6.2M edges. That is 6.6 edges per parameter, almost no sharing, and hopeless to
fit from a reward signal.

The reason flyvis gets ~734 parameters is that the optic lobe is **columnar**: ~65 cell
types repeat across ~700 retinotopic columns, so each type pair covers hundreds of
synapses. The whole CNS is not like that. MaleCNS has **11,748 cell types** and the
median type pair covers **2 edges**:

| edges per type pair | groups | share of edges |
|---|---|---|
| ≥1 | 947,555 | 100% |
| ≥5 | 166,192 | 76% |
| ≥20 | 21,922 | 57% |
| ≥100 | 5,020 | 47% |

So the loader uses **hierarchical backoff**: keep the cell-type pair where it is backed
by at least `min_group_edges` edges (default 20), otherwise fall back to the ordered
*superclass* pair. That gives **22,165 parameters for 6.2M edges, a 281× reduction**,
while leaving the well-sampled circuits at full type resolution. `min_group_edges=1`
recovers exact flyvis behaviour.

### Superclass partition survives thresholding

| superclass | count |
|---|---|
| ol_sensory | 4,027 |
| ol_intrinsic | 89,365 |
| visual_projection | 9,200 |
| cb_intrinsic | 32,159 |
| descending_neuron | 1,314 |
| vnc_intrinsic | 13,124 |
| vnc_motor | 701 |

Note `ol_sensory` drops from 6,098 to 4,027 at threshold 5: photoreceptors make many
weak connections. Worth revisiting if the visual front end looks starved.

All target cell types survive intact: T3 1,940 · LC11 143 · LC18 208 · LC10a 275 ·
LPLC2 185 · LC4 126 · EPG 46 · PFL3 24 · DNa02 2.

## 2. Analytic visual drive

`flypp/vision.py` computes the retinal image in closed form instead of rasterising it,
by inverting FlyGym's own pipeline: ommatidium → centroid in the corrected image →
inverse fisheye (`norm/(1 − k·r²)`, k=3.8, zoom=2.72) → MuJoCo perspective ray at
fovy 157°. Result is a unit viewing direction per ommatidium, spanning ±75° azimuth and
±73° elevation, which is what a 157° eye should give.

### It is validated, not assumed

`phase0/validate_vision.py` places a sphere along a randomly chosen ommatidium's own
viewing direction, renders the true readout, differences it against a ball-free
baseline to isolate the ball from the ground plane and the fly's body, and compares the
rendered response centroid against the prediction.

```
32/60 trials with the target visible
  separation, predicted vs rendered centroid:
      median 0.84 deg, p90 3.79 deg, max 4.36 deg
  best-fit rigid rotation between the two sets: 0.68 deg
  residual after applying it: median 0.92 deg
```

Median error is **0.84°, comfortably inside one ommatidium's ~4.6° acceptance angle**.
The best-fit rigid rotation is 0.68° and removing it does not reduce the residual, so
there is **no systematic misalignment to correct**; the remaining error is centroid
discretisation. Earlier pattern-correlation runs on the same setup gave median 0.995.

Two false starts worth recording, because both produced confident-looking wrong numbers:
- Comparing against the raw readout instead of a differenced baseline gave correlation
  0.12, because the rendered image is dominated by the ground plane, not the ball.
- Reading `cam_xpos` before calling `mj_forward` gives a stale pose, which silently
  placed every target outside the field of view ("no visible trials").

### The payoff

Single process, 2 flies + ball, vision at 100 Hz:

| | steps/s | control steps/s |
|---|---|---|
| rendered | 2,587 | 25.9 |
| analytic | 3,770 | 37.7 |

Only 1.46×, but 3,770 is the **physics-only ceiling** measured in Phase 0 (3,774), so
vision now costs essentially nothing. The real result is that it scales, which rendering
did not:

| workers | rendered (agg) | **analytic (agg)** | analytic control steps/s |
|---|---|---|---|
| 1 | 2,615 | 3,841 | 38.4 |
| 2 | 2,180 | 7,569 | 75.7 |
| 4 | 2,043 | 15,059 | 150.6 |
| 6 | 2,522 | 17,768 | 177.7 |
| 8 | 1,926 | 18,941 | **189.4** |

**189.4 vs 25.9 control steps/s, a 7.3× improvement**, and near-linear to 4 workers
(per-worker rate holds at 3,765 of 3,841). 1M control steps is now ~1.5 hours instead
of 11; 10M is ~15 hours instead of 4.5 days. Self-play becomes viable.

### Honest limits of the analytic drive

- No self-occlusion, shading, texture or mesh detail. It models dark spheres on a plain
  background, which is exactly the arena we designed, and nothing more.
- The other fly is currently approximated as a sphere too. If inter-fly appearance turns
  out to matter for LC10a, that needs revisiting; a two-sphere or capsule proxy would
  be the cheap fix.
- Validated on the left eye only, static pose. Re-run the validation on the right eye
  and on a moving fly before trusting it in a closed loop.
- Rendering stays available and should be used for evaluation and video, where 26
  steps/s is perfectly adequate.

## Next

Phase 2: the flyvis-style forward model over this graph in PyTorch, validated against
the Shiu et al. sugar→MN9 result (correct connectome activates MN9; shuffled does not).
