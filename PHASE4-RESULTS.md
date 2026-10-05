# Phase 4: first training, and the fly turns toward the ball. 2026-09-09

Code: `flypp/network.py` (model fixes), `flypp/train_azimuth.py` (Stage 1),
`flypp/track.py` (Stage 2), `flypp/export_viz2.py`. Diagnostics in `phase0/`.

## Headline

With the ball 20 mm away at 35° left and a fixed-speed CPG gait:

| | untrained | **trained** |
|---|---|---|
| closest approach | 11.17 mm | **5.09 mm** |
| min heading error | 24.3° | **16.4°** |
| steering contrast, mean magnitude | 0.032 | **0.072** |
| contrast sign | flips randomly | consistent, correct |

The trained fly gets more than twice as close and turns further toward the ball. It
does not stop there: nothing has taught it to stop, the CPG walks at constant speed,
and after passing the ball both flies wander off (end distance 37 / 47 mm). That is
stages 1-2 of the curriculum (fixate, track), not 3-4 (approach, contact).

## What had to be fixed first: three model changes

Phase 4 opened with a control I should have run in Phase 3: the identical loop with no
ball reproduces every "modulation" reported there to five decimal places. The visual
part was ~1e-6. **Phase 3's dynamical claim was wrong** and is now marked as such.

Three structural problems, each verified by measurement:

**1. In-degree spans three orders of magnitude.** Median 201 input synapses, p99
5,888, max 118,701. No scalar gain can serve both ends: what keeps the top 1% from
diverging leaves everyone else starved, and signal died ~100× per synapse. Fix:
per-neuron input normalisation: each neuron's incoming weights divided by its total
input synapse count, so drive is a weighted average of presynaptic rates. This is the
synaptic-scaling homeostasis real neurons perform. `alpha` moves from O(1e-4) to O(1).

**2. A ReLU rate network has a hard instability edge.** Even normalised, alpha 0.5
settles and alpha 1.0 diverges to 1e12; 65% of synapses are excitatory and the global
loop has gain > 1. Fix: a saturating rate, `r = r_max·tanh(ReLU(V)/r_max)`, r_max = 5.
Real neurons have a maximum firing rate. The network is now bounded at every gain
tested (1 to 16), and gain can finally be set high enough to transmit.

**3. Population means hide localised responses.** Measured per cell instead, with the
ball 30° left, alpha = 4:

| type | cells | max \|ΔV\| | cells > 0.01 |
|---|---|---|---|
| T3 | 1,940 | **0.70** | **79** |
| LPLC2 | 185 | 0.20 | 15 |
| LC11 | 143 | 0.07 | 8 |

79 of 1,940 T3 cells, strongly; that is exactly the retinotopic small-object response
T3 is supposed to have. At alpha = 8 the signal reaches DNa02 (0.034), DNp01 (0.109),
PFL3 (0.035) and EPG (0.047). Default gain is now 8.

## Stage 1: supervised azimuth, end to end through the connectome

A ball at random azimuth drives the analytic retina; the network runs; a linear readout
over the 1,314 descending neurons must report sin(azimuth). Gradients flow through the
dynamics into 45,661 gains / time constants / biases. Signs and synapse counts never
move. 150 iterations, batch 8, ~5 s/iteration, 10 GB.

It took three attempts, and each failure is worth recording:

- **v1** exploded to loss 4,327 at iteration 1. The readout summed 1,314 activities at
  ~0.5 baseline; Adam moves every weight ~lr per step regardless of gradient, so the
  output jumped ~6 per iteration.
- **v2** settled the network *with the ball present* for 150 steps before the 20
  gradient steps; the descending neurons are ~5 synapses from the eye with τ ≈ 20 ms,
  and 25 steps was less than one time constant, so in v1 the signal had never reached
  them (fixed DNa swing 0.0000). The swing became real (0.0134). Still exploded.
- **v3** subtracted the blank-scene baseline from the readout input and scaled it to
  O(1). Stable.

Result: readout correlation with azimuth **0.00 → +0.847**. The **fixed** DNa01/02
right−left contrast, the known walking-steering pair, no readout involved, tracks
azimuth at **r = −0.54, and negative is the biologically correct sign**: DNa02 is
ipsilateral, so a ball on the left activates the left side and R−L goes negative. Only
471 of 22,165 alphas moved by more than 10%.

MSE barely changed (0.294 → 0.303) because the readout output scale stayed tiny, high
correlation, near-zero magnitude, a too-conservative readout learning rate. It does not
matter for steering, which uses the fixed DNa contrast, but it is unfinished.

## Stage 2: closing the loop with a gait

`HybridTurningController` from `flygym_demo` (bundled with FlyGym, not on PyPI) takes a
two-value descending signal and produces the gait, as in the NeuroMechFly v2 paper's
brain/VNC split. The connectome supplies the steering command: DNa01/02 R−L contrast,
gain 2.5, clipped to ±0.25.

Two bugs cost most of the day, both worth knowing:

- **The ball was a leg.** A static MuJoCo body with no joint is fused into the world at
  compile time and its *name* disappears; `mj_name2id` returned −1 and `xpos[-1]` was
  silently the fly's own hind tarsus. That is why the first tracking runs reported the
  ball 2.3 mm away at 167°. Look static objects up as geoms.
- **The turn sign was calibrated wrong, then "corrected" wrong.** A test at
  asymmetry 0.4 read a heading that had wrapped past 180° as a left turn. Redone at
  ±0.15 with the fly verified upright and heading and path agreeing:
  `[1+d, 1−d]` turns **right**. Every 8 mm run before that was steering the trained fly
  correctly *away* from a correctly detected ball.

With those fixed, at 8 mm the trained fly curved left toward the ball (path ends at
y = +12.6 vs −3.6 untrained) but walked past: ~50°/s of turn against ~11 mm/s of
walking is not enough to cover 35° in the 0.7 s before it passes. Moving the ball to
20 mm is what produced the headline numbers. That is a task-geometry fact, not a brain
one.

## Honest limits

- The steering signal is a 4-cell contrast with a hand-set gain. The network's
  *contribution* is the correct, consistent sign and the 2.2× stronger contrast after
  training. The gait is the CPG's.
- The trained contrast keeps a −0.06 offset after the ball leaves the visual field,
  recurrent hysteresis relative to the blank-scene baseline. Harmless here; it would
  matter for a rally.
- The training loss oscillated (medians per 25 iterations: 9.7, 1.5, 0.8, 1.5, 1.4,
  0.57). Trending down, not converged.
- One ball position, one seed, one gait. This is a demonstration that the pipeline
  learns and steers, not a benchmark.

## Next

Stage 3-4: approach and contact need a forward-speed command (a second descending
readout, e.g. DNp09 / MDN for stop and back), a ball with physics, and a reward rather
than a supervised target. Then Phase 5, the second fly.
