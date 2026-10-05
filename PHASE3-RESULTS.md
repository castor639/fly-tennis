# Phase 3: closed loop. 2026-09-09

Code: `flypp/loop.py`, `phase0/build_retino.py`, `phase0/run_loop.py`,
`phase0/export_viz.py`.

## The loop closes

```
analytic vision -> photoreceptors -> 163,903-neuron connectome
   -> leg motor neurons -> joint targets -> MuJoCo -> new visual scene
```

**38.4 control steps/s (0.384x real time)**, one fly and one ball, physics at 10 kHz,
control at 100 Hz, 10 network steps per control step. Network state stays finite
(max |V| = 6.49) over the whole run, no divergence with the Phase 2 gain.

That is close to the physics-only ceiling of 37.7 control steps/s measured in Phase 1,
so **the connectome costs almost nothing on top of the body**: the network runs at
0.97x real time on the A10G while the physics runs at 0.377x, and the physics dominates.

## Two interfaces, both grounded in the annotations

### Input: photoreceptors have no retinotopic coordinates, so they were derived

MaleCNS assigns hex coordinates (`assignedOlHex1/2`) to **23,720 optic-lobe intrinsic
neurons across 1,771 columns**, but to **zero photoreceptors**. Since R1-R6 terminate on
the lamina of one column and R7/R8 on the medulla of that same column, each
photoreceptor was assigned the column of its strongest hex-labelled partner:

| type | assigned | total | |
|---|---|---|---|
| R8y / R8p | 481 / 330 | 481 / 330 | **100%** |
| R7y | 468 | 482 | 97.1% |
| R8_unclear | 436 | 442 | 98.6% |
| R7p | 296 | 332 | 89.2% |
| **R1-R6** | **1,386** | **3,377** | **41.0%** |

R7/R8 come out essentially complete. R1-R6 lags because **the lamina is the one optic
neuropil this dataset does not fully reconstruct**; Nern et al. 2025 state it directly
("all optic lobe neuropils in their entirety, except for the most peripheral neuropil,
the lamina"). That is a dataset limitation, not a mapping bug. Total: 3,937
photoreceptors over 1,418 columns.

The ommatidia-to-column match runs **column -> nearest ommatidium**, not the reverse.
Matching ommatidium -> column collided badly, driving only ~41% of columns and leaving
most of the optic lobe dark; inverting it drives **all 657 left and 734 right columns**.
The cost is resolution: those columns draw on 262 (L) and 279 (R) distinct ommatidia, so
effective visual resolution is roughly **270 of 721 per eye**, not the full retina.

> **Stated plainly:** the absolute rotational alignment between FlyGym's eye and the
> MaleCNS hex lattice is not established by anything in either dataset. The map is
> retinotopically smooth and consistent, enough to learn spatial vision, but
> "column (3,7) is this exact ommatidium" is not claimed.

### Output: the leg mapping is anatomical, the within-leg readout is learned

`vnc_motor` neurons carry `somaSide` (L/R) and `subclass` (`fl`/`ml`/`hl` = front,
middle, hind leg), which maps one-to-one onto FlyGym's six legs:

| leg | lf | lm | lh | rf | rm | rh |
|---|---|---|---|---|---|---|
| motor neurons | 67 | 58 | 64 | 63 | 58 | 64 |

374 leg motor neurons in total. The mapping from each pool to its 7 actuated DoFs is a
trainable linear readout, initialised to hold the neutral pose. The anatomical
motor-neuron-to-muscle assignment exists in principle and is not built here.

## CORRECTION (2026-09-09, Phase 4 diagnostic)

The modulation reported in the table below **was not visual**. Running the identical
loop with no ball at all reproduces every swing to five decimal places; the visual
component is ~1e-6. What I reported as the ball driving DNa02 and EPG was the network
settling from its initial state while the ball happened to be moving. Two root causes,
both fixed in Phase 4: in-degree spans three orders of magnitude so no scalar gain can
both transmit and stay stable (fixed by per-neuron input normalisation), and a ReLU rate
network has a hard instability edge that pinned the gain ~100x too low to transmit
(fixed by a saturating rate). With those, a ball 30 deg left produces a localised T3
response in 79 of 1,940 cells (max 0.70) and reaches DNa02 and DNp01. Third time the
same lesson: **always difference against a stimulus-absent baseline.**

## What the loop actually does (superseded, see correction above)

Driving a ball in a circle around the fly and reading group activity:

| group | cells | swing over the run |
|---|---|---|
| Turn L (DNa01/02) | 2 | **0.096** |
| Heading (EPG) | 46 | **0.095** |
| Turn R (DNa01/02) | 2 | 0.088 |
| Escape (DNp01) | 2 | 0.078 |
| Looming (LPLC2, LC4) | 311 | 0.044 |
| Backward (MDN) | 4 | 0.021 |
| Steering (PFL3) | 24 | 0.019 |
| Legs (vnc_motor) | 701 | 0.016 |
| Vision R (visual_projection) | 4,612 | 0.007 |
| Small object (T3, LC11, LC18) | 2,291 | 0.001 |

The signal reaches the descending steering neurons: DNa01/02 modulate most of anything
measured, and EPG (the heading compass) tracks the ball's angular position. That is the
end-to-end result: a measured wiring diagram carries a visual signal from photoreceptors
to the neurons that command turning.

**But modulation is weak, and T3 is nearly flat.** Two reasons, both expected: the
per-cell-type gains are untrained, so the network transmits poorly; and averaging
thousands of cells washes out a response that is spatially localised to the few columns
seeing the ball. T3 being flat specifically is worth chasing in Phase 4; it is the
small-object detector this whole task depends on.

## Display

The display page at this stage (data from `phase0/export_viz.py`; the page itself is not in
this repo) rendered 9,000 of the 139,622 neurons that have measured soma positions
in MaleCNS, coloured by activity, rotatable, with the readout channels and the arena
trajectory alongside. Positions come straight from the `somaLocation` column of the
annotations file, no extra download needed.

Readout bars are normalised per channel, which is stated on the page. The page also says
plainly that the brain is untrained.

## Caveats

- One fly, not two. The second fly and the ball physics come in Phase 5.
- The ball is teleported on a scripted circle rather than being hit, no contact
  dynamics yet.
- Visual drive gain (6.0 for the export) is a free choice, not fit to anything.
- Effective visual resolution ~270 of 721 ommatidia per eye, per the mapping above.

## Next

Phase 4: train stages 1-4 (fixate, track, approach, contact) on a single fly.
The immediate diagnostic is why T3 does not modulate, since small-object detection is
the input the whole task rests on.
