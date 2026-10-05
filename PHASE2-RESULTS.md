# Phase 2: connectome forward model, validated. 2026-09-09

Code: `flypp/network.py`. Validation: `phase0/validate_network.py`,
`phase0/check_grad.py`, `phase0/stability.py`. Measured on the A10G machine.

## What was built

`ConnectomeNetwork` is the flyvis formulation over the full MaleCNS graph in PyTorch:

```
tau_i dV_i/dt = -V_i + b_i + sum_j w_ij ReLU(V_j) + x_i(t)
w_ij          = sign_ij * N_syn,ij * alpha_g(i,j)
```

`sign` and `N_syn` are frozen buffers, so Dale's law holds by construction and no
amount of training can flip an excitatory synapse to inhibitory. `alpha` is stored as
`log_alpha` so it stays positive and cannot smuggle the sign. Trainable:

| parameter | shape | meaning |
|---|---|---|
| `log_alpha` | 22,165 | synapse strength per parameter group |
| `log_tau` | 11,748 | membrane time constant per cell type |
| `bias` | 11,748 | resting potential per cell type |
| **total** | **45,661** | for 163,903 neurons and 6,235,682 synapses |

Cost on the A10G: **972 neural steps/s at dt=1 ms, 0.97x real time**, 0.28 GB for
inference, 4.62 GB for 40-step BPTT.

## Validation

Shiu et al. validated their whole-brain model by driving sugar GRNs and showing MN9
activates with the true connectome but not a shuffled one. **Sugar GRNs are not
unambiguously labelled in MaleCNS**, `receptorType` covers only ppk23, ppk25 and
IR52b, not the Gr64f sugar receptor, so I ran the same logic on a pathway this dataset
labels unambiguously and that is independently quantified: the looming pathway
LC4 + LPLC2 onto the giant fibre DNp01. (MN9 itself does exist here, 2 cells in
`cb_motor`, so a sugar test becomes possible if the GRN subtypes get pinned down.)

### Structural check: falsifiable against published numbers

```
DNp01: 2 cells, 35,729 input synapses
  LC4     6,362 syn   56.4% of optic-lobe input
  LPLC2   4,836 syn   42.9% of optic-lobe input
  LC4 + LPLC2 = 99.3% of optic-lobe input
```

The published hemibrain figure is **97.5%** (PLOS Biology 2025; Ache et al. 2019).
Getting 99.3% from an independently built loader over a different specimen, sex and
dataset is strong evidence the graph import is correct. The two dominant types match;
their relative split differs (published 45.2% LC4 / 52.2% LPLC2 vs 56.4 / 42.9 here),
which is what you would expect across specimens.

### Dynamical check with the shuffle control

Drive 311 LC4/LPLC2 cells, read DNp01, measure response **above its own undriven
baseline**:

| condition | DNp01 delta | ratio |
|---|---|---|
| true connectome | 0.3672 | n/a |
| shuffled wiring (n=10) | 0.0045 ± 0.0023 | **81.5x lower** |
| random drive, same cell count (n=10) | 0.0059 ± 0.0057 | **61.7x lower** |

0 of 10 shuffled and 0 of 10 random trials beat the true connectome. The shuffle
preserves every weight and sign and permutes only the targets, so the effect is
attributable to the wiring, not the parameters, the same conclusion Shiu et al. drew
from 100% vs 1/100.

### Gradients flow: what Phase 4 depends on

```
log_alpha  21,912 / 22,165 nonzero grads   |g|max 5.1e-01
log_tau    11,603 / 11,748 nonzero         |g|max 2.2e-01
bias       11,608 / 11,748 nonzero         |g|max 9.9e-01
40-step BPTT peak memory: 4.62 GB
```

98.9% of the parameter set is reachable by gradient from a DNa02 readout. RL through
this network is mechanically viable.

## Two findings that changed the model

### 1. Zero resting potential silences the whole network

Initialising `bias = 0` with a ReLU nonlinearity puts every neuron exactly at the
rectification boundary, so inhibitory input has nothing to subtract from. That is fatal
here specifically because **photoreceptors are histaminergic, i.e. inhibitory**: driving
the eye produced literally zero downstream activity.

```
drive all photoreceptors, bias = 0:
  ol_sensory         mean|V| 0.994   max 1.000
  visual_projection  mean|V| 0.003   max 0.000
  everything below   mean|V| 0.000   max 0.000
```

This is the exact failure mode Shiu et al. warn about ("inhibitory connections to an
inactive neuron have no effect"). Fixed by initialising the resting potential positive
(`bias_init=0.5`). With that, the full loop propagates:

```
  ol_sensory         mean|V| 1.490
  visual_projection  mean|V| 0.483
  cb_intrinsic       mean|V| 1.221
  descending_neuron  mean|V| 0.658
  vnc_intrinsic      mean|V| 0.531
  vnc_motor          mean|V| 0.627
```

and every target type responds: T3, LC11, LC10a, LPLC2, EPG, PFL3, DNa02. Gradient
coverage also jumped from 12,269 to 21,912 alphas: a silent network has no gradients.

### 2. flyvis's default gain diverges on the whole CNS

flyvis initialises `alpha = 0.01 / <N_syn>`. That is tuned for the ~45k-node optic
lobe. On the 164k-node whole CNS, with far denser recurrence, it blows up:

| alpha_scale | max\|V\| @50ms | @100ms | @300ms | @600ms | |
|---|---|---|---|---|---|
| 0.01 (flyvis default) | 20.3 | 20.9 | 302.6 | 12,681.4 | **diverging** |
| 0.005 | 13.4 | 13.5 | 13.5 | 13.5 | settles |
| 0.002 | 6.1 | 6.5 | 6.5 | 6.5 | settles |
| 0.001 | 3.4 | 3.6 | 3.6 | 3.6 | settles |

Default is now **0.002**, which leaves margin for alphas to move during training rather
than starting at the edge of stability at 0.005.

## Caveats

- The validation is a *logical analogue* of Shiu et al.'s test, not a reproduction of
  it. Different pathway, rate dynamics rather than spiking LIF.
- Absolute response magnitudes are meaningless until alpha, tau and bias are fit. The
  model currently shows that signals reach the right places, not that they have the
  right sizes. Shiu et al. make the same disclaimer about their own firing rates.
- Stability was checked from one drive condition (all photoreceptors). Training will
  need a stability guard, since gradient steps can push alpha back into the divergent
  regime.
- 40-step BPTT costs 4.62 GB. Full-episode BPTT will not fit; Phase 4 needs truncated
  BPTT or gradient checkpointing.

## Next

Phase 3: close the loop, analytic visual drive into `ol_sensory`, `vnc_motor` out to
joint targets, one fly and one ball, no learning yet. The number that matters is the
true end-to-end env-steps/s with the network in the loop, since the network alone runs
at 0.97x real time and the physics at 0.377x.
