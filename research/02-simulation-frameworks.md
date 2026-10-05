# 02: Turning a connectome into a runnable network

Two published approaches. They differ on the one axis that matters here: **can you
backpropagate a task loss through it?**

## flyvis: trainable, differentiable, PyTorch  ← the base to build on

Lappalainen et al., "Connectome-constrained networks predict neural activity across the
fly visual system", *Nature* 2024. https://doi.org/10.1038/s41586-024-07939-3
Code: https://github.com/TuragaLab/flyvis · Docs: https://turagalab.github.io/flyvis/

I read the source rather than the paper summary. The model is in
`flyvis/network/dynamics.py` and `flyvis/network/initialization.py`.

### The neuron model (verified, `PPNeuronIGRSynapses` in dynamics.py)

"Passive point neurons with instantaneous graded release synapses", a **rate model**, not
spiking. From `write_state_velocity`:

```
dV_i/dt = (1/τ_i) * ( -V_i + b_i + Σ_j w_ij · ReLU(V_j) + x_i(t) )
```

and from `write_derived_params`, the critical line:

```python
params.edges.weight = (
    params.edges.sign * params.edges.syn_count * params.edges.syn_strength
)
```

**w_ij = sign_ij × N_syn,ij × α_(type_i,type_j)**

- `sign`: fixed, from the connectome's transmitter prediction. Dale's law holds by
  construction; training can never flip an excitatory synapse to inhibitory.
- `syn_count`: fixed, the measured synapse count.
- `syn_strength` (α): **the trainable part.**

### What is actually free (verified, initialization.py)

Three `nn.Parameter` groups, all `requires_grad=True`:

| class | meaning |
|---|---|
| `SynapseCountScaling` | α, per **cell-type pair** |
| `TimeConstant` | τ, per cell type |
| `RestingPotential` | b (bias), per cell type |

The trick is `get_scatter_indices(dataframe, grouped_dataframe, groupby)`: parameters are
stored once per *group* and scattered to every edge in that group. With
`groupby=[source_type, target_type]`, millions of synapses share a few thousand numbers.
That is why the paper has ~734 free parameters for a network of ~45k nodes.

Initialization (docstring, verbatim): α_titj = ρ / ⟨N⟩_titj with default ρ = 0.01.

`InitialDistribution.clamp` supports `mode="non_negative"`, which keeps α ≥ 0 so the sign
stays fixed through training.

### Why this matters here

It is ordinary PyTorch. `nn.Parameter` everywhere, a ReLU nonlinearity, an explicit Euler
integration of the ODE. **Gradients flow end-to-end**, so a task loss, including an RL
policy-gradient loss, can be backpropagated into α, τ and b. This is the single most
important fact for the project.

### Limitation

flyvis models the **visual system only**, the optic lobe columnar network on a hex
lattice, trained on an optic-flow task. It has no central complex, no descending neurons,
no motor output. The *architecture pattern* transfers to the whole CNS; the shipped
network does not.

Tutorials worth copying from:
- Connectome exploration: https://turagalab.github.io/flyvis/examples/01_flyvision_connectome/
- Training on the optic flow task: https://turagalab.github.io/flyvis/examples/02_flyvision_optic_flow_task/

## Shiu et al. whole-brain LIF: NOT trainable

Shiu, Sterne et al., "A *Drosophila* computational brain model reveals sensorimotor
processing", *Nature* 634:210-219 (2024). https://doi.org/10.1038/s41586-024-07763-9
Code: https://github.com/philshiu/Drosophila_brain_model (MIT)

Spiking LIF over all 127,400 FlyWire v630 neurons in Brian2:

```
dv_i/dt = (g_i − (v_i − V_rest)) / T_mbr
dg_i/dt = −g_i / τ
g_i ← g_i + w_ji   on presynaptic spike
```

V_rest = V_reset = −52 mV, V_thresh = −45 mV, T_mbr = 20 ms, τ = 5 ms, refractory 2.2 ms,
delay 1.8 ms. Weight is **w = (synapse count) × (±1) × W_syn** with **W_syn = 0.275 mV as
the single free parameter of the whole model**, hand-tuned so sugar GRNs at 100 Hz drive
MN9 to ~80% of max.

Results: 91% of 164 testable predictions consistent with experiment (84% excluding the
split-GAL4 screen). Correct connectivity activates MN9 in 100% of simulations vs 1/100 for
shuffled connectomes, the wiring is load-bearing, not decorative.

**It is forward simulation only.** The Brian2 synapse is declared `'w : volt'` with
`on_pre='g += w'`. No plasticity rule, no `on_post`, no optimizer anywhere in the repo.
Weights are set once and never change.

Cost: **~5 CPU-minutes per 1 second of simulated brain time**, single-threaded;
parallelism is across independent trials only. No GPU.

### The faster reimplementation

https://github.com/eonsystemspbc/fly-brain (GPL-2.0, ~480 stars, active to 2026-08-29)
re-implements the same model across six backends: Brian2, Brian2CUDA, PyTorch CUDA,
NEST GPU, GeNN, Brian2GeNN. Its own `data/benchmark-results.csv` shows 32 trials × 100 s of
brain time in ~4,520 s on Brian2 CPU, roughly **0.7× real time**, far better than the
paper's figure. Notably **Brian2 CPU beat Brian2CUDA by ~2.7×**; this network is sparse and
sparsely active, so the GPU is not an automatic win for the spiking formulation.

Hardware for those benchmarks is not stated in the repo, treat the absolute numbers as
unportable.

## The decision

| | flyvis | Shiu LIF |
|---|---|---|
| dynamics | rate ODE | spiking LIF |
| framework | PyTorch | Brian2 |
| differentiable | **yes** | no |
| coverage | optic lobe only | whole brain, no VNC |
| free params | ~734, per cell-type pair | 1 scalar |
| speed | GPU tensor ops | ~0.7-1× real time on CPU |

**Build the flyvis formulation over the MaleCNS graph.** You get a differentiable network
that covers eye→leg, keeps Dale's law, and trains on an A10G. The Shiu model is the right
reference for parameter values and the right sanity check: if a forward-only LIF over the
same graph reproduces known circuit responses, the wiring import is correct.
