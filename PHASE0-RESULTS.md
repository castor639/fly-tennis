# Phase 0 results: measured on an A10G GPU machine, 2026-09-09

Every number here was measured on that machine. Scripts are in `phase0/`
(`bench.py`, `bench3.py`, `bench_gpu.py`, `gpu_vision.py`, `bench_mp.py`).

## Verdict

**Multi-fly works. Vision is the wall.** Ommatidia rendering serialises on the single
GPU and caps the whole system at ~26 control steps/s no matter how much CPU or GPU
parallelism you add. Physics is not the problem and never was.

The plan's Phase 4-5 schedule does not survive this. The offline visual pretraining
mitigation is now mandatory rather than optional, and there is a better option below.

## 1. Two flies in one scene: yes, after patching FlyGym

FlyGym 2.1.0 has **two real bugs** that make `world.add_fly()` fail on the second fly.
Both are in `flygym/compose/world/base_world.py::_add_ground_contact_sensors`:

1. The sensor name is not namespaced per fly
   (`name=f"ground_contact_{leg}_leg"`), so the second fly raises
   `ValueError: Error: repeated name 'ground_contact_lf_leg' in sensor`.
2. `legpos_to_groundcontactsensors_by_fly` is re-assigned to a fresh `defaultdict(dict)`
   on every call, so registering fly1 would wipe fly0's entry even if (1) were fixed.

The registry beside it is already keyed by `fly.name`, and readback in
`simulation.py::_map_internal_groundcontactsensor_ids` resolves via `sensor.name`, so
the intent is clearly multi-fly and this is an oversight, not a design limit.

Patch is `phase0/patch_flygym.py`, idempotent, writes a `.orig` backup, aborts if the
source does not match exactly. **Worth reporting upstream; I have not done so.**

After patching, compilation is exactly what it should be:

| | 1 fly | 2 flies |
|---|---|---|
| `nq` | 133 | 266 |
| `nu` | 48 | 96 |
| `nbody` | 71 | 142 |
| `ncam` | 2 | 4 |
| ommatidia readout | (2, 721, 2) | (2, 721, 2) per fly |
| actuated DoFs/fly | 42 | 42 |

Physics timestep is **1e-4 s (10 kHz)**.

## 2. CPU throughput, 2 flies, one world

| vision cadence | steps/s | × real time |
|---|---|---|
| none | 3,774 | 0.377 |
| 50 Hz | 3,023 | 0.302 |
| **100 Hz** | **2,519** | **0.252** |
| 200 Hz | 1,898 | 0.190 |
| 1 kHz | 646 | 0.065 |
| 10 kHz (every step) | 80 | 0.008 |

Rendering every physics step is both catastrophic *and* wrong: real *Drosophila*
photoreceptors respond in the ~100-200 Hz range, not 10 kHz. 100 Hz is the honest setting.

> Methodology note: my first sweep produced non-monotonic garbage (decim=100 slower than
> decim=50). Cause: the EGL renderer initialises lazily on the first
> `get_ommatidia_readouts` call, so a one-time ~4 s cost sat inside the timed region and
> hit the rarely-rendering configs hardest. Warming up the vision path fixed it. The
> table above is the corrected run.

## 3. MJWarp GPU batching: physics scales, vision does not

`flygym.warp.GPUSimulation(world, n_worlds=N)` runs N copies on the GPU.

| n_worlds | steps/s | env-steps/s | × real time (aggregate) |
|---|---|---|---|
| 1 | 72.7 | 73 | 0.0 |
| 16 | 60.0 | 960 | 0.1 |
| 64 | 53.8 | 3,444 | 0.3 |
| 256 | 40.3 | 10,316 | 1.0 |
| 1,024 | 17.4 | 17,853 | 1.8 |
| 4,096 | 4.8 | 19,565 | 2.0 |

Saturates around 1,024 worlds at ~19.5k env-steps/s. That is ~5× a single CPU world,
useful, but nowhere near the "~300× / 60× real time" headline in the FlyGym README, which
is physics-only marketing and does not describe a vision workload.

**The killer:** `get_ommatidia_readouts` returns `(2, 721, 2)` at *every* `n_worlds`
value: 16, 64, 256, all identical. It is not batched. Confirmed structurally:

```
GPUSimulation.__mro__            = [GPUSimulation, Simulation, object]
GPUSimulation.step               -> overridden (GPU)
GPUSimulation.get_ommatidia_readouts -> inherited from CPU Simulation
```

There is no ommatidia code anywhere in `flygym/warp/`. So MJWarp batches the physics on
GPU while vision falls back to the single-world CPU renderer. **Vision-driven RL cannot
use MJWarp batching as shipped.**

## 4. CPU multiprocessing: scales without vision, flat with it

2 flies per worker, 8 vCPU available.

| workers | physics only (agg steps/s) | with vision@100Hz (agg steps/s) |
|---|---|---|
| 1 | 3,840 | 2,615 |
| 2 | 7,561 | 2,180 |
| 4 | 13,152 | 2,043 |
| 6 | 12,923 | 2,522 |
| 8 | 15,346 | 1,926 |

Physics scales ~4× to 8 workers. **Vision is completely flat: 8 workers are no better
than 1, and slightly worse.** Every worker's EGL rendering contends for the same GPU
context. Throwing cores at it does nothing.

So the ceiling is **~2,600 physics steps/s ⇒ ~26 control steps/s at 100 Hz control**,
whatever the parallelism strategy. 1M control steps ≈ 11 hours. 10M ≈ 4.5 days. Self-play
RL from scratch is not viable at that rate.

## 5. Version pinning (this cost real time)

- `mujoco-warp` pulls `mujoco` 3.12, which violates FlyGym's `mujoco<3.10` pin. Pin
  **`mujoco==3.9.0` + `mujoco-warp==3.9.0.1`**.
- `mujoco-warp` declares `warp-lang>=1.13` with no upper bound, but **warp-lang 1.17.0
  fails kernel codegen** with `WarpCodegenKeyError: Error while parsing function
  "_sensor_pos"`. Pin **`warp-lang==1.15.0`**. Clear `~/.cache/warp` after changing.
- The failure is at kernel compile time, not import: `import mujoco_warp._src.sensor`
  succeeds on all three versions, so an import smoke test does not catch it.

Working set: `flygym==2.1.0`, `mujoco==3.9.0`, `mujoco-warp==3.9.0.1`, `warp-lang==1.15.0`,
Python 3.13, `MUJOCO_GL=egl`.

## 6. What this changes

**Recommended: replace rendered vision with an analytic visual drive during training.**
The arena is a ball and one other fly against a plain background. The ball's angular
position and subtended size at each eye are closed-form from MuJoCo state, so the
photoreceptor array can be driven directly at negligible cost, no rasterisation. That
should restore throughput to the physics ceiling (~15k steps/s on 8 CPU workers, ~19.5k
on GPU), a **6× to 8× improvement**, and it composes with MJWarp batching because it
needs no renderer at all. Use true ommatidia rendering only for evaluation and video,
where 26 steps/s is perfectly adequate.

This is defensible rather than a cheat: the LC/T3 circuits we care about are small-object
detectors whose relevant input *is* angular size and position. It should be validated by
checking that the analytic drive and the rendered drive produce matching LC11/T3 responses
on the same trajectory.

Secondary options, in order of cost:
- Pretrain the visual front end offline on recorded sequences (flyvis does exactly this),
  then fine-tune the closed loop at low vision rate.
- Implement batched per-world ommatidia readout for `GPUSimulation`. Real work, but
  genuinely upstreamable and would benefit everyone using FlyGym for RL.

Phases 1-3 of the plan are unaffected. Phase 4-5 should assume the analytic visual drive.
