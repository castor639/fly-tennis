# 04: The Sept 2026 demo wave, and what nobody has done yet

## The wave

MaleCNS v1.0 dropped and a fortnight of demos followed. Ranked by what they actually do:

| Demo | Author | Repo | What it really is |
|---|---|---|---|
| Fly Escape Room | @dzhng | `dzhng/fly-escape` (20★) | game NPCs, 166k neurons, built with GPT-6 Astra over 3 days |
| NeuroCraft Fly (Minecraft) | @evnsnclr | `evnsnclr/neurocraft-fly-public` (61★) | **docs only, no code released** |
| DOOMFLY | @nftechie_ | `nftechie/doomfly` (32★) | full graph, no cropping; **failed its own validation gates** |
| Fly64 (Mario 64) | @barrelshifter | `ornata/fly` (8★) | "literally 100% vibe coded… I have not reviewed the code" |
| Bad Apple | @mellowskrub | none | no code, unverifiable |
| YMCA poses | @dhruvbhatia0 | none | "fine-tuned", i.e. learned weights, not connectome-derived |
| Beat Saber | unattributed | none found | HUD reads 165,122 neurons (a `status = Traced` filter) |
| Neural Canvas | Xenova | HF Space, **Apache-2.0** | WebGPU LIF kernel, the only reusable runtime |
| Infinite Sugar | cnqso | `cnqso/infinite-sugar` (15★) | FlyWire + `flybody` in WASM |
| closed-loop-fly | ZeroXClem | `ZeroXClem/closed-loop-fly` (1★) | composes two others' work; publishes negative results |

Pre-dating the wave and still the most-starred: `DenisSergeevitch/desktop-fly` (792★,
created 2026-08-18), originally FlyWire, retrofitted with a MaleCNS locomotor extract in
v1.1.0 on 2026-09-05.

**The wave barely touched Hacker News.** The only thread with real discussion (198
comments) is the pre-MaleCNS desktop-fly post. Everything MaleCNS-specific sits at 1-7
points. This lived on X.

## The honest read: what is missing from all of it

Read the READMEs rather than the tweets and a consistent pattern appears.

1. **The motor side is scripted.** NeuroCraft: neural readouts "choose and adjust
   movements **I programmed**" (the author's own words). Infinite Sugar: "The leg motion
   pattern is **supplied by the application**." This is neural activity *selecting* from a
   library of animations, not producing motion.
2. **The loop is open.** Xenova's Neural Canvas states it plainly: "**body motion does not
   feed sensory signals back into the neural network**." No physics, no contact, no
   aerodynamics.
3. **Nothing learns.** DOOMFLY is the only one that attempted learning (dopaminergic
   plasticity on 4,184 KC→MBON11 connections) and its own status line reads: "**not
   demonstrated learned survival. The current v6 candidate failed its visual, conditioning
   and survival validation gates.**"
4. **No agent interacts with another agent.** Every demo is one fly against a scripted
   environment.

`closed-loop-fly` is the most intellectually honest of the lot: it publishes its own
failures, including that a headline "30× drift reduction" turned out to be a readout
artefact, and that with the artefact gated out the loop drifts ~200° in 20 s.

**So the gap is: a closed sensorimotor loop, with real physics, that actually trains, with
two agents.** That is the whole project.

## There is no shared toolkit

Checked PyPI and npm. Nothing named `malecns` exists on either. The established packages
did not respond to the release at all:

| Package | Latest | Touched for MaleCNS? |
|---|---|---|
| `neuprint-python` 0.6.3 (2026-07-20) | | no |
| `flyvis` 1.2.0 (2026-08-06) | | no, still visual system only |
| `navis` 1.12.0 | | no |
| `flygym` 2.1.0 (2026-06-24) | | no |
| `philshiu/Drosophila_brain_model` | last push 2024-09-14 | dormant |

Everyone loads the raw feather files with pandas and reimplements the Shiu LIF equations
from scratch, in Python, JS, Rust and Swift. `dtch1997/fly-api` names the gap directly:
"Each result is a bespoke codebase. fly-api aims to be **the seam**."

Practical consequence: **you will write the connectome→network loader yourself.** Budget
for it; it is a day, not a week, and my measurements in `01-connectome-data.md` remove most
of the guesswork.

## A note on the circulating numbers

Three different neuron counts circulate, which is a useful fingerprint for who copied whom:

- **166,700 / 25,582,938 connections / 124,177,617 synapses**: neurocraft, doomfly,
  Xenova, closed-loop-fly, fly-escape
- **139,255**: FlyWire, not MaleCNS (desktop-fly, infinite-sugar, fly-api)
- **165,122 / 25.56M**: the Beat Saber demo, filtered to `status = Traced`

My own measurement of `connectome-weights-...-traced-only.feather` gives **25,563,197 edges
over 164,587 bodies, 124,025,046 synapses**, close to but not identical with the
circulating 25,582,938, because that figure comes from the unfiltered `minconf-0.5` file.
Worth stating which file you used whenever you publish a count.

## Prior art for a biological network playing a game

- **DishBrain / Cortical Labs**: Kagan et al. 2022, *Neuron*: cultured cortical neurons on
  a multi-electrode array playing Pong via closed-loop feedback. The canonical "biological
  substrate plays Pong" reference. It is *cultured neurons*, not a connectome, and the
  "sentience" framing drew heavy criticism; the learning effect sizes were small. Cite it
  as inspiration, not as method.
- **Neural circuit policies** (Lechner & Hasani): C. elegans-inspired sparse wiring for
  end-to-end driving. Precedent for "biological wiring as an architecture prior", though
  the wiring is inspired rather than measured.
- **CHIMERA** (`caparison1234/chimera`): the *Drosophila larva* connectome (1,373 neurons,
  22,400 synapses, Winding et al. *Science* 2023) as LIF, wired to a **MuJoCo body**. The
  closest existing architectural template to what we want, just two orders of magnitude
  smaller.
- **flyvis**: the only published, trainable, connectome-constrained network. See
  `02-simulation-frameworks.md`.

Nobody has trained a connectome network on a control task with RL. That is the novel part,
and it is also the risky part.
