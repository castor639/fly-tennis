# 03: Is fly ping pong biologically grounded, and what body do you run it in?

## Short answer on plausibility

**Visual pursuit of a small moving object is a real, well-characterised fly behaviour.**
**Flies hitting a ball back and forth is not.** The honest framing of this project is:
take the circuits a fly genuinely uses to chase a moving target, and put them in an arena
where the target is a ball. The sensing and steering are biological. The rally is invented.

The nearest insect precedent for object play is in **bumblebees**, not flies; Galpayage
Dona et al. and related work on bees rolling wooden balls with no food reward
(https://pubmed.ncbi.nlm.nih.gov/28232576/, review "Object use in insects"
https://pubmed.ncbi.nlm.nih.gov/37828914/). No *Drosophila* equivalent exists. Do not
claim one.

## The circuits a fly really uses to chase a moving target

### 1. Small-object detection (the ball)

- **T3**: 1,940 cells in MaleCNS. Responds omnidirectionally to small moving objects and
  drives body saccades toward them. "Feature detecting columnar neurons mediate object
  tracking saccades in Drosophila", https://doi.org/10.1101/2022.09.21.508959, and
  "Columnar neurons support saccadic bar tracking in Drosophila",
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10115446/
- **LC11**: 143 cells. Excited by objects as small as ~2.2°, **inhibited by large objects
  and by widefield motion**, a matched filter for "small thing moving against a static
  background". Keleş & Frye 2017, *Current Biology* 27:680-687,
  https://doi.org/10.1016/j.cub.2017.01.009
- **LC18**: 208 cells. Detects objects *smaller than the lens resolution*. Klapoetke et al.
  2022, *Neuron* 110:1700-1711, https://doi.org/10.1016/j.neuron.2022.02.013

### 2. Pursuit (the other fly)

- **LC10a**: 275 cells. The courtship pursuit pathway: males use it to orient toward and
  follow a female. Unilateral activation recapitulates orienting.
  Ribeiro et al. 2018, https://pubmed.ncbi.nlm.nih.gov/30033367/
  Gain is gated by arousal state via P1 neurons:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8973426/
- LC10 axons target the **AOTu**, unlike most LC types which target optic glomeruli in
  **PVLP**/**PLP**. Wu, Nern et al. 2016, *eLife* 5:e21022,
  https://elifesciences.org/articles/21022

### 3. Looming (don't get hit)

- **LPLC2** (185 cells) and **LC4** (126) are the looming detectors, and together they are
  ~97.5% of the optic-lobe input to the giant fibre **DNp01**. LPLC2 achieves selectivity
  by radial motion opponency across four dendritic arms.
  Klapoetke et al. 2017, *Nature* 551:237-241, https://doi.org/10.1038/nature24626
  Ache et al. 2019, https://doi.org/10.1016/j.cub.2019.02.017

### 4. Heading and steering (aim)

- **EPG** (46 cells): the head-direction compass in the ellipsoid body.
- **PFL3** (24 cells): computes the **heading−goal error and projects directly onto the
  descending neurons that issue steering commands.** This is the fly's literal
  "turn toward the target" neuron, and it is only 24 cells.
  Westeinde et al., "Transforming a head direction signal into a goal-oriented steering
  command", https://pmc.ncbi.nlm.nih.gov/articles/PMC10881397/
  Mussells Pires et al., "Converting an allocentric goal into an egocentric steering
  signal", https://pmc.ncbi.nlm.nih.gov/articles/PMC10881393/

### 5. Motor command

- **DNa02** (2 cells): activity predicts steering magnitude in walking flies.
  Rayshubskiy et al., *eLife* 2024, https://elifesciences.org/articles/102230
- **DNa01, DNa03, DNp09, MDN** all present. DN naming scheme from Namiki et al. 2018,
  *eLife* 7:e34272, https://elifesciences.org/articles/34272
- MaleCNS has 1,314 descending neurons and 708 VNC motor neurons, so the command→muscle
  step is inside the same graph.

**The chain is complete and every stage is present in MaleCNS v1.0:**
photoreceptors → optic lobe → T3/LC11/LC18/LC10a → optic glomeruli → central complex
(EPG→PFL3) → DNa02 → VNC premotor → leg motor neurons.

## The body

### FlyGym / NeuroMechFly v2: use this one

https://github.com/NeLy-EPFL/flygym · https://neuromechfly.org
Paper: *Nature Methods* 2024, https://www.nature.com/articles/s41592-024-02497-y

- MuJoCo physics, body geometry from a micro-CT scan of a real adult fly.
- Six legs with **adhesion actuators** (can climb walls and ceilings).
- Vision: compound eyes, ommatidia on a hexagonal lattice, the same geometry flyvis
  expects as input.
- Also olfaction (antennae + maxillary palps), joint angles, actuator forces, contact
  forces.
- Explicitly designed for a **two-part brain/VNC model with descending and ascending
  interfaces**, which is exactly the split the MaleCNS graph gives you.

**Multi-fly: supported.** `neuromechfly.org/api_ref/simulation.html` documents the
`Simulation` class as a multi-fly environment holding multiple flies in one MuJoCo scene.
*Flagged: I confirmed this from the docs index, not by running it. Verify it on the GPU machine in
step 1 before committing to the architecture.*

**FlyGym 2.x was rewritten in March 2026** and is not backward compatible. Reported:
~10× CPU speed-up (~2× real time) and **~300× on GPU via Warp/MJWarp (~60× real time)**.
Whether those figures include vision rendering is not stated; assume they do not.
FlyGym 1.x lives on at https://github.com/NeLy-EPFL/flygym-gymnasium.

### flybody: the alternative

https://github.com/TuragaLab/flybody, Google DeepMind + HHMI Janelia.
Paper: *Nature* 2025, https://www.nature.com/articles/s41586-025-09029-4

MuJoCo model with **flight as well as walking**. Ships walking-imitation,
flight-imitation and vision-guided-flight RL environments; 59-dim walking action space;
trained with DMPO distributed over Ray. FlyGym now bundles tutorials for using the flybody
model too (`neuromechfly.org/tutorials/5b_using_flybody_model/`), so the two are not
mutually exclusive.

Use flybody if the ball game ends up airborne. For a walking rally, FlyGym is the better
fit because of multi-fly support and the explicit brain/VNC interface.

## Two-fly reference data

If the rally needs to look fly-like rather than robot-like, real two-fly interaction data
exists: SLEAP (https://pmc.ncbi.nlm.nih.gov/articles/PMC9007740/) is the standard
multi-animal pose tracker and ships *Drosophila* social-interaction datasets. Useful as an
imitation-learning prior or an evaluation baseline, not required for a first version.
