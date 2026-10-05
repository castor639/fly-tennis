# 03a: Simulated fly bodies: FlyGym/NeuroMechFly v2 vs flybody

Research date: 2026-09-09. Every claim carries its source URL. Uncertainty is flagged inline.

---

## 1. FlyGym / NeuroMechFly v2

### Repo, paper, docs
- Code: https://github.com/NeLy-EPFL/flygym (NeLy-EPFL, Ramdya lab)
- Docs: https://neuromechfly.org
- Preprint: https://www.biorxiv.org/content/10.1101/2023.09.18.556649v3.full ("NeuroMechFly 2.0, a framework for simulating embodied sensorimotor control in adult Drosophila")
- Postprint PDF: https://www.epfl.ch/labs/ramdya-lab/wp-content/uploads/2024/08/NMF2_postprint.pdf

### Body model and physics
- Physics engine: **MuJoCo**. NMF 1.0 used PyBullet; v2 moved to MuJoCo for stability, performance, and a wider actuator range including **leg adhesion** (https://www.biorxiv.org/content/10.1101/2023.09.18.556649v3.full).
- Geometry derived from a **micro-CT scan of an adult female Drosophila melanogaster**; antennal segments were adjusted for biological accuracy (https://github.com/NeLy-EPFL/flygym).
- Six legs with **adhesion actuators** enabling climbing of vertical surfaces and overhangs (https://github.com/NeLy-EPFL/flygym).
- GPU acceleration path advertised via **Warp / MJWarp** (https://github.com/NeLy-EPFL/flygym).
- UNVERIFIED so far: exact total DoF count and wing DoFs. The README excerpt did not state them; see the API reference section below.

### Sensory modalities (from https://github.com/NeLy-EPFL/flygym)
- **Vision**: compound eyes modeled as **ommatidia on a hexagonal lattice**; visual input simulated on the fly's retinas.
- **Olfaction**: odor receptors on **antennae and maxillary palps**; chemical intensity computed at receptor locations.
- **Proprioception / mechanosensation**: joint angles, actuator forces, contact forces, and user-defined anatomical joint-site positions.

### Architecture
Hierarchical: a brain level (sensory integration / decision making) and a VNC level (motor control), with descending and ascending representations (https://github.com/NeLy-EPFL/flygym).

### Reported simulation speed
From the repo README (https://github.com/NeLy-EPFL/flygym):
- CPU: **~10x speed-up → ~2x real-time throughput**
- GPU: **~300x speed-up → ~60x real-time throughput**
FLAG: the README text as fetched does not say whether those numbers include vision rendering, nor the baseline they speed up over. Treat as headline marketing numbers, not a vision-RL benchmark.

### MULTI-FLY SUPPORT: preliminary
A web search surfaced the docs statement that FlyGym's **`Simulation` class is a multi-fly simulation environment** using MuJoCo, and that **there can be multiple flies in a `Simulation`** (https://neuromechfly.org/api_ref/simulation.html). Verification of the exact API and its limits follows below.
