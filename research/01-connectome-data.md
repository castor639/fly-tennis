# 01: Connectome data sources

Verified 2026-09-09 by direct download and inspection, not by reading summaries.
Every claim below that says "verified" was checked against the actual file.

## The dataset to use: MaleCNS v1.0

The first complete connectome of an entire male *Drosophila* central nervous system,
central brain + optic lobes + ventral nerve cord, with an intact neck connective.
Janelia FlyEM / MRC LMB / University of Cambridge / Google Research.

- Project page: https://www.janelia.org/project-team/flyem/male-cns-connectome
- Data site: https://janelia-flyem.github.io/male-cns/
- Announcement 2025-10-06: https://www.janelia.org/news/researchers-reveal-connectome-of-the-male-fruit-fly-central-nervous-system
- Preprint: "Sexual dimorphism in the complete connectome of the Drosophila male central
  nervous system", https://doi.org/10.1101/2025.10.09.680999
- Interactive: https://neuprint.janelia.org/?dataset=male-cns%3Av1.0

**Why this one and not FlyWire.** FlyWire FAFB is the female *brain only*. It stops at the
neck. MaleCNS includes the ventral nerve cord and the motor neurons, so a sensory→motor
loop lives inside one graph with no cross-dataset stitching. That matters here because the
whole task is visual input → steering → legs.

### Bulk download: public, no auth, no registration

Files live in a public Google Storage bucket. Verified readable anonymously over plain
HTTPS on 2026-09-09:

```
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<file>
```

| File | Size | What it is |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14.5 MB | cell types, class, superclass, soma side |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43.3 MB | per-neuron transmitter predictions |
| `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` | 508 MB | **the adjacency list** |
| `connectome-weights-...-significant-only.feather` | 502 MB | stricter edge filter |
| `connectome-weights-...minconf-0.5.feather` | 1.05 GB | unfiltered |
| `body-stats-...feather` | 778 MB | per-body synapse stats |
| `syn-partners-...feather` | 3.0-6.8 GB | synapse-level partner table |
| `syn-points-...feather` | 13.1 GB | every synapse XYZ |
| `tbar-neurotransmitters-...feather` | 2.7 GB | per-T-bar transmitter |

**You only need the first three: ~566 MB total.** Everything else is for morphology work.

### Verified schema and graph size

`connectome-weights-...-traced-only.feather`, read with pyarrow:

```
cols: ['body_pre', 'body_post', 'weight', 'type_pre', 'type_post']
rows: 25,563,197
```

`weight` is the raw synapse count for that ordered pair. Measured:

```
edges total       : 25,563,197
sum of synapses   : 124,025,046
edges weight>=2   : 15,270,273
edges weight>=3   : 10,511,038
edges weight>=5   :  6,235,682
edges weight>=10  :  2,749,407
unique bodies     : 164,587
```

The ≥5 threshold reproduces the 6,242,118 connections Codex reports for MCNS v1.0, so
**threshold 5 is the community-standard edge filter.** That is the number to build against:
**~165k nodes, ~6.2M edges.** As fp32 that is 25 MB of weights, trivial for an A10G. Even
the unthresholded 25.5M edges is only ~100 MB.

### Verified annotations

`body-annotations-...feather`: 211,577 rows, 164,506 with a `type` assigned (this is the
"166k neurons" figure in the press coverage). The `superclass` column already partitions
the network into exactly the stages a sensorimotor model needs:

| superclass | count | role in the loop |
|---|---|---|
| `ol_sensory` | 6,098 | photoreceptors, **visual input** |
| `ol_intrinsic` | 89,403 | optic lobe processing (54% of the brain) |
| `visual_projection` | 9,201 | LC/LPLC feature detectors → central brain |
| `cb_intrinsic` | 32,164 | central brain, incl. central complex |
| `descending_neuron` | 1,314 | **brain → VNC steering command** |
| `vnc_intrinsic` | 13,161 | VNC premotor |
| `vnc_motor` | 708 | **motor output** |
| `ascending_neuron` | 1,846 | VNC → brain feedback |
| `cb_sensory` / `vnc_sensory` | 4,868 / 6,370 | other modalities |

Useful columns: `type`, `class`, `subclass`, `superclass`, `somaSide`, `flywireType`,
`hemibrainType`, `mancType` (cross-dataset keys), `dimorphism`, `fruDsx`.

### Verified transmitter / sign data

`body-neurotransmitters-...feather`, 1,835,518 rows, columns include `body`, `cell_type`,
`predicted_nt`, `celltype_predicted_nt`, `consensus_nt`, `ground_truth`.

`consensus_nt` distribution:

```
unclear         1,671,117
acetylcholine     104,193
glutamate          29,443
gaba               22,196
histamine           8,024
dopamine              396
octopamine            101
serotonin              48
```

Standard sign convention, following Shiu et al. 2024 Methods
(https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/): acetylcholine → **+1**; GABA and
glutamate → **−1** (glutamate is inhibitory in the fly via GluCl); histamine → **−1**
(photoreceptor transmitter, inhibitory); dopamine/octopamine/serotonin → treated as +1.
Shiu et al. assign a single sign per neuron (Dale's law), taking the majority over that
neuron's presynaptic sites.

**Caveat, flagged:** the large `unclear` count is per-row, and rows are not one-per-neuron.
The per-cell-type columns (`celltype_predicted_nt`) are the ones to fall back on when an
individual body is unclear. I have not yet measured coverage at the neuron level; do that
before assuming every node gets a sign.

Transmitter predictions come from Eckstein et al. 2024 *Cell*,
https://doi.org/10.1016/j.cell.2024.03.016, reported 87% per-synapse, 94% per-neuron.

### Verified circuit cell types present

Queried the annotations directly. All target circuits exist and are typed:

| type | count | function |
|---|---|---|
| `LC10a` | 275 | courtship pursuit of a moving target |
| `LC10*` | 960 | LC10 family |
| `T3` | 1,940 | small-object detection, drives tracking saccades |
| `LC11` | 143 | small moving object |
| `LC18` | 208 | objects smaller than lens resolution |
| `LPLC2` | 185 | looming |
| `LC4` | 126 | looming (velocity) |
| `EPG` | 46 | heading / compass |
| `PFL3` | 24 | **heading−goal error → steering** |
| `PFL2` | 12 | steering |
| `DNa02` | 2 | descending turn command |
| `DNa01`, `DNa03`, `DNp09`, `MDN` | 2, 2, 2, 4 | other descending commands |
| `R7*`, `R8*` | 1,385, 1,329 | photoreceptors |

## Alternatives, for the record

From the Codex download page (https://codex.flywire.ai/api/download, requires Google
sign-in for compute features, static pages readable):

| Dataset | Neurons | Connections |
|---|---|---|
| FAFB v783 (female central brain) | 139,255 | 3,732,460 |
| **BANC v888 (female CNS, brain+VNC)** | 158,262 | 3,037,361 |
| MANC v1.2.1 (male VNC only) | 23,665 | 5,305,638 |
| MAOL v1.1 (male right optic lobe) | 52,445 | 6,484,936 |
| MCNS v1.0 (male CNS) | 166,700 | 6,242,118 |

**BANC is the one real alternative**, a female whole-CNS connectome, also brain+nerve cord.
Worth knowing about if a female-specific circuit ever matters. For this project MaleCNS
wins on the public no-auth bulk download alone.

## Python access

```bash
pip install pyarrow pandas          # bulk files, what I used
pip install neuprint-python navis   # API access
```

neuPrint API (needs a free account + token from https://neuprint.janelia.org):

```python
from neuprint import Client, fetch_neurons, fetch_adjacencies
client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)
neurons, syndist = fetch_neurons("DNge104")
edges, info = fetch_adjacencies("DNge104", None)
```

For building a whole-network model, skip the API and use the bulk feather files, one
`pyarrow.feather.read_table` gets the entire graph in seconds.
