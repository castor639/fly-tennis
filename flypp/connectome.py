"""MaleCNS v1.0 connectome -> signed sparse tensors for a flyvis-style network.

Follows the flyvis formulation (Lappalainen et al., Nature 2024):

    w_ij = sign_ij * N_syn,ij * alpha_(type_i, type_j)

`sign` and `N_syn` are frozen from the connectome. Only `alpha` trains, and it is
shared per ORDERED CELL-TYPE PAIR, which is what keeps the free-parameter count in
the thousands rather than the millions.

Sign convention follows Shiu et al. 2024 (Nature 634:210-219) Methods:
acetylcholine excitatory; GABA and glutamate inhibitory (glutamate acts on GluCl in
the fly); histamine inhibitory (photoreceptor transmitter); dopamine, octopamine and
serotonin folded into excitatory. Dale's law: one sign per presynaptic neuron.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

EXCITATORY = {"acetylcholine", "dopamine", "octopamine", "serotonin"}
INHIBITORY = {"gaba", "glutamate", "histamine"}

DATA = Path(__file__).resolve().parent.parent / "data"
WEIGHTS = DATA / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
ANNOT = DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANS = DATA / "body-neurotransmitters-male-cns-v1.0.feather"


@dataclass
class Connectome:
    """A thresholded, signed, cell-type-grouped MaleCNS graph."""

    body_ids: np.ndarray        # (N,) int64, connectome body id per node
    cell_type: np.ndarray       # (N,) object, cell type string ("" if untyped)
    superclass: np.ndarray      # (N,) object
    sign: np.ndarray            # (N,) int8, +1 / -1, per PRESYNAPTIC neuron (Dale)
    sign_known: np.ndarray      # (N,) bool, False where we defaulted
    src: np.ndarray             # (E,) int32 node index
    dst: np.ndarray             # (E,) int32 node index
    n_syn: np.ndarray           # (E,) float32 synapse count
    edge_sign: np.ndarray       # (E,) float32, sign of the source neuron
    group_id: np.ndarray        # (E,) int32, index into group_keys
    group_keys: np.ndarray      # (G, 2) object, (type_pre, type_post)
    threshold: int

    @property
    def n_nodes(self) -> int:
        return len(self.body_ids)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    @property
    def n_groups(self) -> int:
        return len(self.group_keys)

    def summary(self) -> str:
        exc = int((self.sign > 0).sum())
        return (
            f"MaleCNS v1.0, synapse threshold >= {self.threshold}\n"
            f"  nodes            {self.n_nodes:,}\n"
            f"  edges            {self.n_edges:,}\n"
            f"  synapses         {int(self.n_syn.sum()):,}\n"
            f"  cell-type pairs  {self.n_groups:,}   (= trainable alpha count)\n"
            f"  excitatory       {exc:,} ({100 * exc / self.n_nodes:.1f}%)\n"
            f"  sign defaulted   {int((~self.sign_known).sum()):,} "
            f"({100 * (~self.sign_known).sum() / self.n_nodes:.2f}%)"
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            body_ids=self.body_ids,
            cell_type=self.cell_type.astype("U32"),
            superclass=self.superclass.astype("U32"),
            sign=self.sign,
            sign_known=self.sign_known,
            src=self.src,
            dst=self.dst,
            n_syn=self.n_syn,
            edge_sign=self.edge_sign,
            group_id=self.group_id,
            group_keys=self.group_keys.astype("U32"),
            threshold=np.int32(self.threshold),
        )

    @classmethod
    def load(cls, path: Path) -> "Connectome":
        z = np.load(path, allow_pickle=False)
        return cls(
            body_ids=z["body_ids"], cell_type=z["cell_type"],
            superclass=z["superclass"], sign=z["sign"],
            sign_known=z["sign_known"], src=z["src"], dst=z["dst"],
            n_syn=z["n_syn"], edge_sign=z["edge_sign"],
            group_id=z["group_id"], group_keys=z["group_keys"],
            threshold=int(z["threshold"]),
        )

    def index_of_types(self, *types: str) -> np.ndarray:
        """Node indices whose cell type is exactly one of `types`."""
        return np.flatnonzero(np.isin(self.cell_type, np.asarray(types, dtype=object)))

    def index_of_superclass(self, *classes: str) -> np.ndarray:
        return np.flatnonzero(np.isin(self.superclass, np.asarray(classes, dtype=object)))


def _resolve_signs(bodies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-neuron sign, Dale's law. Returns (sign int8, known bool).

    consensus_nt first, celltype_predicted_nt as fallback, excitatory as last resort
    (acetylcholine is 63% of the graph, so it is the least-surprising default). The
    `known` mask makes every defaulted neuron auditable rather than silently assumed.
    """
    nt = feather.read_table(
        NEUROTRANS, columns=["body", "consensus_nt", "celltype_predicted_nt"]
    ).to_pandas().drop_duplicates("body").set_index("body")
    sub = nt.reindex(bodies)

    chosen = sub["consensus_nt"].where(
        sub["consensus_nt"].notna() & (sub["consensus_nt"] != "unclear"),
        sub["celltype_predicted_nt"],
    )
    known = chosen.notna() & chosen.isin(EXCITATORY | INHIBITORY)
    sign = np.where(chosen.isin(INHIBITORY), -1, 1).astype(np.int8)
    return sign, known.to_numpy()


def _group_edges(
    cell_type: np.ndarray,
    superclass: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    min_group_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each edge a trainable-parameter group, with hierarchical backoff.

    flyvis groups by ordered cell-type pair and gets ~734 parameters, because the
    optic lobe is columnar: ~65 cell types repeat across ~700 retinotopic columns,
    so every type pair covers hundreds of synapses.

    The whole CNS is not like that. MaleCNS has ~11.7k cell types and the median
    type pair covers 2 edges, so pure type-pair grouping yields ~948k parameters -
    barely any sharing, and far too many to fit from a reward signal.

    So: use the cell-type pair where it is backed by at least `min_group_edges`
    edges, and otherwise back off to the ordered superclass pair (~20 superclasses,
    so at most a few hundred coarse groups). Set `min_group_edges=1` to recover the
    pure flyvis behaviour.
    """
    fine = pd.MultiIndex.from_arrays([cell_type[src], cell_type[dst]])
    fine_codes, fine_uniques = pd.factorize(fine, sort=True)
    counts = np.bincount(fine_codes, minlength=len(fine_uniques))
    keep_fine = counts[fine_codes] >= min_group_edges

    coarse = pd.MultiIndex.from_arrays([superclass[src], superclass[dst]])
    coarse_codes, coarse_uniques = pd.factorize(coarse, sort=True)

    kept_fine_ids = np.unique(fine_codes[keep_fine])
    remap = np.full(len(fine_uniques), -1, dtype=np.int64)
    remap[kept_fine_ids] = np.arange(len(kept_fine_ids))

    group_id = np.empty(len(src), dtype=np.int32)
    group_id[keep_fine] = remap[fine_codes[keep_fine]]
    group_id[~keep_fine] = len(kept_fine_ids) + coarse_codes[~keep_fine]

    keys = [list(fine_uniques[i]) for i in kept_fine_ids]
    keys += [[f"sc:{a}", f"sc:{b}"] for a, b in coarse_uniques]
    return group_id, np.array(keys, dtype=object)


def build(threshold: int = 5, min_group_edges: int = 20,
          weights_path: Path = WEIGHTS) -> Connectome:
    edges = feather.read_table(
        weights_path, columns=["body_pre", "body_post", "weight", "type_pre", "type_post"]
    ).to_pandas()
    edges = edges[edges["weight"] >= threshold]

    bodies = np.union1d(
        edges["body_pre"].to_numpy(), edges["body_post"].to_numpy()
    ).astype(np.int64)
    lookup = pd.Series(np.arange(len(bodies), dtype=np.int32), index=bodies)

    src = lookup.reindex(edges["body_pre"].to_numpy()).to_numpy().astype(np.int32)
    dst = lookup.reindex(edges["body_post"].to_numpy()).to_numpy().astype(np.int32)
    n_syn = edges["weight"].to_numpy().astype(np.float32)

    annot = feather.read_table(
        ANNOT, columns=["bodyId", "type", "superclass"]
    ).to_pandas().drop_duplicates("bodyId").set_index("bodyId").reindex(bodies)
    cell_type = annot["type"].fillna("").to_numpy(dtype=object)
    superclass = annot["superclass"].fillna("").to_numpy(dtype=object)

    sign, sign_known = _resolve_signs(bodies)
    edge_sign = sign[src].astype(np.float32)

    group_id, group_keys = _group_edges(
        cell_type, superclass, src, dst, min_group_edges
    )

    return Connectome(
        body_ids=bodies, cell_type=cell_type, superclass=superclass,
        sign=sign, sign_known=sign_known,
        src=src, dst=dst, n_syn=n_syn, edge_sign=edge_sign,
        group_id=group_id, group_keys=group_keys,
        threshold=threshold,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--min-group-edges", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("cache/connectome_thr5.npz"))
    args = ap.parse_args()

    c = build(threshold=args.threshold, min_group_edges=args.min_group_edges)
    print(c.summary())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    c.save(args.out)
    print(f"\nsaved -> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

    print("\n--- sanity: circuit cell types present ---")
    for t in ("T3", "LC11", "LC18", "LC10a", "LPLC2", "LC4", "EPG", "PFL3", "DNa02"):
        print(f"  {t:8s} {len(c.index_of_types(t)):6,}")
    print("\n--- sanity: superclass partition ---")
    for sc in ("ol_sensory", "ol_intrinsic", "visual_projection", "cb_intrinsic",
               "descending_neuron", "vnc_intrinsic", "vnc_motor"):
        print(f"  {sc:20s} {len(c.index_of_superclass(sc)):7,}")
