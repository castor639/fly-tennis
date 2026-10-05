"""Assign each photoreceptor to a retinotopic column via its hex-labelled partners.

MaleCNS gives hex coordinates (assignedOlHex1/2) to optic-lobe INTRINSIC neurons but
not to the photoreceptors. R1-R6 terminate on the lamina monopolar cells of one
column; R7/R8 bypass the lamina and terminate in the medulla of that column. So the
anchor set has to be every hex-labelled optic-lobe neuron, not just the lamina, and a
photoreceptor takes the column of its strongest such partner.
"""
import numpy as np
import pandas as pd
import pyarrow.feather as feather

ANNOT = "data/body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = "data/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"

a = feather.read_table(ANNOT, columns=["bodyId", "type", "superclass", "somaSide",
                                       "assignedOlHex1", "assignedOlHex2"]).to_pandas()
a = a.drop_duplicates("bodyId").set_index("bodyId")

anchors = a[a.assignedOlHex1.notna() & a.assignedOlHex2.notna()]
print(f"hex-labelled anchor neurons: {len(anchors):,}")
print(f"  distinct (side, hex1, hex2): "
      f"{anchors.groupby(['somaSide','assignedOlHex1','assignedOlHex2'], dropna=False).ngroups}")

photo = a[a.superclass == "ol_sensory"]
print(f"photoreceptors: {len(photo):,}")

edges = feather.read_table(WEIGHTS,
                           columns=["body_pre", "body_post", "weight"]).to_pandas()
e = edges[edges.body_pre.isin(set(photo.index)) & edges.body_post.isin(set(anchors.index))]
print(f"photoreceptor -> hex-labelled edges: {len(e):,}")

e = e.join(anchors[["somaSide", "assignedOlHex1", "assignedOlHex2"]], on="body_post")
best = e.sort_values("weight", ascending=False).drop_duplicates("body_pre")
best = best.join(photo[["type"]], on="body_pre")
print(f"\nphotoreceptors assigned: {len(best):,} ({100*len(best)/len(photo):.1f}%)")
print(f"columns covered: "
      f"{best.groupby(['somaSide','assignedOlHex1','assignedOlHex2'], dropna=False).ngroups}")
print("\nby type:")
tot = photo.type.value_counts()
got = best.type.value_counts()
for t in tot.index[:8]:
    print(f"  {t:14s} {got.get(t,0):5d} / {tot[t]:5d}  ({100*got.get(t,0)/tot[t]:5.1f}%)")
print("\nby side:", dict(best.somaSide.value_counts(dropna=False)))
g = best.groupby(["somaSide", "assignedOlHex1", "assignedOlHex2"]).size()
print(f"per-column photoreceptors: min {g.min()} median {int(g.median())} max {g.max()}")

out = best[["body_pre", "somaSide", "assignedOlHex1", "assignedOlHex2", "type"]].rename(
    columns={"body_pre": "bodyId", "assignedOlHex1": "hex1", "assignedOlHex2": "hex2"})
out.to_csv("cache/photoreceptor_columns.csv", index=False)
print("\nsaved -> cache/photoreceptor_columns.csv")
