"""Build the payload for the final display: untrained vs trained fly tracking a ball,
with per-neuron activity, heading error, steering signal and the training log."""
import base64, json, os
import numpy as np
import pyarrow.feather as feather
from flypp.connectome import Connectome
from flypp.track import run

N_VIZ = 9000
SECONDS = 5.0
BALL = (35.0, 20.0)   # azimuth deg, distance mm - far enough that a ~50 deg/s turn completes before the fly passes it
CHAN = {
    "Small object (T3)": ["T3"], "Small object (LC11)": ["LC11"], "Pursuit (LC10a)": ["LC10a"],
    "Looming (LPLC2, LC4)": ["LPLC2", "LC4"], "Heading (EPG)": ["EPG"], "Steering (PFL3)": ["PFL3"],
    "Turn L (DNa01/02 L)": None, "Turn R (DNa01/02 R)": None, "Escape (DNp01)": ["DNp01"],
}

c = Connectome.load("cache/connectome_thr5.npz")
ann = feather.read_table("data/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                         columns=["bodyId", "somaLocation", "somaSide", "superclass"]).to_pandas()
ann = ann.drop_duplicates("bodyId").set_index("bodyId").reindex(c.body_ids)
side = ann.somaSide.to_numpy()
has = ann.somaLocation.notna().to_numpy()
rng = np.random.default_rng(0)
viz = np.sort(rng.choice(np.flatnonzero(has), N_VIZ, replace=False))
pos = np.stack(ann.somaLocation.to_numpy()[viz]).astype(np.float32)
pos = (pos - pos.mean(0)) / np.abs(pos - pos.mean(0)).max()
# region tag per viz neuron for the legend: 0 optic lobe, 1 central brain, 2 VNC
sc = c.superclass[viz]
region = np.where(np.char.startswith(sc.astype(str), "ol") | (sc == "visual_projection") | (sc == "visual_centrifugal"), 0,
         np.where(np.char.startswith(sc.astype(str), "vnc"), 2, 1)).astype(np.uint8)

def lr(t):
    i = c.index_of_types(t); return i[side[i] == "L"], i[side[i] == "R"]
dna_L = np.concatenate([lr("DNa02")[0], lr("DNa01")[0]]); dna_R = np.concatenate([lr("DNa02")[1], lr("DNa01")[1]])
chan_idx = {}
for k, ts in CHAN.items():
    if ts is None: chan_idx[k] = dna_L if k.startswith("Turn L") else dna_R
    else: chan_idx[k] = np.concatenate([c.index_of_types(t) for t in ts])

ball_xy = [BALL[1] * np.cos(np.radians(BALL[0])), BALL[1] * np.sin(np.radians(BALL[0]))]
all_nodes = np.unique(np.concatenate([viz] + list(chan_idx.values())))
runs = {}
for tag, ck in (("untrained", None), ("trained", "cache/stage1.pt")):
    r = run(ck, ball_xy, SECONDS, gain=2.5, viz_nodes=all_nodes)
    A = r.pop("viz_act")                       # (T, len(all_nodes))
    col = {n: i for i, n in enumerate(all_nodes)}
    vizA = A[:, [col[n] for n in viz]]
    chans = {k: A[:, [col[n] for n in idx]].mean(1).round(4).tolist() for k, idx in chan_idx.items()}
    e = np.abs(np.array(r["heading_err"])); dd = np.array(r["dist"])
    print(f"[{tag}] |heading err| start {e[:10].mean():.1f}  min {e.min():.1f}  end {e[-50:].mean():.1f} deg   "
          f"dist start {dd[0]:.2f}  CLOSEST {dd.min():.2f} mm at t={dd.argmin()/100:.2f}s  end {dd[-1]:.2f}")
    runs[tag] = {"act": vizA, "chans": chans, "heading_err": [round(x, 2) for x in r["heading_err"]],
                 "dist": [round(x, 3) for x in r["dist"]], "traj": r["traj"], "signals": [[round(a, 3), round(b, 3)] for a, b in r["signals"]]}

# common quantisation so untrained/trained are comparable on the same colour scale
allA = np.concatenate([runs["untrained"]["act"], runs["trained"]["act"]])
lo, hi = np.percentile(allA, 1), np.percentile(allA, 99.5)
def q(A): return (np.clip((A - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
log = json.load(open("cache/stage1_log.json"))
payload = {
    "n": int(N_VIZ), "frames": int(runs["trained"]["act"].shape[0]),
    "pos": base64.b64encode(pos.astype(np.float16).tobytes()).decode(),
    "region": base64.b64encode(region.tobytes()).decode(),
    "runs": {t: {"act": base64.b64encode(q(v["act"]).tobytes()).decode(), **{k: vv for k, vv in v.items() if k != "act"}}
             for t, v in runs.items()},
    "channel_sizes": {k: int(len(v)) for k, v in chan_idx.items()},
    "ball": {"az": BALL[0], "dist": BALL[1], "xy": [round(x, 3) for x in ball_xy]},
    "training": {"loss": log["loss"], "r_dna": log["r_dna"], "before": {k: log["before"][k] for k in ("r_readout", "r_dna_fixed", "mse", "az", "pred", "fixed")},
                 "after": {k: log["after"][k] for k in ("r_readout", "r_dna_fixed", "mse", "az", "pred", "fixed")}, "alpha_moved": log["alpha_moved"]},
    "total_neurons": int(c.n_nodes), "total_edges": int(c.n_edges), "soma_total": int(has.sum()),
    "trainable": 45661,
}
json.dump(payload, open("cache/viz2_payload.json", "w"))
print("saved cache/viz2_payload.json", f"{os.path.getsize('cache/viz2_payload.json')/1e6:.1f} MB")
