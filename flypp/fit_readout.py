"""Fit the descending-neuron azimuth readout in closed form (ridge regression).

Adam + a global grad-norm clip over 45k network parameters starved the 1,314-weight
readout: every trained readout correlated with azimuth (r 0.64-0.85) but had
near-zero amplitude (MSE worse than predicting 0). A least-squares fit on the same
DN deviations gives the readout its proper scale. Network weights are left as saved.
"""
import argparse, numpy as np, torch
from flypp.connectome import Connectome
from flypp.network import ConnectomeNetwork
from flypp.vision import AnalyticRetina
from flypp.loop import RetinotopicMap
from flypp.train_azimuth import eye_frames, AzimuthTask

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=160); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--dist-min", type=float, default=8.0); ap.add_argument("--dist-max", type=float, default=35.0)
ap.add_argument("--ball-r", type=float, default=1.2); ap.add_argument("--settle", type=int, default=170)
ap.add_argument("--ridge", type=float, default=3.0); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--source", default="vp", choices=["vp", "dn"])
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
c = Connectome.load("cache/connectome_thr5.npz"); net = ConnectomeNetwork(c)
ck = torch.load(a.ckpt, map_location=net.device, weights_only=False) if a.ckpt else {"net": net.state_dict()}
if a.ckpt: net.load_state_dict(ck["net"])
retina = AnalyticRetina(); rmap = RetinotopicMap(c, "cache/photoreceptor_columns.csv", retina.dirs)
task = AzimuthTask(c, retina, rmap, eye_frames(), net.device)
SOURCES = {"dn": lambda: c.index_of_superclass("descending_neuron"),
           "vp": lambda: np.concatenate([c.index_of_types(t) for t in ("T3", "LC11", "LC10a", "LC18", "LC4", "LPLC2")])}
src_idx = SOURCES[a.source](); dn = torch.as_tensor(src_idx, dtype=torch.long, device=net.device)
print(f"readout source '{a.source}': {len(src_idx)} cells")
# Sample the ball exactly where it travels in the game, seen from the game's fly pose.
# A fit at eye level (elevation 0) did not transfer: on the court the ball is ~2 mm
# below the eyes, lands on different columns, and the readout came out constant.
from flypp import tennis as TN
flyA = TN.Fly(-1, task.eyes)
def court_drive(n):
    xs, az = [], []
    for _ in range(n):
        ball = np.array([rng.uniform(-TN.HALF_W, TN.HALF_W), rng.uniform(-TN.HALF_L + 2.0, TN.HALF_L), rng.uniform(TN.BALL_R_VIS, 1.8)])
        d = flyA.eye_dirs_to(ball)
        il = retina.response(d["l"][0][None, :], np.array([TN.angular_radius(d["l"][1], TN.BALL_R_VIS)]))
        ir = retina.response(d["r"][0][None, :], np.array([TN.angular_radius(d["r"][1], TN.BALL_R_VIS)]))
        xs.append(rmap.scatter(il, ir, c.n_nodes))
        rel = ball - flyA.pos(); az.append(np.arctan2(-rel[0], rel[1]))      # +left for fly A (left = -x)
    return torch.as_tensor(np.stack(xs), device=net.device), np.array(az)
with torch.no_grad():
    v0 = net.initial_state(1)
    for _ in range(300): v0 = net.step(v0, task.blank.unsqueeze(0), 1e-3)
    X, Y = [], []
    for _ in range(a.n // a.batch):
        x, az = court_drive(a.batch)
        v = v0.expand(a.batch, -1).clone()
        for _ in range(a.settle): v = net.step(v, x, 1e-3)
        dev = (v[:, dn] - v0[:, dn]); dev = dev - dev.mean(dim=1, keepdim=True)
        X.append(dev.cpu().numpy()); Y.append(np.sin(az))
X = np.concatenate(X); Y = np.concatenate(Y)
scale = float(max(X.std(), 1e-4)); Xs = X / scale
n_tr = int(0.75 * len(Xs)); Xa, Ya, Xb, Yb = Xs[:n_tr], Y[:n_tr], Xs[n_tr:], Y[n_tr:]
mu = Xa.mean(0); A = Xa - mu
W = np.linalg.solve(A.T @ A + a.ridge * np.eye(A.shape[1]), A.T @ (Ya - Ya.mean()))
b = float(Ya.mean() - mu @ W)
pred = Xb @ W + b; r = float(np.corrcoef(pred, Yb)[0, 1]); mse = float(np.mean((pred - Yb) ** 2))
print(f"{a.ckpt}: held-out r={r:+.3f}  mse={mse:.4f} (predict-0 mse {np.mean(Yb**2):.4f})  pred range {pred.min():+.2f}..{pred.max():+.2f}  in_scale {scale:.4f}")
ck["readout"] = {"weight": torch.as_tensor(W[None, :], dtype=torch.float32), "bias": torch.as_tensor([b], dtype=torch.float32)}
ck["in_scale"] = scale; ck["readout_idx"] = src_idx.astype(np.int64); ck["readout_source"] = a.source; torch.save(ck, a.out); print("saved", a.out)
