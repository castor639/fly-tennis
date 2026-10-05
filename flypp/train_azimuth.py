"""Stage 1: teach the connectome to report where the ball is.

Supervised, end-to-end through the connectome dynamics, the way flyvis trains the
optic lobe on optic flow. A ball is placed at a random azimuth; the analytic retina
drives the photoreceptors; the network runs; a linear readout over the descending
neurons must report sin(azimuth). Gradients flow back through the dynamics into the
per-cell-type gains, time constants and biases. Signs and synapse counts never move.

The interpretability check is separate from the training loss: before and after, we
measure how well the FIXED contrast DNa02_R - DNa02_L (the known steering pair) tracks
azimuth with no readout at all. If training makes that correlation rise, the network
has recruited the real steering neurons rather than hiding the answer in the readout.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn

from flypp.connectome import Connectome
from flypp.network import ConnectomeNetwork
from flypp.vision import AnalyticRetina, angular_radius
from flypp.loop import RetinotopicMap

BALL_R, BALL_DIST = 0.4, 3.2


def eye_frames():
    """Rotation of each eye camera in the body frame, from a standing FlyGym fly."""
    import mujoco as mj
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld, NeuroMechFly, KinematicPosePreset
    from flygym.anatomy import AxisOrder, JointPreset, Skeleton, ActuatedDOFPreset, ContactBodiesPreset
    from flygym.utils.math import Rotation3D
    w = FlatGroundWorld(); f = NeuroMechFly(name="fly0")
    sk = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL, axis_order=AxisOrder.ROLL_PITCH_YAW)
    f.add_joints(sk, neutral_pose=KinematicPosePreset.NEUTRAL)
    a = sk.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    f.add_actuators(a, actuator_type="position", neutral_input=KinematicPosePreset.NEUTRAL, kp=50)
    f.add_vision()
    w.add_fly(f, [0, 0, 0.7], Rotation3D(format="quat", values=[1, 0, 0, 0]),
              bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)
    sim = Simulation(w); sim.reset(); mj.mj_forward(sim.mj_model, sim.mj_data)
    out = {}
    for s in ("l", "r"):
        cid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, f"fly0/{s}_eye_cam_camera")
        out[s] = (sim.mj_data.cam_xpos[cid].copy(), sim.mj_data.cam_xmat[cid].reshape(3, 3).copy())
    sim.close()
    return out


class AzimuthTask:
    def __init__(self, c: Connectome, retina: AnalyticRetina, rmap: RetinotopicMap, eyes, device):
        self.c, self.retina, self.rmap, self.eyes, self.device = c, retina, rmap, eyes, device
        self.blank = torch.as_tensor(
            rmap.scatter(np.ones(721, np.float32), np.ones(721, np.float32), c.n_nodes),
            device=device)

    def drive(self, az: np.ndarray, el: float = 0.0, dist=None, ball_r=None) -> torch.Tensor:
        """Batch of azimuths (rad, +left) -> (B, n_nodes) photoreceptor drive."""
        xs = []
        head = np.mean([self.eyes["l"][0], self.eyes["r"][0]], axis=0)
        dist = np.full(len(az), BALL_DIST) if dist is None else np.asarray(dist)
        br = BALL_R if ball_r is None else ball_r
        for a, D in zip(az, dist):
            # ball in the body frame: +x forward, +y left
            ball = head + D * np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)])
            inten = {}
            for s in ("l", "r"):
                p, R = self.eyes[s]
                rel = ball - p; dist_ = float(np.linalg.norm(rel))
                d = R.T @ rel / max(dist_, 1e-9)
                inten[s] = self.retina.response(d[None, :], np.array([angular_radius(dist_, br)]))
            xs.append(self.rmap.scatter(inten["l"], inten["r"], self.c.n_nodes))
        return torch.as_tensor(np.stack(xs), device=self.device)


def main(a):
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    c = Connectome.load("cache/connectome_thr5.npz")
    net = ConnectomeNetwork(c)
    retina = AnalyticRetina()
    rmap = RetinotopicMap(c, "cache/photoreceptor_columns.csv", retina.dirs)
    task = AzimuthTask(c, retina, rmap, eye_frames(), net.device)

    dn = c.index_of_superclass("descending_neuron")
    dn_t = torch.as_tensor(dn, dtype=torch.long, device=net.device)
    readout = nn.Linear(len(dn), 1).to(net.device)
    nn.init.zeros_(readout.weight); nn.init.zeros_(readout.bias)

    import pyarrow.feather as feather
    ann = feather.read_table("data/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                             columns=["bodyId", "somaSide"]).to_pandas().drop_duplicates("bodyId").set_index("bodyId")
    side = ann.somaSide.reindex(c.body_ids).to_numpy()
    def lr(t):
        i = c.index_of_types(t)
        return (torch.as_tensor(i[side[i] == "L"], device=net.device),
                torch.as_tensor(i[side[i] == "R"], device=net.device))
    dna02 = lr("DNa02"); dna01 = lr("DNa01")

    if a.freeze_net:
        for prm in net.parameters(): prm.requires_grad_(False)
    opt = torch.optim.Adam([
        {"params": [net.log_alpha], "lr": a.lr},
        {"params": [net.log_tau, net.bias], "lr": a.lr * 0.3},
        {"params": readout.parameters(), "lr": a.readout_lr, "weight_decay": 1e-3},
    ])

    # settle once on a blank scene, reuse as the initial state for every episode
    with torch.no_grad():
        v0 = net.initial_state(1)
        for _ in range(300):
            v0 = net.step(v0, task.blank.unsqueeze(0), 1e-3)

    def episode(az, grad: bool):
        x = task.drive(az, dist=rng.uniform(a.dist_min, a.dist_max, size=len(az)), ball_r=a.ball_r)
        v = v0.expand(len(az), -1).clone()
        # The descending neurons are ~5 synapses from the eye and tau is ~20 ms, so the
        # visual signal needs >100 ms to reach them. Settle WITH the ball present under
        # no_grad, then backpropagate only through the last `steps` (truncated BPTT).
        with torch.no_grad():
            for _ in range(a.settle):
                v = net.step(v, x, 1e-3)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for _ in range(a.steps):
                v = net.step(v, x, 1e-3)
            # Readout sees the CHANGE from the blank-scene baseline, scaled to O(1).
            # Raw DN activity is ~0.5 x 1,314 cells; Adam moves every weight ~lr per
            # step regardless of gradient, so raw inputs made the output jump ~6 per
            # step and the loss exploded to 4,000 (v1) / 25 (v2).
            pred = readout((v[:, dn_t] - v0[:, dn_t]) / a.in_scale).squeeze(1)
            fixed = (v[:, dna02[1]].mean(1) - v[:, dna02[0]].mean(1)
                     + v[:, dna01[1]].mean(1) - v[:, dna01[0]].mean(1))
        return v, pred, fixed

    def evaluate(tag):
        az = np.linspace(-1.0, 1.0, 21)
        preds, fixed = [], []
        for i in range(0, len(az), a.batch):
            _, p, f = episode(az[i:i + a.batch], grad=False)
            preds.append(p.detach().cpu().numpy()); fixed.append(f.detach().cpu().numpy())
        preds = np.concatenate(preds); fixed = np.concatenate(fixed)
        tgt = np.sin(az)
        r_read = float(np.corrcoef(preds, tgt)[0, 1]) if preds.std() > 1e-9 else 0.0
        r_fixed = float(np.corrcoef(fixed, tgt)[0, 1]) if fixed.std() > 1e-9 else 0.0
        mse = float(np.mean((preds - tgt) ** 2))
        print(f"  [{tag}] readout r={r_read:+.3f}  mse={mse:.4f}   "
              f"FIXED DNa01/02 R-L vs azimuth: r={r_fixed:+.3f}  swing={fixed.max()-fixed.min():.4f}")
        return {"r_readout": r_read, "mse": mse, "r_dna_fixed": r_fixed,
                "az": az.tolist(), "pred": preds.tolist(), "fixed": fixed.tolist()}

    # Calibrate the readout input scale from measured DN deviations in THIS regime, so
    # the readout sees O(1) inputs by construction. A fixed 0.02 tuned at 3 mm blew the
    # loss to 800 at 8-35 mm.
    with torch.no_grad():
        az_c = rng.uniform(-1.0, 1.0, size=a.batch)
        x_c = task.drive(az_c, dist=rng.uniform(a.dist_min, a.dist_max, size=a.batch), ball_r=a.ball_r)
        v_c = v0.expand(a.batch, -1).clone()
        for _ in range(a.settle + a.steps): v_c = net.step(v_c, x_c, 1e-3)
        dev = (v_c[:, dn_t] - v0[:, dn_t])
        a.in_scale = float(max(dev.std().item(), 1e-4))
        print(f"calibrated readout in_scale = {a.in_scale:.5f}  (DN deviation std, ball {a.ball_r} mm at {a.dist_min}-{a.dist_max} mm)")
    print(f"nodes {c.n_nodes:,}  DNs {len(dn)}  trainable {sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in readout.parameters()):,}")
    log = {"before": evaluate("before"), "loss": [], "r_dna": []}
    alpha0 = net.log_alpha.detach().clone()

    t0 = time.perf_counter()
    for it in range(a.iters):
        az = rng.uniform(-1.0, 1.0, size=a.batch)
        _, pred, fixed = episode(az, grad=True)
        tgt = torch.as_tensor(np.sin(az), dtype=torch.float32, device=net.device)
        loss = ((pred - tgt) ** 2).mean()
        # keep alpha in the bounded regime it started in; large drift means instability
        reg = a.reg * ((net.log_alpha - alpha0) ** 2).mean()
        opt.zero_grad(); (loss + (0 if a.freeze_net else reg)).backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(readout.parameters()), 1.0)
        opt.step()
        log["loss"].append(float(loss.detach()))
        if it % 10 == 0 or it == a.iters - 1:
            fx = fixed.detach().cpu().numpy(); tg = np.sin(az)
            r = float(np.corrcoef(fx, tg)[0, 1]) if fx.std() > 1e-9 else 0.0
            log["r_dna"].append(r)
            print(f"iter {it:4d}  loss {float(loss):.4f}  reg {float(reg):.5f}  "
                  f"DNa r={r:+.2f}  {(time.perf_counter()-t0)/(it+1):.1f}s/it  "
                  f"mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    log["after"] = evaluate("after")
    d = (net.log_alpha.detach() - alpha0)
    log["alpha_moved"] = {"mean_abs": float(d.abs().mean()), "max_abs": float(d.abs().max()),
                          "n_gt_0.1": int((d.abs() > 0.1).sum())}
    print("alpha drift:", log["alpha_moved"])
    torch.save({"net": net.state_dict(), "readout": readout.state_dict(), "in_scale": a.in_scale}, a.out)
    json.dump(log, open(a.out.replace(".pt", "_log.json"), "w"))
    print("saved", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--settle", type=int, default=150)
    ap.add_argument("--in-scale", type=float, default=0.02, dest="in_scale")
    ap.add_argument("--readout-lr", type=float, default=2e-3, dest="readout_lr")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--reg", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dist-min", type=float, default=3.2, dest="dist_min")
    ap.add_argument("--dist-max", type=float, default=3.2, dest="dist_max")
    ap.add_argument("--ball-r", type=float, default=0.4, dest="ball_r")
    ap.add_argument("--freeze-net", action="store_true", dest="freeze_net")
    ap.add_argument("--out", default="cache/stage1.pt")
    main(ap.parse_args())
