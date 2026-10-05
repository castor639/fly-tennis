"""Stage 2: close the loop with a real gait and measure whether the fly turns toward
the ball.

analytic vision -> connectome -> DNa01/02 L-R contrast -> two-value descending
signal -> FlyGym hybrid turning CPG -> legs -> the fly's eyes move -> new scene

The CPG does the gait (as in the NeuroMechFly v2 paper's brain/VNC split); the
connectome does sensing and the steering command. The number reported is heading
error: the angle between the fly's heading and the direction to the ball.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import mujoco as mj
import torch

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.anatomy import ContactBodiesPreset
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (HybridTurningController, HybridControllerObservation,
                                         LocomotionAction, PreprogrammedSteps,
                                         apply_locomotion_action, make_locomotion_fly)
from flypp.connectome import Connectome
from flypp.network import ConnectomeNetwork
from flypp.vision import AnalyticRetina, angular_radius
from flypp.loop import RetinotopicMap

BALL_R = 0.4
PHYS_PER_CTRL = 100
NET_STEPS = 10


def build(ball_xy):
    fly = make_locomotion_fly(name="fly0", add_adhesion=True, colorize=True)
    fly.add_vision()
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                  add_ground_contact_sensors=False)
    b = world.mjcf_root.worldbody.add_body(name="ball", pos=[ball_xy[0], ball_xy[1], BALL_R])
    b.add_geom(name="ball_geom", type=mj.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0],
               rgba=[0.15, 0.15, 0.18, 1])
    sim = Simulation(world)
    return sim, fly


def run(ckpt, ball_xy, seconds, gain, seed=0, trace_types=(), viz_nodes=None):
    sim, fly = build(ball_xy)
    steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    ctrl = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps,
                                   output_dof_order=dof_order)
    sim.reset(); ctrl.reset(seed=seed)
    apply_locomotion_action(sim, "fly0", LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(dof_order), adhesion_onoff=np.ones(6, bool)))
    sim.warmup(); mj.mj_forward(sim.mj_model, sim.mj_data)

    c = Connectome.load("cache/connectome_thr5.npz")
    net = ConnectomeNetwork(c)
    if ckpt:
        net.load_state_dict(torch.load(ckpt, map_location=net.device)["net"])
    retina = AnalyticRetina()
    rmap = RetinotopicMap(c, "cache/photoreceptor_columns.csv", retina.dirs)

    import pyarrow.feather as feather
    ann = feather.read_table("data/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                             columns=["bodyId", "somaSide"]).to_pandas().drop_duplicates("bodyId").set_index("bodyId")
    side = ann.somaSide.reindex(c.body_ids).to_numpy()
    def lr(t):
        i = c.index_of_types(t)
        return (torch.as_tensor(i[side[i] == "L"], device=net.device),
                torch.as_tensor(i[side[i] == "R"], device=net.device))
    dna = [lr("DNa02"), lr("DNa01")]
    tr_idx = {t: torch.as_tensor(c.index_of_types(t), dtype=torch.long, device=net.device)
              for t in trace_types if len(c.index_of_types(t))}

    cam = {s: mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, f"fly0/{s}_eye_cam_camera")
           for s in ("l", "r")}
    # A static body with no joint gets fused into the world at compile time and its
    # NAME disappears (mj_name2id returned -1, and xpos[-1] was the fly's own hind
    # tarsus). The geom survives fusing, so look the ball up as a geom.
    ball_gid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, "ball_geom")
    assert ball_gid >= 0, "ball_geom not found"
    thorax_bid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_BODY, "fly0/c_thorax")

    # blank-scene baseline of the steering contrast, so we steer on the CHANGE the ball causes
    with torch.no_grad():
        blank = torch.as_tensor(rmap.scatter(np.ones(721, np.float32), np.ones(721, np.float32), c.n_nodes),
                                device=net.device).unsqueeze(0)
        v = net.initial_state(1)
        for _ in range(300): v = net.step(v, blank, 1e-3)
        base_contrast = sum(float(v[0, r].mean() - v[0, l].mean()) for l, r in dna)

    n_ctrl = int(seconds * 100)
    heading_err, dist, contrast_log, traj, traces = [], [], [], [], {t: [] for t in tr_idx}
    viz_t = torch.as_tensor(viz_nodes, dtype=torch.long, device=net.device) if viz_nodes is not None else None
    viz_act, signals = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for k in range(n_ctrl):
            ball = sim.mj_data.geom_xpos[ball_gid]
            inten = {}
            for s in ("l", "r"):
                p = sim.mj_data.cam_xpos[cam[s]]; R = sim.mj_data.cam_xmat[cam[s]].reshape(3, 3)
                rel = ball - p; d_ = float(np.linalg.norm(rel))
                inten[s] = retina.response((R.T @ rel / max(d_, 1e-9))[None, :],
                                           np.array([angular_radius(d_, BALL_R)]))
            x = torch.as_tensor(rmap.scatter(inten["l"], inten["r"], c.n_nodes), device=net.device).unsqueeze(0)
            for _ in range(NET_STEPS): v = net.step(v, x, 1e-3)

            # + means right side more active; DNa02 is ipsilateral, so right-active = turn right.
            contrast = sum(float(v[0, r].mean() - v[0, l].mean()) for l, r in dna) - base_contrast
            # Clip to the regime where the CPG turns cleanly. Beyond ~0.3 the fly
            # over-rotates past 180 deg within 2 s and heading measurements wrap.
            turn_right = np.clip(gain * contrast, -0.25, 0.25)
            # MEASURED (phase0/turn_sign2.py, gentle asymmetry, heading and path agree,
            # fly upright): signal [1+d, 1-d] turns the fly RIGHT. An earlier test at
            # d=0.4 read a wrapped heading as a left turn and inverted this.
            signal = np.array([1.0 + turn_right, 1.0 - turn_right])

            for _ in range(PHYS_PER_CTRL):
                obs = HybridControllerObservation.from_sim(sim, "fly0")
                apply_locomotion_action(sim, "fly0", ctrl.step(signal, obs))
                sim.step()

            pos = sim.mj_data.xpos[thorax_bid]; head = sim.mj_data.xmat[thorax_bid].reshape(3, 3)[:, 0]
            to_ball = ball - pos; to_ball[2] = 0; head = head.copy(); head[2] = 0
            ang = np.degrees(np.arctan2(np.cross(head, to_ball)[2], np.dot(head, to_ball)))
            heading_err.append(float(ang)); dist.append(float(np.linalg.norm(to_ball)))
            contrast_log.append(contrast)
            traj.append([round(float(q), 3) for q in list(pos[:2]) + list(ball[:2]) + [ang]])
            for t, i in tr_idx.items(): traces[t].append(float(v[0, i].mean()))
            signals.append([float(signal[0]), float(signal[1])])
            if viz_t is not None: viz_act.append(v[0, viz_t].cpu().numpy())
    wall = time.perf_counter() - t0
    sim.close()
    out = {"heading_err": heading_err, "dist": dist, "contrast": contrast_log,
           "traj": traj, "traces": traces, "wall": wall, "ctrl_steps": n_ctrl,
           "signals": signals}
    if viz_act: out["viz_act"] = np.stack(viz_act)
    return out


def main(a):
    ball_xy = [a.ball_dist * np.cos(np.radians(a.ball_az)), a.ball_dist * np.sin(np.radians(a.ball_az))]
    out = {}
    for tag, ck in (("untrained", None), ("trained", a.ckpt)):
        if ck is None and a.only_trained: continue
        r = run(ck, ball_xy, a.seconds, a.gain, trace_types=("T3", "LC11", "LPLC2", "EPG", "PFL3", "DNa02", "DNp01"))
        e = np.abs(np.array(r["heading_err"]))
        print(f"[{tag:9s}] ball at {a.ball_az:+.0f} deg, {a.ball_dist} mm.  "
              f"|heading err| start {e[:10].mean():6.1f}  end {e[-50:].mean():6.1f} deg   "
              f"dist start {r['dist'][0]:.2f} end {r['dist'][-1]:.2f} mm   "
              f"contrast |mean| {np.mean(np.abs(r['contrast'])):.4f}   "
              f"{r['ctrl_steps']/r['wall']:.1f} ctrl steps/s")
        out[tag] = r
    json.dump(out, open(a.out, "w"))
    print("saved", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cache/stage1.pt")
    ap.add_argument("--ball-az", type=float, default=35.0)
    ap.add_argument("--ball-dist", type=float, default=8.0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--gain", type=float, default=2.5)
    ap.add_argument("--only-trained", action="store_true")
    ap.add_argument("--out", default="cache/track.json")
    main(ap.parse_args())
