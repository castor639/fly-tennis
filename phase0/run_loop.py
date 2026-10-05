"""Phase 3: one fly, one ball, closed loop. No learning yet.

analytic vision -> photoreceptors -> connectome -> leg motor neurons -> joint targets
"""
import argparse, json, time
import numpy as np
import mujoco as mj
import torch

from flygym import Simulation
from flygym.compose import (FlatGroundWorld, NeuroMechFly, KinematicPosePreset,
                            ActuatorType)
from flygym.anatomy import (AxisOrder, JointPreset, Skeleton, ActuatedDOFPreset,
                            ContactBodiesPreset)
from flygym.utils.math import Rotation3D
from flypp.connectome import Connectome
from flypp.network import ConnectomeNetwork
from flypp.vision import AnalyticRetina, angular_radius
from flypp.loop import RetinotopicMap, MotorDecoder

Q = Rotation3D(format="quat", values=[1, 0, 0, 0])
NP = KinematicPosePreset.NEUTRAL
BALL_R = 0.4
PHYS_PER_CTRL = 100          # 10 kHz physics, 100 Hz control
NET_STEPS_PER_CTRL = 10      # network at dt=1ms
RECORD_TYPES = ["T3", "LC11", "LC18", "LC10a", "LPLC2", "LC4", "EPG", "PFL3",
                "DNa02", "DNa01", "DNp01", "MDN"]


def build():
    world = FlatGroundWorld()
    fly = NeuroMechFly(name="fly0")
    sk = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL,
                  axis_order=AxisOrder.ROLL_PITCH_YAW)
    fly.add_joints(sk, neutral_pose=NP)
    dofs = sk.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(dofs, actuator_type="position", neutral_input=NP, kp=50)
    fly.add_leg_adhesion(); fly.colorize(); fly.add_vision()
    world.add_fly(fly, [0, 0, 0.7], Q,
                  bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)
    b = world.mjcf_root.worldbody.add_body(name="ball", pos=[3.0, 0, 0.6])
    b.add_joint(name="ball_free", type=mj.mjtJoint.mjJNT_FREE)
    b.add_geom(name="ball_geom", type=mj.mjtGeom.mjGEOM_SPHERE,
               size=[BALL_R, 0, 0], rgba=[0.15, 0.15, 0.18, 1])
    return Simulation(world), len(dofs)


def main(n_ctrl=200, out=None):
    sim, n_dofs = build()
    sim.reset(); mj.mj_forward(sim.mj_model, sim.mj_data)

    c = Connectome.load("cache/connectome_thr5.npz")
    net = ConnectomeNetwork(c)
    retina = AnalyticRetina()
    rmap = RetinotopicMap(c, "cache/photoreceptor_columns.csv", retina.dirs)
    neutral = sim.get_joint_angles("fly0")[:n_dofs]
    motor = MotorDecoder(c, n_dofs_per_leg=n_dofs // 6, neutral=neutral)

    cam = {s: mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA,
                            f"fly0/{s}_eye_cam_camera") for s in ("l", "r")}
    ball_bid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_BODY, "ball")
    thorax_bid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_BODY, "fly0/c_thorax")
    ball_qadr = sim.mj_model.jnt_qposadr[
        mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_JOINT, "ball_free")]

    rec_idx = {t: c.index_of_types(t) for t in RECORD_TYPES}
    rec_idx = {k: v for k, v in rec_idx.items() if len(v)}
    traces = {k: [] for k in rec_idx}
    traj = []

    v = net.initial_state(1)
    t0 = time.perf_counter()
    with torch.no_grad():
        for k in range(n_ctrl):
            # ball orbits the fly so the visual scene actually changes
            ang = 2 * np.pi * k / n_ctrl
            sim.mj_data.qpos[ball_qadr:ball_qadr + 3] = [
                3.5 * np.cos(ang), 3.5 * np.sin(ang), 0.6]
            sim.mj_data.qpos[ball_qadr + 3:ball_qadr + 7] = [1, 0, 0, 0]
            mj.mj_forward(sim.mj_model, sim.mj_data)

            ball = sim.mj_data.xpos[ball_bid]
            inten = {}
            for s in ("l", "r"):
                pos = sim.mj_data.cam_xpos[cam[s]]
                rot = sim.mj_data.cam_xmat[cam[s]].reshape(3, 3)
                rel = ball - pos
                dist = float(np.linalg.norm(rel))
                d = rot.T @ rel / max(dist, 1e-9)
                inten[s] = retina.response(d[None, :],
                                           np.array([angular_radius(dist, BALL_R)]))
            x = rmap.scatter(inten["l"], inten["r"], net.n_nodes)
            xt = torch.as_tensor(x, device=net.device).unsqueeze(0)

            for _ in range(NET_STEPS_PER_CTRL):
                v = net.step(v, xt, 1e-3)

            targets = motor(v)[0].cpu().numpy()
            sim.set_actuator_inputs("fly0", ActuatorType.POSITION, targets)
            sim.set_leg_adhesion_states("fly0", np.ones(6))
            for _ in range(PHYS_PER_CTRL):
                sim.step()

            for t, idx in rec_idx.items():
                traces[t].append(float(v[0, torch.as_tensor(idx, device=net.device)].mean()))
            traj.append({"t": k / 100.0,
                         "fly": sim.mj_data.xpos[thorax_bid].tolist(),
                         "ball": ball.tolist()})
    dt = time.perf_counter() - t0
    print(f"closed loop: {n_ctrl} control steps in {dt:.2f}s "
          f"-> {n_ctrl/dt:.1f} control steps/s ({n_ctrl/dt/100:.3f}x realtime)")
    print(f"network state finite: {bool(torch.isfinite(v).all())}   "
          f"max|V| {float(v.abs().max()):.2f}")
    print("\nrecorded group activity (mean over run):")
    for t, tr in traces.items():
        print(f"  {t:8s} n={len(rec_idx[t]):5d}  mean {np.mean(tr):+.4f}  "
              f"range {np.min(tr):+.3f}..{np.max(tr):+.3f}")
    sim.close()
    if out:
        json.dump({"traces": traces, "traj": traj,
                   "n_cells": {k: int(len(v_)) for k, v_ in rec_idx.items()}},
                  open(out, "w"))
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200, dest="n_ctrl")
    ap.add_argument("--out", default=None)
    main(**vars(ap.parse_args()))
