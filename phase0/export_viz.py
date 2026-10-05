"""Run the closed loop and export soma positions + per-neuron activity for the
brain visualisation."""
import base64, json
import numpy as np
import mujoco as mj
import torch
import pyarrow.feather as feather

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
N_CTRL = 160
DRIVE_GAIN = 6.0          # visual drive amplitude; untrained alphas transmit weakly
N_VIZ = 9000              # neurons sent to the browser

CHANNELS = {
    "Vision L": ("side", "L", ["visual_projection"]),
    "Vision R": ("side", "R", ["visual_projection"]),
    "Small object": ("types", None, ["T3", "LC11", "LC18"]),
    "Pursuit": ("types", None, ["LC10a"]),
    "Looming": ("types", None, ["LPLC2", "LC4"]),
    "Heading": ("types", None, ["EPG"]),
    "Steering": ("types", None, ["PFL3"]),
    "Turn L": ("dn_side", "L", ["DNa02", "DNa01"]),
    "Turn R": ("dn_side", "R", ["DNa02", "DNa01"]),
    "Escape": ("types", None, ["DNp01"]),
    "Backward": ("types", None, ["MDN"]),
    "Legs": ("sc", None, ["vnc_motor"]),
}


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


def main():
    c = Connectome.load("cache/connectome_thr5.npz")

    ann = feather.read_table("data/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                             columns=["bodyId", "somaLocation", "somaSide"]).to_pandas()
    ann = ann.drop_duplicates("bodyId").set_index("bodyId").reindex(c.body_ids)
    has_soma = ann.somaLocation.notna().to_numpy()
    print(f"nodes with a soma position: {has_soma.sum():,} / {c.n_nodes:,}")

    soma_nodes = np.flatnonzero(has_soma)
    rng = np.random.default_rng(0)
    if len(soma_nodes) > N_VIZ:
        soma_nodes = np.sort(rng.choice(soma_nodes, N_VIZ, replace=False))
    pos = np.stack(ann.somaLocation.to_numpy()[soma_nodes]).astype(np.float32)
    pos = (pos - pos.mean(0)) / np.abs(pos - pos.mean(0)).max()
    print(f"exporting {len(soma_nodes):,} neurons")

    side = ann.somaSide.to_numpy()
    sc = c.superclass

    def chan_idx(kind, arg, names):
        if kind == "types":
            return np.concatenate([c.index_of_types(n) for n in names] or [np.array([])])
        if kind == "sc":
            return c.index_of_superclass(*names)
        if kind == "side":
            base = c.index_of_superclass(*names)
            return base[side[base] == arg]
        if kind == "dn_side":
            base = np.concatenate([c.index_of_types(n) for n in names])
            return base[side[base] == arg]
        return np.array([], dtype=int)

    channels = {k: chan_idx(*v) for k, v in CHANNELS.items()}
    channels = {k: v for k, v in channels.items() if len(v)}
    print("channels:", {k: len(v) for k, v in channels.items()})

    sim, n_dofs = build()
    sim.reset(); mj.mj_forward(sim.mj_model, sim.mj_data)
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

    viz_t = torch.as_tensor(soma_nodes, dtype=torch.long, device=net.device)
    chan_t = {k: torch.as_tensor(v, dtype=torch.long, device=net.device)
              for k, v in channels.items()}

    acts, chans, traj = [], {k: [] for k in channels}, []
    v = net.initial_state(1)
    with torch.no_grad():
        for k in range(N_CTRL):
            ang = 2 * np.pi * k / N_CTRL
            bp = [3.2 * np.cos(ang), 3.2 * np.sin(ang), 0.6]
            sim.mj_data.qpos[ball_qadr:ball_qadr + 3] = bp
            sim.mj_data.qpos[ball_qadr + 3:ball_qadr + 7] = [1, 0, 0, 0]
            mj.mj_forward(sim.mj_model, sim.mj_data)
            ball = sim.mj_data.xpos[ball_bid]
            inten = {}
            for s in ("l", "r"):
                p = sim.mj_data.cam_xpos[cam[s]]
                r = sim.mj_data.cam_xmat[cam[s]].reshape(3, 3)
                rel = ball - p
                dist = float(np.linalg.norm(rel))
                d = r.T @ rel / max(dist, 1e-9)
                inten[s] = retina.response(d[None, :],
                                           np.array([angular_radius(dist, BALL_R)]))
            x = rmap.scatter(inten["l"], inten["r"], net.n_nodes) * DRIVE_GAIN
            xt = torch.as_tensor(x, device=net.device).unsqueeze(0)
            for _ in range(10):
                v = net.step(v, xt, 1e-3)
            sim.set_actuator_inputs("fly0", ActuatorType.POSITION,
                                    motor(v)[0].cpu().numpy())
            sim.set_leg_adhesion_states("fly0", np.ones(6))
            for _ in range(100):
                sim.step()
            acts.append(v[0, viz_t].cpu().numpy())
            for kk, idx in chan_t.items():
                chans[kk].append(float(v[0, idx].mean()))
            traj.append([round(float(q), 3) for q in
                         list(sim.mj_data.xpos[thorax_bid]) + list(ball)])
    sim.close()

    A = np.stack(acts)
    print(f"activity {A.shape}  range {A.min():.3f}..{A.max():.3f}")
    lo, hi = np.percentile(A, 1), np.percentile(A, 99)
    Aq = np.clip((A - lo) / max(hi - lo, 1e-9), 0, 1)
    Aq = (Aq * 255).astype(np.uint8)

    payload = {
        "n": int(len(soma_nodes)),
        "frames": int(N_CTRL),
        "pos": base64.b64encode(pos.astype(np.float16).tobytes()).decode(),
        "act": base64.b64encode(Aq.tobytes()).decode(),
        "channels": {k: [round(x, 4) for x in vv] for k, vv in chans.items()},
        "channel_sizes": {k: int(len(v_)) for k, v_ in channels.items()},
        "traj": traj,
        "total_neurons": int(c.n_nodes),
        "total_edges": int(c.n_edges),
        "soma_total": int(has_soma.sum()),
    }
    json.dump(payload, open("cache/viz_payload.json", "w"))
    import os
    print(f"saved -> cache/viz_payload.json "
          f"({os.path.getsize('cache/viz_payload.json')/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
