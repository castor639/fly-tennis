"""Render the fly walking toward the ball, untrained and trained, as MP4 clips.

Same loop as flypp.track.run, plus a tracking camera behind and above the fly. The
ball gets a bright colour so it reads on video.
"""
import argparse, os
import numpy as np
import mujoco as mj
import torch
import pyarrow.feather as feather

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


def build(ball_xy):
    fly = make_locomotion_fly(name="fly0", add_adhesion=True, colorize=True)
    fly.add_vision()
    cam = fly.add_tracking_camera(name="chase", pos_offset=(-9.0, -4.0, 6.0),
                                  rotation=Rotation3D("xyaxes", (0.85, -0.53, 0, 0.32, 0.5, 0.8)),
                                  fovy=42.0)
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                  add_ground_contact_sensors=False)
    b = world.mjcf_root.worldbody.add_body(name="ball", pos=[ball_xy[0], ball_xy[1], BALL_R])
    b.add_geom(name="ball_geom", type=mj.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0],
               rgba=[0.95, 0.72, 0.25, 1])
    return Simulation(world), fly, cam


def run(ckpt, ball_xy, seconds, gain, out_path, fps=25, res=(480, 720)):
    sim, fly, cam = build(ball_xy)
    renderer = sim.set_renderer([cam], camera_res=res, playback_speed=1.0, output_fps=fps)
    steps = PreprogrammedSteps(); dofs = fly.get_actuated_jointdofs_order("position")
    ctrl = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps, output_dof_order=dofs)
    sim.reset(); ctrl.reset(seed=0)
    apply_locomotion_action(sim, "fly0", LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(dofs), adhesion_onoff=np.ones(6, bool)))
    sim.warmup(); mj.mj_forward(sim.mj_model, sim.mj_data)

    c = Connectome.load("cache/connectome_thr5.npz")
    net = ConnectomeNetwork(c)
    if ckpt: net.load_state_dict(torch.load(ckpt, map_location=net.device)["net"])
    retina = AnalyticRetina(); rmap = RetinotopicMap(c, "cache/photoreceptor_columns.csv", retina.dirs)
    ann = feather.read_table("data/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                             columns=["bodyId", "somaSide"]).to_pandas().drop_duplicates("bodyId").set_index("bodyId")
    side = ann.somaSide.reindex(c.body_ids).to_numpy()
    def lr(t):
        i = c.index_of_types(t)
        return (torch.as_tensor(i[side[i] == "L"], device=net.device), torch.as_tensor(i[side[i] == "R"], device=net.device))
    dna = [lr("DNa02"), lr("DNa01")]
    eye = {s: mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, f"fly0/{s}_eye_cam_camera") for s in ("l", "r")}
    ball_gid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, "ball_geom"); assert ball_gid >= 0

    with torch.no_grad():
        blank = torch.as_tensor(rmap.scatter(np.ones(721, np.float32), np.ones(721, np.float32), c.n_nodes), device=net.device).unsqueeze(0)
        v = net.initial_state(1)
        for _ in range(300): v = net.step(v, blank, 1e-3)
        base = sum(float(v[0, r].mean() - v[0, l].mean()) for l, r in dna)
        for k in range(int(seconds * 100)):
            ball = sim.mj_data.geom_xpos[ball_gid]; inten = {}
            for s in ("l", "r"):
                p = sim.mj_data.cam_xpos[eye[s]]; R = sim.mj_data.cam_xmat[eye[s]].reshape(3, 3)
                rel = ball - p; d_ = float(np.linalg.norm(rel))
                inten[s] = retina.response((R.T @ rel / max(d_, 1e-9))[None, :], np.array([angular_radius(d_, BALL_R)]))
            x = torch.as_tensor(rmap.scatter(inten["l"], inten["r"], c.n_nodes), device=net.device).unsqueeze(0)
            for _ in range(10): v = net.step(v, x, 1e-3)
            contrast = sum(float(v[0, r].mean() - v[0, l].mean()) for l, r in dna) - base
            tr = np.clip(gain * contrast, -0.25, 0.25)
            signal = np.array([1.0 + tr, 1.0 - tr])
            for _ in range(100):
                apply_locomotion_action(sim, "fly0", ctrl.step(signal, HybridControllerObservation.from_sim(sim, "fly0")))
                sim.step(); sim.render_as_needed()
    renderer.save_video(out_path)
    sim.close()
    print(f"saved {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball-az", type=float, default=35.0); ap.add_argument("--ball-dist", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=5.0); ap.add_argument("--gain", type=float, default=2.5)
    a = ap.parse_args()
    ball_xy = [a.ball_dist * np.cos(np.radians(a.ball_az)), a.ball_dist * np.sin(np.radians(a.ball_az))]
    os.makedirs("cache/video", exist_ok=True)
    for tag, ck in (("untrained", None), ("trained", "cache/stage1.pt")):
        run(ck, ball_xy, a.seconds, a.gain, f"cache/video/{tag}.mp4")
