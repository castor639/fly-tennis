"""Validate + calibrate the analytic visual drive against FlyGym's real rendering.

Sampling: place the ball ALONG a randomly chosen ommatidium's own viewing direction,
so the target is guaranteed to be inside the field of view. Then ask where the render
actually put it. Any consistent gap is a calibration error in the derived directions,
not noise.
"""
import numpy as np
import mujoco as mj

from flygym import Simulation
from flygym.compose import FlatGroundWorld, NeuroMechFly, KinematicPosePreset
from flygym.anatomy import (AxisOrder, JointPreset, Skeleton, ActuatedDOFPreset,
                            ContactBodiesPreset)
from flygym.utils.math import Rotation3D
from flypp.vision import AnalyticRetina, angular_radius

Q = Rotation3D(format="quat", values=[1, 0, 0, 0])
NP = KinematicPosePreset.NEUTRAL
BALL_R = 0.4
BALL_DIST = 2.0


def build_with_ball():
    world = FlatGroundWorld()
    fly = NeuroMechFly(name="fly0")
    sk = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL,
                  axis_order=AxisOrder.ROLL_PITCH_YAW)
    fly.add_joints(sk, neutral_pose=NP)
    a = sk.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(a, actuator_type="position", neutral_input=NP, kp=50)
    fly.add_leg_adhesion(); fly.colorize(); fly.add_vision()
    world.add_fly(fly, [0, 0, 0.7], Q,
                  bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)
    body = world.mjcf_root.worldbody.add_body(name="ball", pos=[0, 0, 5])
    body.add_joint(name="ball_free", type=mj.mjtJoint.mjJNT_FREE)
    body.add_geom(name="ball_geom", type=mj.mjtGeom.mjGEOM_SPHERE,
                  size=[BALL_R, 0, 0], rgba=[0, 0, 0, 1])
    return Simulation(world)


def eye_pose(sim, cam):
    cid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, cam)
    return sim.mj_data.cam_xpos[cid].copy(), sim.mj_data.cam_xmat[cid].reshape(3, 3).copy()


def main(n_trials=60, eye_index=0):
    sim = build_with_ball()
    sim.reset()
    cam = [mj.mj_id2name(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, i)
           for i in range(sim.mj_model.ncam)][eye_index]
    ball_qadr = sim.mj_model.jnt_qposadr[
        mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_JOINT, "ball_free")]
    retina = AnalyticRetina()
    rng = np.random.default_rng(0)

    sim.mj_data.qpos[ball_qadr:ball_qadr + 3] = [0, 0, 500.0]
    mj.mj_forward(sim.mj_model, sim.mj_data)
    pos_eye, rot_eye = eye_pose(sim, cam)   # AFTER mj_forward, or cam_xpos is stale
    base = sim.get_ommatidia_readouts("fly0")[eye_index].sum(axis=1).astype(float)

    rows = []
    for _ in range(n_trials):
        j = int(rng.integers(retina.n_ommatidia))
        d_cam = retina.dirs[j]
        target = pos_eye + rot_eye @ (d_cam * BALL_DIST)
        sim.mj_data.qpos[ball_qadr:ball_qadr + 3] = target
        sim.mj_data.qpos[ball_qadr + 3:ball_qadr + 7] = [1, 0, 0, 0]
        mj.mj_forward(sim.mj_model, sim.mj_data)
        delta = sim.get_ommatidia_readouts("fly0")[eye_index].sum(axis=1).astype(float) - base
        if -delta.min() < 0.3:
            continue
        w = np.clip(-delta, 0, None)
        true_dir = retina.dirs.T @ w / w.sum()
        true_dir /= np.linalg.norm(true_dir)
        sep = np.degrees(np.arccos(np.clip(d_cam @ true_dir, -1, 1)))
        rows.append((d_cam, true_dir, sep))

    sim.close()
    if not rows:
        print("no visible trials"); return

    seps = np.array([r[2] for r in rows])
    D = np.array([r[0] for r in rows])
    T = np.array([r[1] for r in rows])
    print(f"eye camera: {cam}")
    print(f"{len(rows)}/{n_trials} trials visible")
    print(f"  separation predicted vs rendered centroid: "
          f"median {np.median(seps):.2f} deg, p90 {np.percentile(seps,90):.2f}, "
          f"max {seps.max():.2f}")

    # Is the gap a rigid rotation? Kabsch: best R with T ~ R @ D
    H = D.T @ T
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    resid = np.degrees(np.arccos(np.clip(
        np.einsum("ij,nj->ni", R, D) * T, -1, 1).sum(axis=1).clip(-1, 1)))
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"  best-fit rigid rotation: {ang:.2f} deg")
    print(f"  residual after that rotation: median {np.median(resid):.2f} deg, "
          f"p90 {np.percentile(resid,90):.2f}")
    np.save("cache/eye_dir_correction.npy", R)
    print("  saved correction -> cache/eye_dir_correction.npy")


if __name__ == "__main__":
    main()
