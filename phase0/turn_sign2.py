"""Turning sign, done carefully: gentle asymmetry, upright check, heading AND path agree."""
import numpy as np, mujoco as mj
from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.anatomy import ContactBodiesPreset
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (HybridTurningController, HybridControllerObservation,
                                         LocomotionAction, PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly)
print(f"{'signal':>14} {'heading chg':>12} {'path bearing':>13} {'up.z min':>9} {'final (x,y)':>14}  verdict")
for d in (+0.15, -0.15, +0.30, -0.30):
    fly = make_locomotion_fly(name="fly0", add_adhesion=True); fly.add_vision()
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY, add_ground_contact_sensors=False)
    sim = Simulation(world); steps = PreprogrammedSteps(); dofs = fly.get_actuated_jointdofs_order("position")
    ctrl = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps, output_dof_order=dofs)
    sim.reset(); ctrl.reset(seed=0)
    apply_locomotion_action(sim, "fly0", LocomotionAction(joint_angles=steps.default_pose_by_dof_order(dofs), adhesion_onoff=np.ones(6, bool)))
    sim.warmup()
    tb = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_BODY, "fly0/c_thorax")
    p0 = sim.mj_data.xpos[tb].copy(); h0 = sim.mj_data.xmat[tb].reshape(3, 3)[:, 0].copy()
    sig = np.array([1.0 + d, 1.0 - d]); upmin = 1.0
    for _ in range(int(2.0 / sim.timestep)):
        apply_locomotion_action(sim, "fly0", ctrl.step(sig, HybridControllerObservation.from_sim(sim, "fly0"))); sim.step()
        upmin = min(upmin, sim.mj_data.xmat[tb].reshape(3, 3)[2, 2])
    p1 = sim.mj_data.xpos[tb]; h1 = sim.mj_data.xmat[tb].reshape(3, 3)[:, 0]
    yaw = np.degrees(np.arctan2(np.cross(h0, h1)[2], np.dot(h0, h1)))
    disp = p1 - p0; bearing = np.degrees(np.arctan2(disp[1], disp[0]))
    agree = np.sign(yaw) == np.sign(bearing)
    verdict = ("LEFT" if yaw > 0 else "RIGHT") + (" (heading & path agree)" if agree else " (DISAGREE - suspect)")
    print(f"[1{d:+.2f}, 1{-d:+.2f}] {yaw:+12.1f} {bearing:+13.1f} {upmin:9.2f} ({p1[0]:+6.1f},{p1[1]:+6.1f})  {verdict}")
    sim.close()
