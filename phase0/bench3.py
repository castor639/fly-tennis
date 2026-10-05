"""Phase 0b (corrected): vision cadence sweep for two flies.

Fix vs first attempt: the EGL renderer initialises lazily on the FIRST
get_ommatidia_readouts call. Warming up physics only left that one-time cost
inside the timed region, which made rarely-rendering configs look slower than
frequently-rendering ones. Warm up the vision path too, and hold the number of
timed physics steps constant across configs.
"""
import os, time
import numpy as np
from flygym import Simulation
from flygym.compose import FlatGroundWorld, NeuroMechFly, KinematicPosePreset, ActuatorType
from flygym.anatomy import (AxisOrder, JointPreset, Skeleton, ActuatedDOFPreset,
                            ContactBodiesPreset)
from flygym.utils.math import Rotation3D

Q = Rotation3D(format="quat", values=[1, 0, 0, 0])
NP = KinematicPosePreset.NEUTRAL
CONTACT = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD
N_STEPS = 4000


def build(n_flies):
    world = FlatGroundWorld()
    names, n_act = [], 0
    for i in range(n_flies):
        f = NeuroMechFly(name=f"fly{i}")
        sk = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL,
                      axis_order=AxisOrder.ROLL_PITCH_YAW)
        f.add_joints(sk, neutral_pose=NP)
        a = sk.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        n_act = len(a)
        f.add_actuators(a, actuator_type="position", neutral_input=NP, kp=50)
        f.add_leg_adhesion(); f.colorize(); f.add_vision()
        world.add_fly(f, [i * 5.0, 0.0, 0.7], Q,
                      bodysegs_with_ground_contact=CONTACT)
        names.append(f"fly{i}")
    return Simulation(world), names, n_act


def bench(n_flies, decim):
    sim, names, n_act = build(n_flies)
    sim.reset()
    rng = np.random.default_rng(0)
    # warm up BOTH paths: physics and the lazily-created EGL renderer
    for _ in range(50):
        sim.step()
        for nm in names:
            sim.get_ommatidia_readouts(nm)
    t0 = time.perf_counter()
    n_renders = 0
    for k in range(N_STEPS):
        for nm in names:
            sim.set_actuator_inputs(nm, ActuatorType.POSITION,
                                    rng.normal(0, 0.01, n_act))
        sim.step()
        if decim and k % decim == 0:
            for nm in names:
                sim.get_ommatidia_readouts(nm)
            n_renders += 1
    dt = time.perf_counter() - t0
    ts = sim.timestep
    sim.close()
    return N_STEPS / dt, ts, n_renders


if __name__ == "__main__":
    print("MUJOCO_GL =", os.environ.get("MUJOCO_GL", "<unset>"))
    print(f"2 flies, {N_STEPS} timed physics steps per config, "
          f"vision warm-up included\n")
    print(f"{'decim':>6} {'vision Hz':>10} {'renders':>8} {'steps/s':>10} "
          f"{'x realtime':>11} {'sim-s/wall-hr':>14}")
    print("-" * 66)
    for decim in (0, 200, 100, 50, 20, 10, 1):
        sps, ts, nr = bench(2, decim)
        vhz = (1 / ts / decim) if decim else 0
        print(f"{decim if decim else 'none':>6} {vhz:>10.0f} {nr:>8} {sps:>10.1f} "
              f"{sps*ts:>11.3f} {sps*ts*3600:>14.0f}")
