"""Phase 0c: MJWarp GPU batched throughput, two flies per world."""
import os, time
import numpy as np
from flygym.warp import GPUSimulation
from flygym.compose import FlatGroundWorld, NeuroMechFly, KinematicPosePreset, ActuatorType
from flygym.anatomy import (AxisOrder, JointPreset, Skeleton, ActuatedDOFPreset,
                            ContactBodiesPreset)
from flygym.utils.math import Rotation3D

Q = Rotation3D(format="quat", values=[1, 0, 0, 0])
NP = KinematicPosePreset.NEUTRAL
CONTACT = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD


def build_world(n_flies):
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
    return world, names, n_act


def bench(n_worlds, n_flies=2, n_steps=1000):
    world, names, n_act = build_world(n_flies)
    sim = GPUSimulation(world, n_worlds=n_worlds)
    sim.reset()
    rng = np.random.default_rng(0)
    ctrl = {nm: rng.normal(0, 0.01, (n_worlds, n_act)) for nm in names}
    for _ in range(100):
        for nm in names:
            sim.set_actuator_inputs(nm, ActuatorType.POSITION, ctrl[nm])
        sim.step()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        for nm in names:
            sim.set_actuator_inputs(nm, ActuatorType.POSITION, ctrl[nm])
        sim.step()
    dt = time.perf_counter() - t0
    ts = sim.timestep
    sim.close()
    return n_steps / dt, ts


if __name__ == "__main__":
    print("MJWarp GPU, 2 flies per world, physics only\n")
    print(f"{'n_worlds':>9} {'steps/s':>10} {'env-steps/s':>13} "
          f"{'x realtime (agg)':>17} {'sim-hr/wall-hr':>15}")
    print("-" * 70)
    for nw in (1, 16, 64, 256, 1024, 4096):
        try:
            sps, ts = bench(nw)
            env_sps = sps * nw
            print(f"{nw:>9} {sps:>10.1f} {env_sps:>13.0f} "
                  f"{env_sps*ts:>17.1f} {env_sps*ts:>15.1f}")
        except Exception as e:
            print(f"{nw:>9}  FAILED: {type(e).__name__}: {str(e)[:80]}")
