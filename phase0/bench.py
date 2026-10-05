"""Phase 0: does FlyGym really run two flies in one scene, and how fast?

Measures physics-only and physics+vision throughput for 1 and 2 flies.
Everything printed is measured, not estimated.
"""
import os
import time
import numpy as np

from flygym import Simulation
from flygym.compose import (
    FlatGroundWorld, NeuroMechFly, KinematicPosePreset, ActuatorType,
)
from flygym.anatomy import (
    AxisOrder, JointPreset, Skeleton, ActuatedDOFPreset, ContactBodiesPreset,
)
from flygym.utils.math import Rotation3D

SPAWN_ROT = Rotation3D(format="quat", values=[1, 0, 0, 0])
NEUTRAL = KinematicPosePreset.NEUTRAL
CONTACT = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD


def make_fly(name, with_vision):
    fly = NeuroMechFly(name=name)
    skeleton = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL,
                        axis_order=AxisOrder.ROLL_PITCH_YAW)
    fly.add_joints(skeleton, neutral_pose=NEUTRAL)
    actuated = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(actuated, actuator_type="position",
                      neutral_input=NEUTRAL, kp=50)
    fly.add_leg_adhesion()
    fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
    fly.colorize()
    if with_vision:
        fly.add_vision()
    return fly, len(actuated)


def build(n_flies, with_vision):
    world = FlatGroundWorld()
    names, n_act = [], 0
    for i in range(n_flies):
        fly, n_act = make_fly(f"fly{i}", with_vision)
        world.add_fly(fly, [i * 5.0, 0.0, 0.7], SPAWN_ROT,
                      bodysegs_with_ground_contact=CONTACT)
        names.append(f"fly{i}")
    return Simulation(world), names, n_act


def bench(n_flies, n_steps, with_vision, vision_every=1):
    sim, names, n_act = build(n_flies, with_vision)
    sim.reset()
    rng = np.random.default_rng(0)
    ctrl = {nm: np.zeros(n_act) for nm in names}
    for _ in range(50):                      # warm up, keep alloc out of timing
        sim.step()
    t0 = time.perf_counter()
    for k in range(n_steps):
        for nm in names:
            sim.set_actuator_inputs(nm, ActuatorType.POSITION,
                                    ctrl[nm] + rng.normal(0, 0.01, n_act))
            sim.set_leg_adhesion_states(nm, np.ones(6))
        sim.step()
        if with_vision and k % vision_every == 0:
            for nm in names:
                sim.get_ommatidia_readouts(nm)
    dt = time.perf_counter() - t0
    ts = sim.timestep
    sim.close()
    return n_steps / dt, ts, n_act


if __name__ == "__main__":
    print("MUJOCO_GL =", os.environ.get("MUJOCO_GL", "<unset>"))
    print()

    print("--- model compilation ---")
    for n in (1, 2):
        sim, names, n_act = build(n, with_vision=True)
        print(f"  n_flies={n}: flies={names}  nq={sim.mj_model.nq} "
              f"nu={sim.mj_model.nu} nbody={sim.mj_model.nbody} "
              f"ncam={sim.mj_model.ncam}  actuated_dofs/fly={n_act}  "
              f"timestep={sim.timestep}")
        r = sim.get_ommatidia_readouts(names[-1])
        print(f"          ommatidia readout shape for {names[-1]}: {r.shape}")
        sim.close()
    print()

    print("--- throughput ---")
    rows = []
    for n in (1, 2):
        sps, ts, _ = bench(n, 2000, with_vision=False)
        rows.append((f"physics only, {n} fly/flies", sps, sps * ts))
    for n in (1, 2):
        sps, ts, _ = bench(n, 300, with_vision=True, vision_every=1)
        rows.append((f"physics+vision every step, {n} fly/flies", sps, sps * ts))
    for n in (1, 2):
        sps, ts, _ = bench(n, 1000, with_vision=True, vision_every=10)
        rows.append((f"physics+vision every 10 steps, {n} fly/flies", sps, sps * ts))

    for label, sps, rt in rows:
        print(f"  {label:42s} {sps:9.1f} steps/s   {rt:7.3f}x real time")
