"""Phase 0d: CPU multiprocessing scaling, 2 flies + vision@100Hz per worker.

This is the practical path for vision-driven RL, since MJWarp batches physics
but NOT the ommatidia readout.
"""
import multiprocessing as mp
import os, time
import numpy as np

N_STEPS = 2500
DECIM = 100


def worker(_):
    os.environ["MUJOCO_GL"] = "egl"
    from flygym import Simulation
    from flygym.compose import (FlatGroundWorld, NeuroMechFly,
                                KinematicPosePreset, ActuatorType)
    from flygym.anatomy import (AxisOrder, JointPreset, Skeleton,
                                ActuatedDOFPreset, ContactBodiesPreset)
    from flygym.utils.math import Rotation3D
    Q = Rotation3D(format="quat", values=[1, 0, 0, 0])
    NP = KinematicPosePreset.NEUTRAL
    world = FlatGroundWorld()
    names, n_act = [], 0
    for i in range(2):
        f = NeuroMechFly(name=f"fly{i}")
        sk = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL,
                      axis_order=AxisOrder.ROLL_PITCH_YAW)
        f.add_joints(sk, neutral_pose=NP)
        a = sk.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        n_act = len(a)
        f.add_actuators(a, actuator_type="position", neutral_input=NP, kp=50)
        f.add_leg_adhesion(); f.colorize(); f.add_vision()
        world.add_fly(f, [i * 5.0, 0, 0.7], Q,
                      bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)
        names.append(f"fly{i}")
    sim = Simulation(world)
    sim.reset()
    rng = np.random.default_rng(0)
    for _ in range(50):
        sim.step()
        for nm in names:
            sim.get_ommatidia_readouts(nm)
    t0 = time.perf_counter()
    for k in range(N_STEPS):
        for nm in names:
            sim.set_actuator_inputs(nm, ActuatorType.POSITION,
                                    rng.normal(0, 0.01, n_act))
        sim.step()
        if k % DECIM == 0:
            for nm in names:
                sim.get_ommatidia_readouts(nm)
    dt = time.perf_counter() - t0
    sim.close()
    return N_STEPS / dt


if __name__ == "__main__":
    mp.set_start_method("spawn")
    print(f"CPU workers, 2 flies + vision@{10000//DECIM}Hz each, "
          f"{N_STEPS} steps per worker")
    print(f"{'workers':>8} {'steps/s/worker':>15} {'aggregate steps/s':>19} "
          f"{'control steps/s':>16}")
    print("-" * 64)
    for nw in (1, 2, 4, 6, 8):
        t0 = time.perf_counter()
        with mp.Pool(nw) as pool:
            rates = pool.map(worker, range(nw))
        agg = sum(rates)
        print(f"{nw:>8} {np.mean(rates):>15.1f} {agg:>19.0f} "
              f"{agg/DECIM:>16.1f}")
