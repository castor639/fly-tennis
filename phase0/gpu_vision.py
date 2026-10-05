"""Can MJWarp batch the ommatidia rendering, or is vision still per-world CPU?"""
import time
import numpy as np
from flygym.warp import GPUSimulation
from flygym.compose import FlatGroundWorld, NeuroMechFly, KinematicPosePreset, ActuatorType
from flygym.anatomy import (AxisOrder, JointPreset, Skeleton, ActuatedDOFPreset,
                            ContactBodiesPreset)
from flygym.utils.math import Rotation3D

Q = Rotation3D(format="quat", values=[1, 0, 0, 0])
NP = KinematicPosePreset.NEUTRAL
CONTACT = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD


def build_world(n_flies=2):
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


for nw in (16, 64, 256):
    world, names, n_act = build_world()
    sim = GPUSimulation(world, n_worlds=nw)
    sim.reset()
    ctrl = {nm: np.zeros((nw, n_act)) for nm in names}
    try:
        r = sim.get_ommatidia_readouts(names[0])
        print(f"n_worlds={nw:4d}  ommatidia readout shape = {r.shape}")
    except Exception as e:
        print(f"n_worlds={nw:4d}  ommatidia FAILED: {type(e).__name__}: {str(e)[:110]}")
        sim.close(); continue

    for _ in range(20):
        sim.step()
        for nm in names:
            sim.get_ommatidia_readouts(nm)
    N = 200
    t0 = time.perf_counter()
    for k in range(N):
        for nm in names:
            sim.set_actuator_inputs(nm, ActuatorType.POSITION, ctrl[nm])
        sim.step()
        if k % 100 == 0:
            for nm in names:
                sim.get_ommatidia_readouts(nm)
    dt = time.perf_counter() - t0
    sps = N / dt
    print(f"            with vision@100Hz: {sps:7.1f} steps/s  "
          f"{sps*nw:9.0f} env-steps/s  {sps*nw*sim.timestep:6.2f}x realtime")
    sim.close()
