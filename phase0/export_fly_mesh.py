"""Export the real NeuroMechFly micro-CT body meshes in a standing pose, grouped by
body part so legs can be animated, as compact base64 buffers for Three.js."""
import base64, json, os, re
import numpy as np, mujoco as mj
from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.anatomy import ContactBodiesPreset
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly, PreprogrammedSteps, LocomotionAction, apply_locomotion_action

fly = make_locomotion_fly(name="fly0", add_adhesion=True, colorize=True)
world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]), bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY, add_ground_contact_sensors=False)
sim = Simulation(world)
steps = PreprogrammedSteps(); dofs = fly.get_actuated_jointdofs_order("position")
sim.reset(); apply_locomotion_action(sim, "fly0", LocomotionAction(joint_angles=steps.default_pose_by_dof_order(dofs), adhesion_onoff=np.ones(6, bool)))
sim.warmup(); mj.mj_forward(sim.mj_model, sim.mj_data)
m, d = sim.mj_model, sim.mj_data
tb = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "fly0/c_thorax"); T0 = d.xpos[tb].copy(); R0 = d.xmat[tb].reshape(3, 3).copy()
parts = {}
for g in range(m.ngeom):
    if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH: continue
    bname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
    if not bname.startswith("fly0/"): continue
    seg = bname.split("/", 1)[1]
    mid = m.geom_dataid[g]; v0 = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]; f0 = m.mesh_faceadr[mid]; nf = m.mesh_facenum[mid]
    V = m.mesh_vert[v0:v0 + nv].copy(); F = m.mesh_face[f0:f0 + nf].copy()
    W = (d.geom_xmat[g].reshape(3, 3) @ V.T).T + d.geom_xpos[g]        # world
    L = (R0.T @ (W - T0).T).T                                            # thorax-relative, fly faces +x
    leg = re.match(r"^(lf|lm|lh|rf|rm|rh)_", seg); key = leg.group(1) if leg else ("wing_l" if seg.startswith("l_wing") else "wing_r" if seg.startswith("r_wing") else "body")
    P = parts.setdefault(key, {"v": [], "f": [], "n": 0})
    P["f"].append(F + P["n"]); P["v"].append(L); P["n"] += nv
# leg pivots = coxa body origin (thorax-relative)
pivots = {}
for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
    bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"fly0/{leg}_coxa")
    if bid >= 0: pivots[leg] = (R0.T @ (d.xpos[bid] - T0)).round(4).tolist()
out = {"parts": {}, "pivots": pivots, "ground_z": float(-(T0[2]))}
tot_v = tot_f = 0
for k, P in parts.items():
    V = np.concatenate(P["v"]).astype(np.float32); F = np.concatenate(P["f"]).astype(np.uint32)
    out["parts"][k] = {"v": base64.b64encode(V.tobytes()).decode(), "f": base64.b64encode(F.tobytes()).decode(), "nv": int(len(V)), "nf": int(len(F))}
    tot_v += len(V); tot_f += len(F)
json.dump(out, open("cache/fly_mesh.json", "w"))
print("parts:", {k: (p["nv"], p["nf"]) for k, p in out["parts"].items()})
print(f"total verts {tot_v:,} faces {tot_f:,}  file {os.path.getsize('cache/fly_mesh.json')/1e6:.2f} MB  pivots {list(pivots)}  ground_z {out['ground_z']:.3f}")
sim.close()
