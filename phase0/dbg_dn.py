import numpy as np, torch
from flypp.connectome import Connectome
from flypp.network import ConnectomeNetwork
from flypp.vision import AnalyticRetina, angular_radius
from flypp.loop import RetinotopicMap
from flypp.train_azimuth import eye_frames, AzimuthTask
from flypp.tennis import Fly
c = Connectome.load("cache/connectome_thr5.npz"); net = ConnectomeNetwork(c)
ck = torch.load("cache/stage1_tennis_ls.pt", map_location=net.device); net.load_state_dict(ck["net"])
retina = AnalyticRetina(); rmap = RetinotopicMap(c, "cache/photoreceptor_columns.csv", retina.dirs); eyes = eye_frames()
task = AzimuthTask(c, retina, rmap, eyes, net.device); f = Fly(-1, eyes)
groups = {"photoreceptors": c.index_of_superclass("ol_sensory"), "T3": c.index_of_types("T3"), "LC10a": c.index_of_types("LC10a"), "DN": c.index_of_superclass("descending_neuron")}
gt = {k: torch.as_tensor(v, device=net.device) for k, v in groups.items()}
def tennis_drive(xb):
    d = f.eye_dirs_to(np.array([xb, 8.0, 1.0]))
    il = retina.response(d["l"][0][None,:], np.array([angular_radius(d["l"][1], 1.2)])); ir = retina.response(d["r"][0][None,:], np.array([angular_radius(d["r"][1], 1.2)]))
    return torch.as_tensor(rmap.scatter(il, ir, c.n_nodes), device=net.device).unsqueeze(0)
def train_drive(az): return task.drive(np.array([az]), dist=np.array([27.0]), ball_r=1.2)
with torch.no_grad():
    v0 = net.initial_state(1)
    for _ in range(300): v0 = net.step(v0, task.blank.unsqueeze(0), 1e-3)
    def settle(x):
        v = v0.clone()
        for _ in range(250): v = net.step(v, x, 1e-3)
        return v
    for name, (xa, xb) in {"tennis x=-6 vs +6": (tennis_drive(-6), tennis_drive(6)), "train az=-12.5 vs +12.5": (train_drive(-0.218), train_drive(0.218))}.items():
        print(f"--- {name} ---   drive diff norm {(xa-xb).norm():.3f}  nonzero drive nodes {int((xa!=0).sum())} / {int((xb!=0).sum())}")
        va, vb = settle(xa), settle(xb)
        for k, t in gt.items():
            da = (va[0,t]-v0[0,t]); db = (vb[0,t]-v0[0,t])
            print(f"   {k:14s} |dev a| {da.norm():8.4f}  |dev b| {db.norm():8.4f}  |a-b| {(da-db).norm():8.4f}")
