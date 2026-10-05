"""Two things Phase 4 depends on: gradients flow, and the eye reaches the legs."""
import numpy as np
import torch
from flypp.connectome import Connectome
from flypp.network import ConnectomeNetwork

c = Connectome.load("cache/connectome_thr5.npz")
net = ConnectomeNetwork(c)

# ---- 1. end-to-end propagation: photoreceptors -> motor neurons --------------
stages = [("ol_sensory", c.index_of_superclass("ol_sensory")),
          ("visual_projection", c.index_of_superclass("visual_projection")),
          ("cb_intrinsic", c.index_of_superclass("cb_intrinsic")),
          ("descending_neuron", c.index_of_superclass("descending_neuron")),
          ("vnc_intrinsic", c.index_of_superclass("vnc_intrinsic")),
          ("vnc_motor", c.index_of_superclass("vnc_motor"))]
drive = c.index_of_superclass("ol_sensory")
with torch.no_grad():
    x = net.drive(drive, 1.0)
    v = net.run(x, 300, dt=1e-3)
print("--- drive all photoreceptors, mean |V| by stage after 300 ms ---")
for name, idx in stages:
    t = torch.as_tensor(idx, dtype=torch.long, device=net.device)
    print(f"  {name:20s} n={len(idx):6,}  mean|V| = {v[0, t].abs().mean():.5f}"
          f"   max = {v[0, t].max():.4f}")

# specific readouts on the pursuit pathway
print("\n--- specific cell types ---")
for t_ in ("T3", "LC11", "LC10a", "LPLC2", "EPG", "PFL3", "DNa02"):
    idx = c.index_of_types(t_)
    if len(idx) == 0:
        continue
    tt = torch.as_tensor(idx, dtype=torch.long, device=net.device)
    print(f"  {t_:8s} n={len(idx):5,}  mean V = {v[0, tt].mean():+.5f}"
          f"   max = {v[0, tt].max():+.4f}")

# ---- 2. gradients ------------------------------------------------------------
print("\n--- gradient check (this is what Phase 4 needs) ---")
net.zero_grad()
x = net.drive(c.index_of_types("LC4"), 1.0)
read = torch.as_tensor(c.index_of_types("DNa02"), dtype=torch.long, device=net.device)
v = net.initial_state(1)
for _ in range(40):                      # short horizon keeps BPTT memory sane
    v = net.step(v, x, 1e-3)
loss = v[:, read].mean()
loss.backward()
for name, p in net.named_parameters():
    g = p.grad
    if g is None:
        print(f"  {name:12s} grad = None  <-- BROKEN")
    else:
        print(f"  {name:12s} shape {tuple(p.shape)}  "
              f"nonzero {int((g != 0).sum()):,}  |g|max {g.abs().max():.3e}")
print(f"  peak GPU mem for 40-step BPTT: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
