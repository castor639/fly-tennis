import numpy as np, torch
from flypp.connectome import Connectome
from flypp.network import ConnectomeNetwork

c = Connectome.load("cache/connectome_thr5.npz")
print(f"{'alpha_scale':>12} {'t=50ms':>10} {'100ms':>10} {'300ms':>10} "
      f"{'600ms':>10} {'verdict':>12}")
print("-" * 70)
for scale in (0.01, 0.005, 0.002, 0.001):
    net = ConnectomeNetwork(c, alpha_scale=scale)
    drive = c.index_of_superclass("ol_sensory")
    with torch.no_grad():
        x = net.drive(drive, 1.0)
        v = net.initial_state(1)
        marks = {}
        for t in range(600):
            v = net.step(v, x, 1e-3)
            if (t + 1) in (50, 100, 300, 600):
                marks[t + 1] = float(v.abs().max())
    vals = [marks[k] for k in (50, 100, 300, 600)]
    growing = vals[-1] > vals[-2] * 1.5
    verdict = "diverging" if (growing or vals[-1] > 1e3) else "settles"
    print(f"{scale:>12} " + " ".join(f"{x:>10.3f}" for x in vals) + f" {verdict:>12}")
    del net; torch.cuda.empty_cache()
