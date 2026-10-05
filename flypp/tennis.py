"""Fly tennis: two connectome flies on a fly-scaled court, brain in the loop.

Court is a real singles court scaled 1/570 (human 1.7 m -> fly 3 mm):
23.77 x 8.23 m -> 41.7 x 14.4 mm, net 0.914 m -> 1.60 mm, service line 6.40 m -> 11.2 mm.
The ball is drawn at 0.5 mm radius (a real ball would be 0.06 mm, invisible).

What the connectome does: each fly's eyes see the ball through the analytic retina,
the untrained network runs on the measured wiring, and with --steer bump the medulla
small-object cells (BumpDecoder) give the bearing that drives lateral movement along
the baseline. --steer contrast (the DNa01/02 right-left pair) and --steer none are
controls. What is scripted: the racket swing, triggered when the ball is within
reach, and the return velocity.
"""
from __future__ import annotations
import argparse, base64, json, os
import numpy as np
import torch
import pyarrow.feather as feather

from flypp.connectome import Connectome
from flypp.network import ConnectomeNetwork
from flypp.vision import AnalyticRetina, angular_radius
from flypp.loop import RetinotopicMap, retinotopic_azimuth, BumpDecoder
from flypp.train_azimuth import eye_frames

# court geometry (mm). SCALE shrinks a true-proportion court (1/570 of real, 41.7 mm)
# into the fly's visual range: a 1.2 mm ball at 27 mm perturbs 7 of 3,900 photoreceptors
# and the position signal is gone by T3 (measured, phase0/dbg_dn.py). At 3-9 mm the
# small-object detectors respond (Phase 4). So: a fly on a table, not a stadium.
SCALE = 0.36
HALF_L, HALF_W = 41.7 / 2 * SCALE, 14.4 / 2 * SCALE
NET_H, SERVICE = 1.60 * SCALE, 11.2 * SCALE
BALL_R_VIS = 0.6
G = 9.0                      # mm/s^2 at this scale: a 10 mm/s lob crosses the court with a ~2 mm apex
E_GROUND, E_RACKET = 0.68, 0.82
REACH = 1.6                  # swing trigger radius (mm); contact itself is geometric
LAT_SPEED = 16.0             # mm/s
BALL_SPEED = 10.0
RK_R, RK_ARM, RK_H = 0.62, 1.15, 0.95          # racket head radius, arm length from shoulder, height
SWING_T = 0.24
SWING_LEAD = 0.31            # trigger this long before the ball reaches the body: face crosses the line at theta ~ -20 deg
CTRL_HZ = 100
REC_HZ = 25


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Fly:
    def __init__(self, side, eyes):
        self.side = side                    # -1: near baseline (y<0), +1: far
        self.y = side * (HALF_L - 1.0)     # side -1 -> near baseline (y<0), faces +y toward the net
        self.x = 0.0
        self.heading = np.pi / 2 if side < 0 else -np.pi / 2     # faces the net
        self.R = rot_z(self.heading - 0.0)  # body frame: +x forward -> rotate to heading
        self.eyes = eyes
        self.swing = 0.0; self.swinging = False; self.swing_phase = 0.0; self.theta = np.radians(65.0); self.hdg_off = 0.0
        self.hits = 0

    def pos(self):
        return np.array([self.x, self.y, 0.9])

    # racket: head swings on an arc around the right shoulder. theta=+65deg is the ready
    # position (held out right/back), a swing sweeps to -55deg in SWING_T seconds.
    def fwd(self):  return np.array([0.0, 1.0, 0.0]) * (1 if self.side < 0 else -1)
    def right(self): return np.array([1.0, 0.0, 0.0]) * (1 if self.side < 0 else -1)
    def racket(self):
        th = np.radians(65.0 - 120.0 * self.swing_phase) if self.swinging else np.radians(65.0)
        head = self.pos() + 0.35 * self.fwd() + 0.45 * self.right() + RK_ARM * (np.cos(th) * self.fwd() + np.sin(th) * self.right()) + np.array([0, 0, RK_H - 0.9])
        n = np.cos(th) * self.fwd() - np.sin(th) * self.right() * 0.12 + np.array([0, 0, 0.22])     # face nearly square, ~12deg loft
        n /= np.linalg.norm(n)
        return head, n, th

    def eye_dirs_to(self, ball):
        out = {}
        for s in ("l", "r"):
            p_local, R_local = self.eyes[s]
            p = self.pos() + self.R @ p_local
            Rw = self.R @ R_local
            rel = ball - p; d = float(np.linalg.norm(rel))
            out[s] = (Rw.T @ rel / max(d, 1e-9), d)
        return out


def serve(rng, toward_side, server):
    """Serve leaves the server's racket toward the far service box with a real loft."""
    head, n, _ = server.racket()
    tgt = np.array([rng.uniform(-HALF_W * 0.6, HALF_W * 0.6), toward_side * SERVICE * 0.7, 0.0])
    d = tgt - head; dxy = np.hypot(d[0], d[1]); T = dxy / BALL_SPEED
    vz = (0 - head[2] + 0.5 * G * T * T) / T + 0.5
    return head.copy(), np.array([d[0] / T, d[1] / T, vz])


def main(a):
    rng = np.random.default_rng(a.seed)
    c = Connectome.load("cache/connectome_thr5.npz")
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(os.cpu_count() or 4)
    net = ConnectomeNetwork(c, device=dev_)
    if a.shuffle: net.shuffle_edges_(0); print(f"[{a.tag}] CONTROL: edges shuffled (same synapses and signs, random wiring)")
    ck = torch.load(a.ckpt, map_location=net.device, weights_only=False) if a.ckpt else None
    if ck is not None and "readout_idx" in ck:
        dn_idx = torch.as_tensor(np.asarray(ck["readout_idx"]), dtype=torch.long, device=net.device); readout_source = ck.get("readout_source", "?")
    else:
        dn_idx = torch.as_tensor(c.index_of_superclass("descending_neuron"), dtype=torch.long, device=net.device); readout_source = "dn"
    readout = torch.nn.Linear(len(dn_idx), 1).to(net.device); in_scale = 0.02
    if ck is not None:
        net.load_state_dict(ck["net"]); readout.load_state_dict(ck["readout"]); in_scale = ck.get("in_scale", 0.02)
    print(f"[{a.tag}] steering readout over {len(dn_idx)} cells (source: {readout_source})")
    retina = AnalyticRetina()
    rmap = RetinotopicMap(c, "cache/photoreceptor_columns.csv", retina.dirs)
    eyes = eye_frames()
    # the connectome's own view of the ball: population-vector azimuth of the small-object
    # bump in the medulla (Mi1/Tm3/T2/T3), read against a ball-free twin of the network
    phi = retinotopic_azimuth(c, rmap, retina.dirs, eyes); decoder = BumpDecoder(c, phi)
    print(f"[{a.tag}] retinotopic map {rmap.coverage()}; bump decoder over {len(decoder.idx)} medulla cells")

    ann = feather.read_table("data/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                             columns=["bodyId", "somaSide", "somaLocation"]).to_pandas()
    ann = ann.drop_duplicates("bodyId").set_index("bodyId").reindex(c.body_ids)
    side_arr = ann.somaSide.to_numpy()
    def lr(t):
        i = c.index_of_types(t)
        return (torch.as_tensor(i[side_arr[i] == "L"], device=net.device),
                torch.as_tensor(i[side_arr[i] == "R"], device=net.device))
    dna = [lr("DNa02"), lr("DNa01")]

    # viz neurons for fly A's brain
    has = ann.somaLocation.notna().to_numpy()
    viz = np.sort(rng.choice(np.flatnonzero(has), a.n_viz, replace=False))
    pos_v = np.stack(ann.somaLocation.to_numpy()[viz]).astype(np.float32)
    pos_v = (pos_v - pos_v.mean(0)) / np.abs(pos_v - pos_v.mean(0)).max()
    sc = c.superclass[viz].astype(str)
    region = np.where(np.char.startswith(sc, "ol") | (sc == "visual_projection") | (sc == "visual_centrifugal"), 0,
                      np.where(np.char.startswith(sc, "vnc"), 2, 1)).astype(np.uint8)
    viz_t = torch.as_tensor(viz, dtype=torch.long, device=net.device)
    CHAN = {"Small object (T3)": ["T3"], "Pursuit (LC10a)": ["LC10a"], "Looming (LPLC2)": ["LPLC2"],
            "Heading (EPG)": ["EPG"], "Steering (PFL3)": ["PFL3"], "Escape (DNp01)": ["DNp01"]}
    chan_t = {k: torch.as_tensor(np.concatenate([c.index_of_types(t) for t in v]), device=net.device) for k, v in CHAN.items()}
    chan_t["Turn L (DNa L)"] = torch.cat([dna[0][0], dna[1][0]]); chan_t["Turn R (DNa R)"] = torch.cat([dna[0][1], dna[1][1]])

    flies = [Fly(-1, eyes), Fly(+1, eyes)]
    # Calibrate the steering contrast with the ball PRESENT, straight ahead at mid-court.
    # Calibrating against an empty scene left a constant offset once any ball was in
    # view, and both flies drifted to their left and pinned against the edge.
    def drive_for(ball_pos):
        xs = []
        for f in flies:
            d = f.eye_dirs_to(ball_pos)
            il = retina.response(d["l"][0][None, :], np.array([angular_radius(d["l"][1], BALL_R_VIS)]))
            ir = retina.response(d["r"][0][None, :], np.array([angular_radius(d["r"][1], BALL_R_VIS)]))
            xs.append(rmap.scatter(il, ir, c.n_nodes))
        return torch.as_tensor(np.stack(xs), device=net.device)
    blank = torch.as_tensor(rmap.scatter(np.ones(721, np.float32), np.ones(721, np.float32), c.n_nodes), device=net.device)
    with torch.no_grad():
        v = net.initial_state(3)                       # fly A, fly B, ball-free twin
        for _ in range(300): v = net.step(v, blank.expand(3, -1), 1e-3)
        v0 = v.clone()
        base = np.array([sum(float(v[i, r].mean() - v[i, l].mean()) for l, r in dna) for i in range(2)])
        def steer(vv):
            """Trained linear readout over all descending neurons -> +left azimuth estimate."""
            dev = (vv[:, dn_idx] - v0[:, dn_idx]); dev = dev - dev.mean(dim=1, keepdim=True)    # drop the common mode
            return readout(dev / in_scale).squeeze(1).cpu().numpy()
        # Open-loop check IN THIS GEOMETRY: fly A held still, ball swept across the far
        # service line, readout should rise monotonically with the ball's left offset.
        xs_sweep = np.linspace(-6, 6, 9); cs = []; bs = []   # in unscaled court units
        for xb in xs_sweep:
            vv = v.clone(); xb_drive = torch.cat([drive_for(np.array([xb * SCALE, 8.0 * SCALE, 0.9])), blank[None]])
            for _ in range(250): vv = net.step(vv, xb_drive, 1e-3)
            cs.append(float(steer(vv)[0])); bz, ba = decoder(vv[:2], vv[2]); bs.append((np.degrees(bz[0]), ba[0]))
        cs = np.array(cs); slope = np.polyfit(-xs_sweep, cs, 1)[0]
        print(f"[{a.tag}] open-loop sweep, readout vs ball x: {np.round(cs, 3)}  slope(left-offset) {slope:+.4f}  (should be POSITIVE)")
        print(f"[{a.tag}] open-loop sweep, bump azimuth (deg, +left) vs ball x: {[round(b[0]) for b in bs]}  amp {[round(float(b[1]), 2) for b in bs]}  (should FALL with x)")
        a.steer_sign = 1.0

    ball, vel = serve(rng, -1, flies[1])
    dt = 1.0 / CTRL_HZ; decoder.reset(); amp_log = []
    frames, acts, chans, rally, rallies, events = [], [], {k: [] for k in chan_t}, 0, [], []
    point_pause = 0.0; last_hitter = None
    n_steps = int(a.seconds * CTRL_HZ)
    with torch.no_grad():
        for k in range(n_steps):
            t = k * dt
            if k % 500 == 0: print(f"[{a.tag}] t={t:.0f}s hits {[f.hits for f in flies]} rally {rally}", flush=True)
            # --- vision -> network, both flies in one batch ---
            xs = []
            for f in flies:
                d = f.eye_dirs_to(ball)
                il = retina.response(d["l"][0][None, :], np.array([angular_radius(d["l"][1], BALL_R_VIS)]))
                ir = retina.response(d["r"][0][None, :], np.array([angular_radius(d["r"][1], BALL_R_VIS)]))
                xs.append(rmap.scatter(il, ir, c.n_nodes))
            x = torch.cat([torch.as_tensor(np.stack(xs), device=net.device), blank[None]])
            for _ in range(10): v = net.step(v, x, 1e-3)
            bump_az, bump_amp = decoder(v[:2], v[2], dt); amp_log.append(bump_amp.copy())
            contrast = np.array([sum(float(v[i, r].mean() - v[i, l].mean()) for l, r in dna) for i in range(2)]) - base
            est = steer(v)                     # +left azimuth estimate per fly
            if a.steer == "contrast":
                est = -contrast                # DNa01/02 R-L: negative = ball on this fly's left (ipsilateral)
            elif a.steer == "bump":
                est = bump_az                  # radians, +left, from the medulla small-object bump

            # --- steering: readout -> lateral velocity along the baseline ---
            for i, f in enumerate(flies):
                if a.steer == "scripted":
                    # walk to where the ball will cross this baseline (a scripted player, stated on the page)
                    toward = (vel[1] < 0) if f.side < 0 else (vel[1] > 0)
                    if toward and point_pause <= 0:
                        tt = abs(ball[1] - f.y) / max(abs(vel[1]), 1e-6); xt = float(np.clip(ball[0] + vel[0] * tt - 0.06 * (1 if f.side < 0 else -1), -HALF_W - 0.3, HALF_W + 0.3))
                        lat = np.clip((xt - f.x) * 6.0, -LAT_SPEED, LAT_SPEED)
                    else:
                        lat = -f.x * 2.0
                elif a.steer == "none":
                    lat = -f.x * 2.0               # CONTROL: no steering, hold the centre
                elif a.steer == "bump":
                    # walk toward the ball's bearing, full speed beyond ~15 deg; no bump -> back to centre
                    if bump_amp[i] < a.amp_min: lat = -f.x * 3.0
                    else: lat = np.clip(est[i] / 0.26, -1, 1) * LAT_SPEED * (-1 if f.side < 0 else 1)
                else:
                    lat = np.clip(a.gain * est[i], -1, 1) * LAT_SPEED * (-1 if f.side < 0 else 1)
                if point_pause > 0: lat = -f.x * 2.0
                lat = float(np.clip(lat, -LAT_SPEED, LAT_SPEED))
                f.x = float(np.clip(f.x + lat * dt, -HALF_W - 0.4, HALF_W + 0.4))
                f.hdg_off += (np.clip(lat / LAT_SPEED, -1, 1) * 0.55 - f.hdg_off) * min(1, 6 * dt)   # turn body into the walk
                if f.swinging:
                    f.swing_phase += dt / SWING_T
                    if f.swing_phase >= 1.0: f.swinging = False; f.swing_phase = 0.0
                f.swing = f.swing_phase if f.swinging else 0.0
                _, _, f.theta = f.racket()

            # --- ball physics ---
            if point_pause > 0:
                point_pause -= dt
                if point_pause <= 0:
                    srv = rng.choice([0, 1]); ball, vel = serve(rng, flies[1 - srv].side, flies[srv]); rally = 0; last_hitter = srv
            else:
                # swing trigger: ball approaching this fly and about to arrive
                for i, f in enumerate(flies):
                    toward = (vel[1] < 0) if f.side < 0 else (vel[1] > 0)
                    dyn = abs(ball[1] - f.y); tta = dyn / max(abs(vel[1]), 1e-6)
                    if toward and last_hitter != i and not f.swinging and tta < SWING_LEAD and abs(ball[0] + vel[0] * tta - f.x) < REACH:
                        f.swinging = True; f.swing_phase = 0.0
                # racket contact (geometric): ball crosses the swept face within its radius
                for i, f in enumerate(flies):
                    if not f.swinging: continue
                    head, n, _ = f.racket()
                    prev_head = getattr(f, "_ph", head); vr = (head - prev_head) / dt; f._ph = head.copy()
                    rel = ball - head; dist_n = float(rel @ n); in_plane = rel - dist_n * n
                    v_rel = vel - vr
                    if np.linalg.norm(in_plane) < RK_R + BALL_R_VIS * 0.6 and abs(dist_n) < BALL_R_VIS + 0.12 and (v_rel @ n) < 0 and last_hitter != i:
                        # a player angles the face: normal = bisector of the incoming ball and the intended
                        # target, so the (physical) mirror reflection sends the ball where they mean to.
                        tgt = np.array([rng.uniform(-HALF_W * 0.55, HALF_W * 0.55), -f.side * SERVICE * 0.85, 0.0])
                        d = tgt - ball; dxy = np.hypot(d[0], d[1]); T = dxy / (BALL_SPEED * 1.05)
                        vz = (0 - ball[2] + 0.5 * G * T * T) / T + 0.6
                        d_out = np.array([d[0] / T, d[1] / T, vz]); d_hat = d_out / np.linalg.norm(d_out)
                        v_in = v_rel / max(np.linalg.norm(v_rel), 1e-9)
                        n_face = d_hat - v_in; n_face /= np.linalg.norm(n_face)
                        v_rel = v_rel - 2.0 * (v_rel @ n_face) * n_face
                        vel = v_rel * E_RACKET + 1.6 * float(vr @ n_face) * n_face
                        vel = d_hat * min(np.linalg.norm(vel), BALL_SPEED * 1.25) if np.linalg.norm(vel) > BALL_SPEED * 0.7 else d_out
                        ball = head + n * (BALL_R_VIS + 0.13)
                        f.hits += 1; last_hitter = i; rally += 1
                        events.append({"t": round(t, 2), "type": "hit", "fly": i, "rally": rally})
                vel[2] -= G * dt; ball = ball + vel * dt
                if ball[2] < BALL_R_VIS and vel[2] < 0:
                    ball[2] = BALL_R_VIS; vel[2] = -vel[2] * E_GROUND; vel[0] *= 0.96; vel[1] *= 0.96
                    events.append({"t": round(t, 2), "type": "bounce"})
                # net: a vertical plane at y=0 up to NET_H
                if abs(ball[1]) < abs(vel[1]) * dt + BALL_R_VIS * 0.5 and ball[2] < NET_H + BALL_R_VIS * 0.5:
                    vel[1] = -vel[1] * 0.25; vel[0] *= 0.5; vel[2] = min(vel[2], 0.0)
                    events.append({"t": round(t, 2), "type": "net"}); rallies.append(rally); point_pause = 1.4
                if abs(ball[1]) > HALF_L + 2.0 * SCALE + 0.8 or abs(ball[0]) > HALF_W + 2.5 * SCALE + 0.6 or (ball[2] <= BALL_R_VIS + 1e-3 and np.linalg.norm(vel[:2]) < 0.8):
                    events.append({"t": round(t, 2), "type": "miss", "rally": rally}); rallies.append(rally); point_pause = 1.4

            # --- record ---
            if k % (CTRL_HZ // REC_HZ) == 0:
                frames.append([round(float(q), 3) for q in [t, flies[0].x, flies[0].y, flies[0].theta,
                                                             flies[1].x, flies[1].y, flies[1].theta,
                                                             ball[0], ball[1], ball[2], rally, est[0], est[1], flies[0].hdg_off, flies[1].hdg_off]])
                acts.append(v[0, viz_t].cpu().numpy())
                for kk, idx in chan_t.items(): chans[kk].append(round(float(v[0, idx].mean()), 4))
    if rally: rallies.append(rally)
    events = [e for e in events if e['type'] != 'bounce'] + [e for e in events if e['type'] == 'bounce'][:400]
    A = np.stack(acts)
    lo, hi = np.percentile(A, 1), np.percentile(A, 99.5)
    Aq = (np.clip((A - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    summary = {"steer": a.steer, "readout_source": readout_source, "readout_cells": int(len(dn_idx)), "steer_sign": float(a.steer_sign), "points": len(rallies), "rallies": rallies, "mean_rally": float(np.mean(rallies)) if rallies else 0.0,
               "max_rally": int(max(rallies)) if rallies else 0, "hits": [f.hits for f in flies],
               "total_hits": int(sum(f.hits for f in flies))}
    Fa = np.array(frames)
    if amp_log: AL = np.array(amp_log); print(f"[{a.tag}] bump amplitude percentiles 10/50/90: {np.percentile(AL, [10, 50, 90]).round(3)}  frac below amp_min {float((AL < a.amp_min).mean()):.2f}")
    offA = -(Fa[:, 7] - Fa[:, 1]); offB = (Fa[:, 7] - Fa[:, 4])
    print(f"[{a.tag}] steer check: corr(est, ball-left-offset) A={np.corrcoef(Fa[:, 11], offA)[0, 1]:+.3f} "
          f"B={np.corrcoef(Fa[:, 12], offB)[0, 1]:+.3f}  (bump: should be POSITIVE; contrast: NEGATIVE)   contrast A range "
          f"{Fa[:, 11].min():+.3f}..{Fa[:, 11].max():+.3f}   net events {sum(e['type'] == 'net' for e in events)}")
    print(f"[{a.tag}] {a.seconds:.0f}s: points {summary['points']}, hits {summary['total_hits']}, "
          f"mean rally {summary['mean_rally']:.2f}, max rally {summary['max_rally']}")
    out = {"tag": a.tag, "rec_hz": REC_HZ, "frames": frames, "events": events, "summary": summary, "chans": chans,
           "channel_sizes": {k: int(len(v_)) for k, v_ in chan_t.items()},
           "court": {"half_l": HALF_L, "half_w": HALF_W, "net_h": NET_H, "service": SERVICE, "ball_r": BALL_R_VIS, "reach": REACH, "scale": SCALE, "rk_r": RK_R, "rk_arm": RK_ARM, "rk_h": RK_H},
           "n_viz": int(a.n_viz), "pos": base64.b64encode(pos_v.astype(np.float16).tobytes()).decode(),
           "region": base64.b64encode(region.tobytes()).decode(), "act": base64.b64encode(Aq.tobytes()).decode(),
           "total_neurons": int(c.n_nodes), "total_edges": int(c.n_edges), "soma_total": int(has.sum()), "trainable": 45661}
    os.makedirs("cache", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"saved {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cache/stage1.pt"); ap.add_argument("--tag", default="trained")
    ap.add_argument("--seconds", type=float, default=40.0); ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--n-viz", type=int, default=7000, dest="n_viz"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="cache/tennis_trained.json")
    ap.add_argument("--amp-min", type=float, default=0.05, dest="amp_min"); ap.add_argument("--steer", default="connectome", choices=["connectome", "scripted", "contrast", "bump", "none"]); ap.add_argument("--shuffle", action="store_true")
    main(ap.parse_args())
