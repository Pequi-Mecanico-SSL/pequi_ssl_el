#!/usr/bin/env python3
"""Measure the carries in place_ball.py logs (CSV t,kind,x,y,th,cmd_vx,cmd_vy,cmd_w) and fit the release model.
    uv run --with numpy python tools/solo/tests/fit_release.py LOG.csv ...
A carry ends where the forward command drops to zero with the ball on the front. For each: robot speed at the stop
command (from the poses), the ball speed just after (if seen), where the robot stopped, and how far the ball went
along the heading from the stop command until it rested (from its position on the front at the stop command)."""
import sys, math
import numpy as np


def load(p):
    R, B = [], []
    for l in open(p).readlines()[1:]:
        a = l.strip().split(",")
        if len(a) < 8: continue
        v = [float(a[0])] + [float(x) for x in a[2:]]
        (R if a[1] == "r" else B).append(v)
    return np.array(R), np.array(B)


def carries(p):
    R, B = load(p)
    if len(R) < 50 or len(B) < 50: return []
    out = []
    i = 1
    while i < len(R):
        if R[i - 1, 4] > 0.15 and abs(R[i, 4]) < 1e-6 and abs(R[i, 5]) < 1e-6:
            t = R[i, 0]; th = R[i, 3]; c, s = math.cos(th), math.sin(th)
            pre = (B[:, 0] > t - 0.5) & (B[:, 0] <= t)
            fresh = (B[:, 0] > t - 0.1) & (B[:, 0] <= t)
            if fresh.sum():
                bp = B[fresh][-1]; fw = c * (bp[1] - R[i, 1]) + s * (bp[2] - R[i, 2]); lt = -s * (bp[1] - R[i, 1]) + c * (bp[2] - R[i, 2])
            else:                                   # hidden by the robot: on the front, as the tool assumes
                fw, lt = HOLD, 0.0; bp = [t, R[i, 1] + c * HOLD, R[i, 2] + s * HOLD]
            if not (0.04 < fw < 0.2 and abs(lt) < 0.11): print("skip", p.split("/")[-1], round(t, 2), round(fw, 3), round(lt, 3), file=sys.stderr); i += 1; continue
            w = (R[:, 0] > t - 0.15) & (R[:, 0] <= t)
            k = np.where(w)[0]
            vr = math.hypot(R[k[-1], 1] - R[k[0], 1], R[k[-1], 2] - R[k[0], 2]) / max(1e-3, R[k[-1], 0] - R[k[0], 0])
            # robot stop: first time after t its speed < 0.02 over 0.1 s
            j = i; stop = None
            while j < len(R) - 6:
                if math.hypot(R[j + 6, 1] - R[j, 1], R[j + 6, 2] - R[j, 2]) / max(1e-3, R[j + 6, 0] - R[j, 0]) < 0.03: stop = R[j]; break
                j += 1
            if stop is None: print('no stop', p, t, file=sys.stderr); i += 1; continue
            # ball rest: median over 0.5 s once it moves < 1 cm per 0.5 s, within 6 s
            rest = None; tb = t + 0.5
            while tb < t + 10:
                m = (B[:, 0] > tb) & (B[:, 0] <= tb + 0.6)
                if m.sum() > 8:
                    q = B[m]; h = len(q) // 2
                    if np.hypot(*(np.median(q[h:, 1:3], 0) - np.median(q[:h, 1:3], 0))) < 0.008: rest = (np.median(q[:, 1]), np.median(q[:, 2]), tb); break
                tb += 0.1
            if rest is None: print('no rest', p, t, file=sys.stderr); i += 1; continue
            # ball speed just after leaving the front: ball samples farther than 0.13 m ahead of the robot, 0..0.6 s
            m = (B[:, 0] > t) & (B[:, 0] < t + 0.8)
            vb = None
            q = B[m]
            if len(q) > 5:
                rr = np.array([R[np.searchsorted(R[:, 0], x) - 1] for x in q[:, 0]])
                f = c * (q[:, 1] - rr[:, 1]) + s * (q[:, 2] - rr[:, 2])
                off = q[f > HOLD + 0.03]
                if len(off) > 4:
                    off = off[:12]; A = np.polyfit(off[:, 0], c * off[:, 1] + s * off[:, 2], 1); vb = A[0]
            trav = c * (rest[0] - bp[1]) + s * (rest[1] - bp[2]); side = -s * (rest[0] - bp[1]) + c * (rest[1] - bp[2])
            rstop = c * (stop[1] - R[i, 1]) + s * (stop[2] - R[i, 2])
            out.append(dict(log=p.split("/")[-1], t=round(t, 2), v=round(vr, 3), cmd=round(R[i - 1, 4], 3), vball=None if vb is None else round(vb, 3),
                            robot_stop=round(rstop, 3), travel=round(trav, 3), side=round(side, 3), gap=round(trav - rstop, 3),
                            fw=round(fw, 3), lat=round(lt, 3)))
            i = j
        i += 1
    return out


HOLD = 0.1115
if __name__ == "__main__":
    allc = [c for p in sys.argv[1:] for c in carries(p)]
    for c in allc: print(c)
    if allc:
        v = np.array([c["v"] for c in allc]); d = np.array([c["travel"] for c in allc]); rs = np.array([c["robot_stop"] for c in allc])
        print("n", len(allc))
        print("travel/v fit (d = a v + b v^2):", np.linalg.lstsq(np.c_[v, v * v], d, rcond=None)[0])
        print("robot stop distance/v^2:", np.round(rs / v ** 2, 3), "mean", round(float(np.mean(rs / v ** 2)), 3))
        sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
        from place_ball import roll
        lat = 0.23
        ks = np.linspace(0.3, 1.0, 71)
        err = [np.sum((d - lat * v - np.array([roll(k * x) for x in v])) ** 2) for k in ks]
        k = ks[int(np.argmin(err))]
        print(f"release_k fit (latency {lat}): {k:.2f}, rms {math.sqrt(min(err) / len(v)):.3f} m;",
              "old model (k 0.9) rms", round(math.sqrt(np.mean((d - lat * v - np.array([roll(0.9 * x) for x in v])) ** 2)), 3))
        for c in allc: print(f"  v {c['v']:.3f}  measured {c['travel']:.3f}  old {lat * c['v'] + roll(0.9 * c['v']):.3f}  new {lat * c['v'] + roll(k * c['v']):.3f}")
