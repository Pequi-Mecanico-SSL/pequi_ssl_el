"""Capture-hold-brake against a plain ram, from characterize.py ram CSV logs. No ROS.
    python3 -I tools/solo/ram_roll.py LOG.csv [LOG.csv ...]
Per log: t_off = last command with cmd_vx > 0.05 (the brake); v_robot = robot speed along the ram line over the 0.10 s
before the ball first moves 6 mm (contact); t_sep = first ball sample after which it stays more than 0.13 m from the
robot centre for 6 samples; v_roll = ball speed, line fit over t_sep + 0.4 .. t_sep + 0.8 s; ratio = v_roll / v_robot;
hold_s = t_off - t_contact (0.4 = capture-hold-brake, about 0 = plain ram). Contact simulation: ratio 1.00 for a
0.4 s hold, 0.89-0.93 for a brake 0.1 s after contact."""
import csv, json, math, sys

def fit(pts, t0, t1):
    w = [p for p in pts if t0 <= p[0] <= t1]
    if len(w) < 4: return None
    mt = sum(p[0] for p in w) / len(w); den = sum((p[0] - mt) ** 2 for p in w)
    s = lambda k: sum((p[0] - mt) * p[k] for p in w) / den
    return s(1), s(2)

def one(path):
    rows = list(csv.DictReader(open(path))); f = lambda r, k: float(r[k])
    rb = [(f(r, "t"), f(r, "x"), f(r, "y")) for r in rows if r["kind"] == "r"]
    b = [(f(r, "t"), f(r, "x"), f(r, "y")) for r in rows if r["kind"] == "b"]
    on = [f(r, "t") for r in rows if f(r, "cmd_vx") > 0.05]
    if not on or len(b) < 20: return dict(log=path.split("/")[-1], error="no ram command or no ball")
    t_on, t_off = on[0], on[-1]
    b0 = [p for p in b if p[0] <= t_on]
    if not b0: return dict(log=path.split("/")[-1], error="ball not seen before the run")
    bx, by = b0[-1][1], b0[-1][2]
    t_c = next((p[0] for p in b if p[0] > t_on and math.hypot(p[1] - bx, p[2] - by) > 0.006), None)
    if t_c is None: return dict(log=path.split("/")[-1], error="the ball never moved")
    near = lambda t: min(rb, key=lambda r: abs(r[0] - t))
    far = [p[0] > t_c and math.hypot(p[1] - near(p[0])[1], p[2] - near(p[0])[2]) > 0.13 for p in b]
    k = next((i for i in range(len(b)) if far[i] and all(far[i:i + 6])), None)
    vr = fit(rb, t_c - 0.10, t_c)
    out = dict(log=path.split("/")[-1], hold_s=round(t_off - t_c, 3), v_robot=round(math.hypot(*vr), 3) if vr else None)
    if k is None: return dict(out, error="the ball never left the robot")
    t_sep = b[k][0]; v = fit(b, t_sep + 0.4, t_sep + 0.8)
    out.update(t_sep_after_contact=round(t_sep - t_c, 3), v_roll=round(math.hypot(*v), 3) if v else None)
    if v and vr and math.hypot(*vr) >= 0.3: out["ratio"] = round(math.hypot(*v) / math.hypot(*vr), 3)   # a slow "contact" is noise
    return out

if __name__ == "__main__":
    R = [one(p) for p in sys.argv[1:]]
    for r in R: print(json.dumps(r))
    for name, sel in (("hold >= 0.25 s", lambda r: r.get("hold_s", 0) >= 0.25), ("hold < 0.15 s", lambda r: r.get("hold_s", 9) < 0.15)):
        g = [r["ratio"] for r in R if "ratio" in r and sel(r)]
        if g: print(f"{name}: n={len(g)} ratio mean {sum(g) / len(g):.3f} (range {min(g):.3f}-{max(g):.3f})")
