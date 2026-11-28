"""From the CSV logs of spinkick.py (or characterize.py spin) to the spin-kick numbers. No ROS, no numpy.
    python3 -I tools/solo/spinkick_analyze.py logs/*.csv [--json out.jsonl]
Per log (times in s after the tool's start; angles in degrees, counted positive in the direction of the spin, so a
clockwise and an anticlockwise run of the same thing give the same numbers when the kick is mirror-symmetric):
  w_cmd        commanded rate of the spin (rad/s, signed)          t_on    first command with |w| >= 2.9
  ball_d0/b0   ball centre before the spin: distance from the robot centre (m), bearing in the body frame (deg)
  t_touch      ball first 6 mm from its resting place              phi_touch  robot turn from t_on to then
  ball_d_touch distance robot centre -> the ball's resting place at t_touch (fspin: how deep in the swept circle)
  t_rel        ball centre first > 0.13 m from the robot centre (clear of the 0.107 m swept circle), for good
  phi_rel      robot turn from t_on to t_rel                       w_rel   measured spin rate at t_rel (rad/s)
  v_launch     ball speed, line fit over t_rel .. t_rel + 0.12 s   v_roll  the same over t_rel + 0.4 .. + 0.8 s
  k_lip        v_launch / (w_rel * 0.082 m): 1 = the ball leaves at the lip's speed
  dir          direction of the launch (field frame)
  d_start      dir - heading at t_on        d_rel   dir - heading at t_rel       d_ball  dir - bearing robot -> resting ball at t_touch (t_on if no touch)
  v_robot      robot centre speed at t_on (a carry test: the carry speed)
The table at the end groups the logs by (test name in the file name, w_cmd): mean and sd of the main numbers.
AIM = the column among d_start / d_rel / d_ball with the smallest sd across start headings, rates and both senses."""
import csv, json, math, sys
R_OUT, LIP_R, W_SPIN = 0.13, 0.082, 2.9
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
deg = lambda a: round(math.degrees(a), 1)

def interp(s, t):
    """value of the series [(t, v)] at t (linear; the end values outside)"""
    if t <= s[0][0]: return s[0][1]
    for a, b in zip(s, s[1:]):
        if t <= b[0]: return a[1] + (b[1] - a[1]) * (t - a[0]) / max(b[0] - a[0], 1e-9)
    return s[-1][1]
def slope(pts):
    """least-squares slope of [(t, v)]"""
    n = len(pts); mt = sum(p[0] for p in pts) / n; mv = sum(p[1] for p in pts) / n
    den = sum((p[0] - mt) ** 2 for p in pts)
    return sum((p[0] - mt) * (p[1] - mv) for p in pts) / den if den > 0 else float("nan")
def fit_v(b, t0, t1):
    """(speed, direction, samples) of the ball from a line fit of x(t), y(t) over t0..t1"""
    w = [r for r in b if t0 <= r[0] <= t1]
    if len(w) < 4: return None, None, len(w)
    vx, vy = slope([(r[0], r[1]) for r in w]), slope([(r[0], r[2]) for r in w])
    return math.hypot(vx, vy), math.atan2(vy, vx), len(w)

def analyze(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k])
    rb = [(f(r, "t"), f(r, "x"), f(r, "y"), f(r, "th")) for r in rows if r["kind"] == "r"]
    b = [(f(r, "t"), f(r, "x"), f(r, "y")) for r in rows if r["kind"] == "b"]
    spin = [r for r in rows if abs(f(r, "cmd_w")) >= W_SPIN]
    out = dict(log=path.split("/")[-1])
    if not spin or len(rb) < 10: return dict(out, error="no spin command or no robot poses")
    t_on, w_cmd = f(spin[0], "t"), f(spin[len(spin) // 2], "cmd_w"); sg = 1.0 if w_cmd > 0 else -1.0
    th = [(rb[0][0], rb[0][3])]
    for r in rb[1:]: th.append((r[0], th[-1][1] + wrap(r[3] - th[-1][1])))                 # unwrapped heading
    rx, ry = [(r[0], r[1]) for r in rb], [(r[0], r[2]) for r in rb]
    w_at = lambda t: (interp(th, t + 0.025) - interp(th, t - 0.025)) / 0.05
    th0 = interp(th, t_on); x0, y0 = interp(rx, t_on), interp(ry, t_on)
    pre = [r for r in rb if t_on - 0.25 <= r[0] <= t_on]
    v_robot = math.hypot(slope([(r[0], r[1]) for r in pre]), slope([(r[0], r[2]) for r in pre])) if len(pre) >= 4 else None
    out.update(w_cmd=w_cmd, t_on=round(t_on, 3), v_robot=round(v_robot, 2) if v_robot is not None else None,
               w_peak=round(max(sg * w_at(r[0]) for r in rb if t_on <= r[0] <= t_on + 1.0), 1))
    b0 = [r for r in b if t_on - 0.5 <= r[0] <= t_on]
    if b0:
        bx, by = sum(r[1] for r in b0) / len(b0), sum(r[2] for r in b0) / len(b0); psi = math.atan2(by - y0, bx - x0)
        out.update(ball_d0=round(math.hypot(bx - x0, by - y0), 4), ball_b0=deg(sg * wrap(psi - th0)))
        t_touch = next((r[0] for r in b if r[0] > t_on and math.hypot(r[1] - bx, r[2] - by) > 0.006), None)
        if t_touch is not None and (v_robot or 0) < 0.15:                                  # a carried ball is moving already
            xt, yt = interp(rx, t_touch), interp(ry, t_touch); psi = math.atan2(by - yt, bx - xt)   # the robot may have come to the ball (fspin)
            out.update(t_touch=round(t_touch - t_on, 3), phi_touch=deg(sg * (interp(th, t_touch) - th0)),
                       ball_d_touch=round(math.hypot(bx - xt, by - yt), 4))
    dist = [math.hypot(r[1] - interp(rx, r[0]), r[2] - interp(ry, r[0])) for r in b]
    k0 = next((i for i in range(len(b)) if b[i][0] >= t_on - 0.5 and dist[i] < R_OUT), None)   # the ball at the robot
    if k0 is None: return dict(out, error="the ball was never seen within 0.13 m of the robot")
    far = [i > k0 and b[i][0] > t_on and dist[i] > R_OUT for i in range(len(b))]
    k = next((i for i in range(len(b)) if far[i] and all(far[i:i + 6])), None)
    if k is None: return dict(out, error="the ball never left the robot")
    t_rel = b[k][0]
    if k > 0 and b[k][0] - b[k - 1][0] > 0.05: out["ball_gap_before_release_s"] = round(b[k][0] - b[k - 1][0], 3)
    v1, d1, n1 = fit_v(b, t_rel, t_rel + 0.12); v2, d2, n2 = fit_v(b, t_rel + 0.4, t_rel + 0.8)
    if v1 is None: return dict(out, error=f"only {n1} ball samples after the release")
    w_rel = sg * w_at(t_rel)
    out.update(t_rel=round(t_rel - t_on, 3), phi_rel=deg(sg * (interp(th, t_rel) - th0)), w_rel=round(w_rel, 1),
               v_launch=round(v1, 2), v_roll=round(v2, 2) if v2 is not None else None,
               k_lip=round(v1 / (w_rel * LIP_R), 2) if w_rel > 1 else None, dir=deg(d1),
               dir_roll_minus_launch=deg(sg * wrap(d2 - d1)) if v2 is not None else None,
               d_start=deg(sg * wrap(d1 - th0)), d_rel=deg(sg * wrap(d1 - interp(th, t_rel))),
               d_ball=deg(sg * wrap(d1 - psi)) if b0 else None)
    return out

def mean_sd(v):
    v = [x for x in v if x is not None]
    if not v: return "-"
    m = sum(v) / len(v); sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
    return f"{m:.2f}+-{sd:.2f}"
def ang_mean_sd(v):
    """mean and sd of angles in degrees (about the first value, so a set near +-180 is not torn apart)"""
    v = [x for x in v if x is not None]
    if not v: return "-"
    u = [v[0] + math.degrees(wrap(math.radians(x - v[0]))) for x in v]
    return mean_sd(u)

if __name__ == "__main__":
    a = sys.argv[1:]; js = None
    if "--json" in a: i = a.index("--json"); js = a[i + 1]; del a[i:i + 2]
    res = [analyze(p) for p in a]
    for r in res: print(json.dumps(r))
    if js: open(js, "w").write("".join(json.dumps(r) + "\n" for r in res))
    groups = {}
    for r in res:
        if "error" not in r: groups.setdefault((r["log"].split("_")[1] if "_" in r["log"] else "", abs(r["w_cmd"])), []).append(r)
    print(f"\n{'test':7s}{'|w|':>5s}{'n':>3s}  {'w_rel':>11s} {'v_launch':>11s} {'v_roll':>11s} {'k_lip':>11s} {'phi_rel':>13s} {'d_start':>13s} {'d_rel':>13s} {'d_ball':>13s}")
    for (name, w), g in sorted(groups.items()):
        col = lambda key: [r.get(key) for r in g]
        print(f"{name:7s}{w:5.0f}{len(g):3d}  {mean_sd(col('w_rel')):>11s} {mean_sd(col('v_launch')):>11s} {mean_sd(col('v_roll')):>11s} {mean_sd(col('k_lip')):>11s} "
              f"{ang_mean_sd(col('phi_rel')):>13s} {ang_mean_sd(col('d_start')):>13s} {ang_mean_sd(col('d_rel')):>13s} {ang_mean_sd(col('d_ball')):>13s}")
