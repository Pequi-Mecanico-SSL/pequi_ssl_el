"""Offline simulation of one robot pushing the ball, for developing tools/solo/place_ball.py (plain Python).
    python3 tools/solo/tests/ballsim.py [--n 200] [--seed 1] [--mode auto] [--tol 0.15] [--set name=value] [--json FILE]
Robot: body-frame velocity commands acted on 0.13 s later, ramped at 1.7 m/s2 forward / 1.5 sideways, braked at
3 m/s2, at most 1.5 m/s, 8 rad/s2 and 4 rad/s turning; a small random speed error per run (+-5 %).
Ball: a disc of radius 0.0215 m. Against the flat front (0.14 m wide, 0.09 m from the centre) it takes the front's
forward velocity (no bounce); it drifts sideways on the front (a random rate per contact, up to 0.04 m/s, and the
sideways motion of the robot is not passed on); it falls off the front past 0.075 m. A first contact faster than
0.8 m/s throws it ahead at 1.0-1.2 x the robot's speed. When the front stops pushing it, it leaves at 0.45-0.8 x its
speed, turned by an angle drawn from N(-4.5 deg, 9 deg) (times the speed fraction above 0.15 m/s, so a creep is not
scattered as much), and rolls with deceleration (0.157 + 0.116 v) / f, f drawn per roll in 0.8..1.35 (the distance
varies -20 %..+35 %). The rest of the hull is a circle of 0.09 m that pushes the ball out (restitution 0.3). Walls
0.3 m beyond the lines and the goals' back walls reflect it with 0.3.
Vision: 60 Hz, 0.1 s old. Robot seen in 97 % of frames inside the field, 50 % within 0.15 m of a line, 40 % beyond
the side lines and 18 % beyond the goal lines. Ball seen in 98 % of frames, 85 % within 0.24 m of the robot, 30 %
touching the front (front_seen; gaps continue with 0.8), 75 % within 0.1 m of the centre mark. Noise 4 mm (ball, as on the real field), 3 mm and
0.01 rad (robot)."""
import math, os, random, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from place_ball import BallPlacer, FIELD_X, FIELD_Y, GOAL_HW, HOLD, BALL_R, FRONT, ROBOT_R  # noqa: E402

# variations for robustness runs (--set name=value): roll = distance scale of every roll (1.3 = 30 % farther than the
# model on average), k_lo/k_hi = release factor, drift = sideways drift rate on the front (m/s), scatter_mu/sd = exit
# angle (deg), delay = command reaction (s), latency = vision age (s), robot_seen = factor on the robot's detection rate
P = dict(front_seen=0.3, roll=1.0, k_lo=0.45, k_hi=0.8, ball_noise=0.004, drift=0.04, scatter_mu=-4.5, scatter_sd=9.0, delay=0.13, latency=0.1, robot_seen=1.0)
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
DT = 0.005


class Sim:
    def __init__(self, robot, ball, rng, ball_v=(0.0, 0.0), delay=None, latency=None):
        delay = P["delay"] if delay is None else delay; latency = P["latency"] if latency is None else latency
        self.rng = rng; self.x, self.y, self.th = robot; self.vb = [0.0, 0.0]; self.w = 0.0   # robot velocity (body)
        self.bx, self.by = ball; self.bvx, self.bvy = ball_v; self.t = 0.0
        self.delay = delay; self.latency = latency; self.q = []          # delayed commands (t_act, cmd)
        self.cmd = (0.0, 0.0, 0.0); self.err = 1.0 + rng.uniform(-0.05, 0.05)
        self.contact = False; self.drift = 0.0; self.f = self.new_f(0.0); self.first_v = 0.0
        self.hist = []; self.hid = False; self.rhid = False; self.robot_wall = False; self.hull_hits = 0
        self.ball_goal = False; self.out = -9.0

    def new_f(self, v):
        return P["roll"] * self.rng.uniform(0.9 if v < 0.3 else 0.8, 1.35)

    def command(self, c):
        self.q.append((self.t + self.delay, c))

    def robot_v_field(self):
        c, s = math.cos(self.th), math.sin(self.th)
        return c * self.vb[0] - s * self.vb[1], s * self.vb[0] + c * self.vb[1]

    def step(self):
        while self.q and self.q[0][0] <= self.t: self.cmd = self.q.pop(0)[1]
        # robot
        for k, acc in ((0, 1.7), (1, 1.5)):
            want = max(-1.5, min(1.5, self.cmd[k] * self.err)); cur = self.vb[k]
            a = 3.0 if (abs(want) < abs(cur) or want * cur < 0) else acc
            self.vb[k] += max(-a * DT, min(a * DT, want - cur))
        sp = math.hypot(*self.vb)
        if sp > 1.5: self.vb = [v * 1.5 / sp for v in self.vb]
        wt = max(-4.0, min(4.0, self.cmd[2])); self.w += max(-8 * DT, min(8 * DT, wt - self.w))
        vx, vy = self.robot_v_field()
        self.x += vx * DT; self.y += vy * DT; self.th = wrap(self.th + self.w * DT)
        if abs(self.x) > FIELD_X + 0.3 - ROBOT_R or abs(self.y) > FIELD_Y + 0.3 - ROBOT_R: self.robot_wall = True
        self.out = max(self.out, abs(self.x) - FIELD_X, abs(self.y) - FIELD_Y)
        # ball free motion
        v = math.hypot(self.bvx, self.bvy)
        if v > 0:
            dec = (0.157 + 0.116 * v) / self.f * DT
            k = max(0.0, v - dec) / v; self.bvx *= k; self.bvy *= k
        self.bx += self.bvx * DT; self.by += self.bvy * DT
        # contact with the front
        c, s = math.cos(self.th), math.sin(self.th)
        dx, dy = self.bx - self.x, self.by - self.y
        fw, lt = c * dx + s * dy, -s * dx + c * dy
        bvf, bvl = c * self.bvx + s * self.bvy, -s * self.bvx + c * self.bvy
        rvf, rvl = self.vb[0], self.vb[1]
        on = 0.0 < fw < HOLD + 0.002 and abs(lt) < 0.075 and fw > FRONT - 0.03
        if on and (bvf <= rvf + 1e-6 or fw < HOLD):
            if not self.contact:
                self.contact = True; self.drift = self.rng.uniform(-P["drift"], P["drift"]); self.first_v = rvf
                self.v_peak = rvf
            fw = HOLD
            if self.first_v > 0.8 and self.v_peak == self.first_v and rvf > 0.8:
                bvf = rvf * self.rng.uniform(1.0, 1.2); self.v_peak = -1.0      # thrown ahead once
            else:
                bvf = max(bvf, rvf)
            bvl = self.drift * min(1.0, max(0.0, rvf) / 0.2)   # sideways: its own drift while pushed, the robot slides under it
            self.bvx, self.bvy = c * bvf - s * bvl, s * bvf + c * bvl
            self.bx, self.by = self.x + c * fw - s * lt, self.y + s * fw + c * lt
        elif self.contact and (fw > HOLD + 0.004 or abs(lt) >= 0.075 or not on):
            self.contact = False                         # released: speed factor, direction scatter, new roll
            v = math.hypot(self.bvx, self.bvy)
            if v > 0.02:
                k = self.rng.uniform(P["k_lo"], P["k_hi"])
                a = math.radians(self.rng.gauss(P["scatter_mu"], P["scatter_sd"])) * min(1.0, max(0.0, (v - 0.05) / 0.25))
                ca, sa = math.cos(a), math.sin(a)
                self.bvx, self.bvy = k * (ca * self.bvx - sa * self.bvy), k * (sa * self.bvx + ca * self.bvy)
                self.f = self.new_f(v)
        d = math.hypot(dx, dy)
        if not on and d < ROBOT_R + BALL_R and not (0.0 < fw and abs(lt) < 0.075):
            nx, ny = dx / max(d, 1e-6), dy / max(d, 1e-6)
            self.bx, self.by = self.x + nx * (ROBOT_R + BALL_R), self.y + ny * (ROBOT_R + BALL_R)
            rn = vx * nx + vy * ny; bn = self.bvx * nx + self.bvy * ny
            if bn < rn:
                nb = rn + 0.3 * (rn - bn); self.bvx += (nb - bn) * nx; self.bvy += (nb - bn) * ny; self.hull_hits += 1
        # walls and goals
        for ax in (0, 1):
            p = self.bx if ax == 0 else self.by; lim = (FIELD_X if ax == 0 else FIELD_Y) + 0.3 - BALL_R
            if abs(p) > lim:
                if ax == 0: self.bx = math.copysign(lim, p); self.bvx *= -0.3
                else: self.by = math.copysign(lim, p); self.bvy *= -0.3
        if abs(self.bx) > FIELD_X and abs(self.by) < GOAL_HW:
            self.ball_goal = True
            if abs(self.bx) > FIELD_X + 0.18 - BALL_R: self.bx = math.copysign(FIELD_X + 0.18 - BALL_R, self.bx); self.bvx *= -0.3
        self.t += DT
        self.hist.append((self.t, self.x, self.y, self.th, self.bx, self.by, self.contact))
        if len(self.hist) > 200: del self.hist[:-200]

    def seen(self, t_obs):
        """what vision reports now: the state latency s ago, with drops and noise -> (robot or None, ball or None)"""
        h = self.hist[-1]
        for e in reversed(self.hist):
            if e[0] <= t_obs - self.latency: h = e; break
        _, x, y, th, bx, by, cont = h; rng = self.rng
        ax, ay = abs(x), abs(y)
        pr = P["robot_seen"] * (0.18 if ax > FIELD_X else 0.4 if ay > FIELD_Y else 0.5 if (ax > FIELD_X - 0.15 or ay > FIELD_Y - 0.15)  else 0.97)
        r = (x + rng.gauss(0, 0.003), y + rng.gauss(0, 0.003), th + rng.gauss(0, 0.01)) if rng.random() < pr else None
        d = math.hypot(bx - x, by - y)
        if self.hid: pb = min(0.2, P["front_seen"]) if (cont or d < HOLD + 0.01) else 0.2   # a gap goes on with 0.8
        elif cont or d < HOLD + 0.01: pb = P["front_seen"]
        elif d < 0.24: pb = 0.85
        elif math.hypot(bx, by) < 0.1: pb = 0.75
        else: pb = 0.98
        if self.hid and d > 0.24: pb = 0.6
        ok = rng.random() < pb; self.hid = not ok
        b = (bx + rng.gauss(0, P["ball_noise"]), by + rng.gauss(0, P["ball_noise"])) if ok else None
        return r, b


def run(target, robot, ball, rng, ball_v=(0.0, 0.0), T=60.0, **kw):
    sim = Sim(robot, ball, rng, ball_v)
    kw.setdefault("timeout", T)
    pl = BallPlacer(target, **kw)
    newb = None; t_vis = 0.0; t_ctl = 0.0; obs = (None, None); last_r = (None, -1.0); last_b = (None, -1.0); min_d = 9.0
    for _ in range(int((T + 10) / DT)):
        sim.step()
        if sim.t >= t_vis:
            t_vis += 1 / 60.0; r, b = sim.seen(sim.t)
            if r: last_r = (r, sim.t)
            if b: last_b = (b, sim.t); newb = b
        if sim.t >= t_ctl:
            t_ctl += 0.02
            rp = last_r[0] if sim.t - last_r[1] < 0.2 else None
            bp = newb; newb = None                              # new detections only
            vx, vy, w, st = pl.step(sim.t, rp, bp)
            sim.command((vx, vy, w) if rp is not None else (0.0, 0.0, 0.0))
            if pl.finished:
                for _ in range(int(3.0 / DT)): sim.step()      # let everything come to rest
                break
    res = pl.result()
    res["true_ball"] = [round(sim.bx, 3), round(sim.by, 3)]
    res["true_error"] = round(math.hypot(sim.bx - target[0], sim.by - target[1]), 3)
    res["robot_ball_end"] = round(math.hypot(sim.bx - sim.x, sim.by - sim.y), 3)
    res["robot_wall"] = sim.robot_wall; res["ball_goal"] = sim.ball_goal; res["hull_hits"] = sim.hull_hits; res["robot_out"] = round(sim.out, 3)
    res["robot_end"] = [round(sim.x, 3), round(sim.y, 3)]
    return res


def rand_pt(rng, mx=FIELD_X - 0.05, my=FIELD_Y - 0.05):
    while True:
        p = (rng.uniform(-mx, mx), rng.uniform(-my, my))
        if not (abs(p[0]) > FIELD_X - 0.1 and abs(p[1]) < GOAL_HW + 0.1): return p


def classify(r, tol):
    if r["status"].startswith("unreachable"): return r["status"].split(" (")[0]
    if r["status"] != "done": return r["status"].split(" at (")[0]
    if r["true_error"] > tol: return "done but off by more than tol"
    if r["robot_ball_end"] < 0.3 + HOLD - 0.01: return "done but the robot body within 0.3 m of the ball"
    return "ok"


AWKWARD = [   # label, target, ball, robot (x, y, th), ball velocity
    ("short move 0.10 m", (0.6, 0.3), (0.5, 0.3), (0.0, 0.0, 0.0), (0, 0)),
    ("short move 0.15 m sideways", (0.3, -0.45), (0.3, -0.3), (-0.5, 0.2, 1.0), (0, 0)),
    ("ball on the touch line, spot along it", (1.0, -1.45), (-0.5, -1.47), (0.0, 0.0, 0.0), (0, 0)),
    ("ball 0.10 m past the touch line", (0.0, 0.0), (0.5, 1.6), (0.0, 0.0, 0.0), (0, 0)),
    ("ball 0.20 m past the touch line", (0.0, 1.0), (0.5, 1.7), (0.0, 0.0, 0.0), (0, 0)),
    ("ball in the corner", (1.0, 0.5), (2.2, 1.45), (0.0, 0.0, 0.0), (0, 0)),
    ("spot in the corner", (2.2, -1.45), (1.0, -0.5), (0.0, 0.0, 0.0), (0, 0)),
    ("ball behind the goal line beside the goal", (1.5, 0.0), (2.35, 0.7), (0.0, 0.0, 0.0), (0, 0)),
    ("ball in a goal", (0.0, 0.0), (2.33, 0.1), (0.0, 0.0, 0.0), (0, 0)),
    ("ball against a goal frame (post)", (0.0, 0.0), (2.24, 0.43), (0.0, 0.0, 0.0), (0, 0)),
    ("spot in the defense area", (-2.0, 0.3), (-1.0, 0.0), (0.0, 0.0, 0.0), (0, 0)),
    ("spot at the goal mouth", (-2.15, 0.0), (-1.2, 0.5), (0.0, 0.0, 0.0), (0, 0)),
    ("ball in the defense area, spot outside", (0.0, 0.0), (2.05, -0.3), (0.0, 0.0, 0.0), (0, 0)),
    ("spot outside the field", (2.4, 0.0), (0.0, 0.0), (-1.0, 0.0, 0.0), (0, 0)),
    ("ball beyond the margin", (0.0, 0.0), (0.0, 1.78), (0.0, 0.0, 0.0), (0, 0)),
    ("moving ball at the start", (0.0, 0.0), (-1.0, 0.5), (0.5, -0.5, 0.0), (0.6, -0.2)),
    ("spot on the centre mark", (0.0, 0.0), (1.2, -0.8), (-1.0, 0.5, 0.0), (0, 0)),
    ("ball already on the spot", (0.5, 0.5), (0.51, 0.5), (0.3, 0.5, 0.0), (0, 0)),
    ("robot touching the ball, wrong side", (-1.0, 0.0), (0.0, 0.0), (-HOLD - 0.01, 0.0, 0.0), (0, 0)),
    ("long move 3.5 m", (-1.9, -1.2), (1.6, 1.0), (2.0, 1.3, 3.1), (0, 0)),
]


def main():
    a = sys.argv[1:]
    def opt(n, d):
        if n in a: i = a.index(n); v = a[i + 1]; del a[i:i + 2]; return v
        return d
    N = int(opt("--n", "200")); seed = int(opt("--seed", "1")); mode = opt("--mode", "auto"); tol = float(opt("--tol", "0.15"))
    out = opt("--json", None)
    while "--set" in a:
        i = a.index("--set"); k, v = a[i + 1].split("="); P[k] = float(v); del a[i:i + 2]
    rng = random.Random(seed); rows = []
    for i in range(N):
        tgt = rand_pt(rng); b = rand_pt(rng); rb = rand_pt(rng)
        while math.hypot(rb[0] - b[0], rb[1] - b[1]) < 0.3: rb = rand_pt(rng)
        r = run(tgt, (rb[0], rb[1], rng.uniform(-math.pi, math.pi)), b, rng, mode=mode, tol=tol)
        r.update(case="random", dist=round(math.hypot(tgt[0] - b[0], tgt[1] - b[1]), 3), cls=classify(r, tol)); rows.append(r)
    for lab, tgt, b, rb, bv in AWKWARD:
        r = run(tgt, rb, b, rng, ball_v=bv, mode=mode, tol=tol)
        r.update(case=lab, dist=round(math.hypot(tgt[0] - b[0], tgt[1] - b[1]), 3), cls=classify(r, tol)); rows.append(r)
    report(rows, tol)
    if out:
        with open(out, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")


def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else float("nan")


def report(rows, tol):
    R = [r for r in rows if r["case"] == "random"]
    tried = [r for r in R if not r["status"].startswith("unreachable")]
    e = [r["true_error"] for r in tried]
    print(f"random cases: {len(R)} ({len(R) - len(tried)} reported unreachable)")
    if tried:
        print(f"  within 0.05 m: {sum(x <= 0.05 for x in e) / len(tried):.1%}   within 0.10 m: {sum(x <= 0.10 for x in e) / len(tried):.1%}"
              f"   ok (status done, within tol, robot backed off): {sum(r['cls'] == 'ok' for r in tried) / len(tried):.1%}")
        print("  final error m  p10 %.3f  p50 %.3f  p90 %.3f  max %.3f" % (q(e, .1), q(e, .5), q(e, .9), max(e)))
        ts = [r["time"] for r in tried]; ps = [r["pushes"] for r in tried]
        print("  time s         p10 %.1f  p50 %.1f  p90 %.1f  max %.1f" % (q(ts, .1), q(ts, .5), q(ts, .9), max(ts)))
        print("  pushes         p10 %d  p50 %d  p90 %d  max %d" % (q(ps, .1), q(ps, .5), q(ps, .9), max(ps)))
        print("  final gain     p10 %.2f  p50 %.2f  p90 %.2f" % tuple(q([r["gain"] for r in tried], p) for p in (.1, .5, .9)))
        print("  unwanted hull touches: %d runs, robot at a wall: %d, ball into a goal: %d" % (
            sum(r["hull_hits"] > 0 for r in tried), sum(r["robot_wall"] for r in tried), sum(r["ball_goal"] for r in tried)))
        print("  robot centre past the lines, max per run: p90 %.3f  max %.3f m" % (q([r["robot_out"] for r in tried], .9), max(r["robot_out"] for r in tried)))
    cls = {}
    for r in rows: cls.setdefault(r["cls"], []).append(r)
    print("classes (all cases):")
    for k, v in sorted(cls.items(), key=lambda kv: -len(kv[1])): print(f"  {len(v):4d}  {k}")
    print("awkward cases:")
    for r in rows:
        if r["case"] != "random":
            print(f"  {r['case']:42s} {r['status'][:60]:60s} err {r['true_error']:.3f} t {r['time']:5.1f} pushes {r['pushes']}")


if __name__ == "__main__":
    main()
