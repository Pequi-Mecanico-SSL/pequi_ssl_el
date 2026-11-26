#!/usr/bin/env python3
"""Check series.py plans and auto_session.py sessions without running anything (numpy only, no ROS).
    python3 tools/solo/sessions/validate.py [--map MAP.npz] [--home x0,x1,y0,y1] FILE.jsonl ...
FILE is a plan (lines with "test") or a session (a first line with "kind": "session"; its "home" applies unless --home
is given). --map defaults to $SOLO_DATA/vision_map/map.npz (the camera coverage map of vision_map.py).
Plans: JSON and argument forms (characterize.py / series.py), start / run / stop points inside the line's box, inside
the home area and 0.25 m inside the field lines (coverage sweeps excepted), coverage along every path (cells with a
detection rate under 0.8 are "weak", cells without evidence "unseen").
Sessions: the steps auto_session.py would refuse at load time (an area outside the home area, a strategy step without
its side, a ball step in a no-ball session), every placement spot (inside the home area, the placer's own check,
coverage, and BallPlacer's plan from where the ball is expected to be before it), every ram (robot path from the
spot placed within 0.15 m, inside the step's box and the home area, coverage along it) and where its ball is expected
to stop: exit ratio 1.0 / 1.2 x the robot's speed, -9 / 0 / +9 deg off the heading, wall restitution 0.3 / 0.5,
deceleration 0.157 + 0.116 v. A problem is printed in capitals; the last line counts them."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SOLO = os.path.dirname(HERE)
sys.path.insert(0, SOLO)
import auto_session as A                   # noqa: E402  (no ROS in it)
import vision_map as V                     # noqa: E402
from place_ball import BallPlacer          # noqa: E402

FX, FY, MARGIN = A.FX, A.FY, 0.25
C0, C1, BALL_R = 0.157, 0.116, 0.0215
problems = 0

def bad(msg):
    global problems
    problems += 1; return msg.upper()

def line_pts(a, b, n=None):
    n = n or max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.05))
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n + 1)]

def coverage(vm, pts):
    """-> text: weak and unseen cells along the points"""
    weak, unseen = set(), set()
    for x, y in pts:
        e, r = vm.evidence(x, y), vm.rate(x, y)
        c = (round(math.floor(x / 0.1) * 0.1 + 0.05, 2), round(math.floor(y / 0.1) * 0.1 + 0.05, 2))
        if e < 6 or r != r: unseen.add(c)
        elif r < 0.8: weak.add(c + (round(r, 2),))
    out = []
    if weak: out.append(f"weak {len(weak)} {sorted(weak)[:3]}")
    if unseen: out.append(f"unseen {len(unseen)} {sorted(unseen)[:3]}")
    return "; ".join(out) or "coverage ok"

def roll_ball(p, v, ang, e):
    """straight roll with the walls and the left goal; -> (x, y, what)"""
    x, y = p; vx, vy = v * math.cos(ang), v * math.sin(ang); dt = 0.004
    WX, WY = FX + 0.30 - BALL_R, FY + 0.30 - BALL_R
    for _ in range(20000):
        s = math.hypot(vx, vy)
        if s < 0.01: break
        d = (C0 + C1 * s) * dt; k = max(0.0, s - d) / s; vx, vy = vx * k, vy * k
        nx, ny = x + vx * dt, y + vy * dt
        if abs(nx) > FX and abs(ny) < A.GOAL_HW and abs(x) <= FX: return nx, ny, "goal"
        if abs(ny) > WY: vy = -e * vy; ny = math.copysign(WY, ny)
        if abs(nx) > WX: vx = -e * vx; nx = math.copysign(WX, nx)
        x, y = nx, ny
    return x, y, "rest"

def ram_rests(spot, v, d):
    out = []
    for ratio in (1.0, 1.2):
        for off in (-9, 0, 9):
            for e in (0.3, 0.5):
                out.append(roll_ball(spot, ratio * v, math.radians(d + off), e) + (ratio, off, e))
    return out

def where(p, home):
    c = A.ball_class(p)[0]
    if c in ("goal", "corner", "frame", "wall"): return c
    if home is not None and not A.in_rect(p, A.ball_area(home)): return "other half" if p[0] > home[1] else "outside home"
    return c

def placer_ok(b, spot):
    pl = BallPlacer(spot); why = pl.check_target() or pl.check_ball(b)
    if why: return why
    plan = pl.make_plan(b)
    return plan if isinstance(plan, str) else None

# ------------------------------------------------------------------ plans
def check_plan(path, vm, home):
    for n, line in enumerate(open(path), 1):
        if not line.strip() or line.lstrip().startswith("#"): continue
        e = json.loads(line); t, a = e["test"], e.get("args", []); box = A.rects(e.get("box", "")) or [A.CHAR_AREA]
        pts, notes = [], []
        if t == "step":
            assert a[0] in ("fwd", "side") and len(a) == 6 and all(isinstance(v, (int, float)) for v in a[1:]), (path, n)
            x0, y0, d, L = a[2:6]; u = (math.cos(math.radians(d)), math.sin(math.radians(d))); stop = L + 0.7
            pts = line_pts((x0, y0), (x0 + stop * u[0], y0 + stop * u[1]))
            v = a[1]; ramp = 1.7; reach = v * v / (2 * ramp); notes.append(f"reaches {v} after {reach:.2f} m, holds {max(0, L - reach):.2f} m")
        elif t == "spin":
            assert len(a) in (1, 2) and len(e["at"]) == 3, (path, n); pts = [tuple(e["at"][:2])]
            pts += [(pts[0][0] + 0.15 * math.cos(k), pts[0][1] + 0.15 * math.sin(k)) for k in range(0, 7)]
        elif t == "sweep":
            assert len(a) == 6 and int(a[4]) >= 1, (path, n); x0, x1, y0, y1 = a[:4]
            for k in range(int(a[4])):
                y = y0 + (y1 - y0) * k / max(int(a[4]) - 1, 1); pts += line_pts((x0, y), (x1, y))
        elif t == "goto":
            assert len(a) == 3, (path, n); pts = [tuple(a[:2])]
        elif t == "ram":
            assert len(a) == 4 and len(e["spot"]) == 2, (path, n)
            v, d, run, past = a; u = (math.cos(math.radians(d)), math.sin(math.radians(d))); sx, sy = e["spot"]
            pts = line_pts((sx - run * u[0], sy - run * u[1]), (sx + (past + 0.6) * u[0], sy + (past + 0.6) * u[1]))
        else:
            raise SystemExit(f"unknown test {t} in {path}:{n}")
        flags = []
        if any(not any(A.in_rect(p, r) for r in box) for p in pts if t != "spin" or p == pts[0]): flags.append(bad("outside its box"))
        if home is not None and not all(A.rect_in(r, home) for r in box): flags.append(bad("box outside the home area"))
        if t != "sweep" and any(abs(p[0]) > FX - MARGIN + 1e-6 or abs(p[1]) > FY - MARGIN + 1e-6 for p in pts): flags.append(bad("within 0.25 m of a line"))
        print(f"  {os.path.basename(path)}:{n} {e.get('label', '')[:36]:36s} {' '.join(flags)} {coverage(vm, pts)}{'; ' + '; '.join(notes) if notes else ''}")

# ------------------------------------------------------------------ sessions
def check_session(path, vm, home_opt):
    head, steps, vs = A.load_session(path, "blue")
    home = A.home_rect(home_opt if home_opt is not None else head.get("home", ""))
    s = A.Session("blue", steps, head, vs, home=home, echo=False)
    print(f"  session {head.get('name', '')}: {len(steps)} steps, home {home}, {'no ball' if s.no_ball else 'ball'}; parking spot {s.park}")
    expect = [s.park] if s.park else []       # where the ball may be before the next placement
    expect += [(-1.0, 1.0), (-1.6, -0.9), (-0.5, 0.0)] if home else []
    last_spot = None
    for st in steps:
        k = st["kind"]; tag = f"  {st.get('n'):>3} {k:8s} {st.get('label', '')[:44]:44s}"
        if st.get("_refused"): print(tag, bad("refused: " + st["_refused"])); continue
        if k == "place":
            spot = (st["x"], st["y"]); flags = []
            if home is not None and not A.in_rect(spot, home): flags.append(bad("spot outside the home area"))
            why = BallPlacer(spot).check_target()
            if why: flags.append(bad(why))
            unreach = [(round(b[0], 2), round(b[1], 2), w) for b in expect for w in [placer_ok(b, spot)] if w]
            if unreach: flags.append(f"placer cannot start from {len(unreach)}/{len(expect)} expected ball spots {unreach[:2]}")
            dist = min(math.hypot(b[0] - spot[0], b[1] - spot[1]) for b in expect) if expect else 0
            print(tag, f"spot ({spot[0]:+.2f}, {spot[1]:+.2f}) {coverage(vm, [spot])}; placed from <= {dist:.2f} m", *flags)
            last_spot = spot; expect = [spot]
        elif k == "test" and st.get("test") == "ram" and last_spot is not None:
            v, d, run, past = st["args"]; u = (math.cos(math.radians(d)), math.sin(math.radians(d)))
            box = A.rects(s.env_for(st).get("SOLO_BOX", "")) or [A.CHAR_AREA]; flags = []
            pts = []
            for ex in (-0.15, 0.0, 0.15):
                for ey in (-0.15, 0.0, 0.15):
                    sp = (last_spot[0] + ex, last_spot[1] + ey)
                    pts += line_pts((sp[0] - run * u[0], sp[1] - run * u[1]), (sp[0] + (past + 0.6) * u[0], sp[1] + (past + 0.6) * u[1]), 20)
            out = [p for p in pts if not any(A.in_rect(p, r) for r in box)]
            if out: flags.append(bad(f"robot path leaves the box for a ball placed 0.15 m off ({len(out)} points, e.g. ({out[0][0]:+.2f}, {out[0][1]:+.2f}))"))
            if home is not None and any(not A.in_rect(p, home) for p in pts): flags.append(bad("robot path leaves the home area"))
            rests = ram_rests(last_spot, v, d)
            kinds = {}
            for x, y, what, *_ in rests:
                w = "goal" if what == "goal" else where((x, y), home); kinds[w] = kinds.get(w, 0) + 1
            xs, ys = [r[0] for r in rests], [r[1] for r in rests]
            bad_k = {w: c for w, c in kinds.items() if w not in ("field", "defense", "outside")}
            if bad_k: flags.append(bad(f"ball may stop: {bad_k}"))
            nxt = next((q for q in steps[steps.index(st) + 1:] if q["kind"] == "place"), None)
            unreach = 0
            if nxt:
                for x, y, what, *_ in rests:
                    if what != "goal" and placer_ok((x, y), (nxt["x"], nxt["y"])): unreach += 1
            print(tag, f"ram {v} m/s dir {d}: {coverage(vm, pts)}; ball rests x {min(xs):+.2f}..{max(xs):+.2f}, y {min(ys):+.2f}..{max(ys):+.2f} "
                       f"({', '.join(f'{w} {c}' for w, c in sorted(kinds.items()))} of {len(rests)}); "
                       f"next placement unreachable from {unreach}/{len(rests)}", *flags)
            expect = [(x, y) for x, y, what, *_ in rests if what != "goal" and not placer_ok((x, y), (nxt["x"], nxt["y"]) if nxt else (x, y))] or [last_spot]
        elif k == "strategy" and st.get("on", True):
            prm = dict(st.get("params") or {}); flags = []
            if home is not None and not st.get("side"): flags.append(bad("no side"))
            if str(prm.get("foe_all", "false")).lower() == "true": flags.append(bad("foe_all true: robots of our colour would count as opponents"))
            print(tag, f"side {st.get('side')} params {prm} {st.get('seconds')} s" + (f", ends when the ball leaves {st['end_ball_out']}" if st.get("end_ball_out") else ""), *flags)
            expect = [(-1.2, 0.0), (-0.8, 0.6), (-0.8, -0.6), (-1.6, 0.9), (-1.6, -0.9)]   # after the strategy: anywhere in the half
        elif k == "series":
            print(tag, f"plan {os.path.basename(A.expand(st['plan'], vs))}: {len(st.get('_boxes') or [])} tests, areas inside the home area")
        elif k in A.MOTION:
            print(tag, f"area {st.get('_boxes')}")

def main():
    args = sys.argv[1:]; mp = os.path.join(os.environ.get("SOLO_DATA", os.path.join(SOLO, "..", "..", "log", "solo")), "vision_map", "map.npz"); home = None
    if "--map" in args: i = args.index("--map"); mp = args[i + 1]; del args[i:i + 2]
    if "--home" in args: i = args.index("--home"); home = args[i + 1]; del args[i:i + 2]
    if not args: sys.exit(__doc__)
    vm = V.load(mp)
    if not vm.ok: sys.exit(f"no coverage map at {mp} (--map or $SOLO_DATA/vision_map/map.npz)")
    for f in args:
        first = next((l for l in open(f) if l.strip() and not l.lstrip().startswith("#")), "{}")
        print(f.split("tools/solo/")[-1])
        if json.loads(first).get("kind") == "session": check_session(f, vm, home)
        else: check_plan(f, vm, A.home_rect(home) if home else None)
    print("problems:", problems)

if __name__ == "__main__":
    main()
