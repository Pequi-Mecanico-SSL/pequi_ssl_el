"""Run a list of characterize.py tests one after the other, with a person handling the ball.
    source tools/solo/env.sh; python3 tools/solo/series.py <team> <plan.jsonl> [--out DIR] [--results FILE] [--from N]
        [--resume] [--max N]
--out defaults to $SOLO_DATA/logs and --results to $SOLO_DATA/results.jsonl (SOLO_DATA default: log/solo in this
repository).
The plan has one JSON object per line:
    {"label": "...", "test": "step", "args": ["fwd", 1.0, -0.72, -0.30, 0, 1.36]}
    {"label": "...", "test": "spin", "at": [0.45, -0.25, 0], "args": [3.0]}          (goes to x, y, heading deg first)
    {"label": "...", "test": "ram", "spot": [-0.05, -0.30], "args": [1.0, 0, 0.67, 0.10]}
Tests without a ball wait (robot still) until no ball is seen inside SOLO_BOX. For a ram the robot goes to its run-up
point and signals:
    one wiggle every 3 s      = please put the ball on the spot straight ahead of the robot
    one full clockwise spin   = ball seen on the spot and still: the ram starts in 2 s
Every result line of characterize.py is appended to --results with the label and the wall-clock time. The series
stops at the first test that aborts.
More plan fields:
    "box", "brake_box", "settle"   per-test SOLO_BOX / SOLO_BRAKE_BOX / SOLO_SETTLE for characterize.py
SOLO_GOTO_BOX (optional): the area the robot may cross on its way to a test's start ("at", a ram's run-up point). A
test's own box is often small and the robot comes from the test before it; without SOLO_GOTO_BOX the way there is
confined to the test's box, and a robot that starts outside it stops at once ("at the edge of the allowed area").
With SOLO_GOTO_BOX the robot is also driven to the start of every step and sweep before the test (they drive there
themselves, but inside their own box), and before the first test of a run to the middle of its box when it has no
start of its own (a goto, a spin without "at"). A start outside SOLO_GOTO_BOX is approached to the nearest point
0.05 m inside it.
    "speeds": [1.0, 1.4, ...]      escalate a step (args[1]) or a spin (args[0]): each value in turn; it stops once the
                                   peak speed (spin: steady rate) rises by less than 4 % over the best before it
    "back": [x, y, dir_deg]        with "speeds": every second run starts here instead (back and forth along a line)
--resume: start after the last test this plan completed (kept in <results>.progress); --from N still works.
--max N: stop after N tests (e.g. the rams one visit of the person handling the ball allows).
SOLO_BRAKE=auto: the early braking assumes 0.7 x the slowest braking measured so far in --results on the same surface
(SOLO_BRAKE_FIRST m/s2, default 0.8, before any measurement; never below 0.6)
(steps commanded at >= 0.8 m/s), within 1.2..4.0 m/s2; 1.5 until there is a measurement.
SIGINT / SIGTERM: the signal only sets a flag (rclpy without its own handlers when SignalHandlerOptions.NO exists); a
running characterize.py gets the signal forwarded and 1.5 s to send its zeros and save (procs.run_child); the series
sends zeros for 0.3 s itself, prints "stopped by SIGINT ..." and exits 130."""
import json, math, os, subprocess, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Twist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from place_ball import Stop, init_rclpy  # noqa: E402  the shared stop handling (no ROS at import)
from procs import run_child              # noqa: E402
class Stopped(Exception): pass

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
DATA = os.environ.get("SOLO_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "log", "solo"))
OUT = opt("--out", os.path.join(DATA, "logs"))
RES = opt("--results", os.path.join(DATA, "results.jsonl"))
FIRST = int(opt("--from", "1"))
MAXN = int(opt("--max", "0"))
RESUME = "--resume" in args
if RESUME: args.remove("--resume")
team, plan_file = args[0], args[1]
plan = [json.loads(l) for l in open(plan_file) if l.strip() and not l.startswith("#")]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
PROG = RES + ".progress"
def progress():
    try: return json.load(open(PROG))
    except (OSError, ValueError): return {}
KEY = os.path.abspath(plan_file)
if RESUME: FIRST = max(FIRST, progress().get(KEY, 0) + 1)
AUTO = os.environ.get("SOLO_BRAKE", "") == "auto"
def brake_for(surface):
    """the braking the early-brake guard may assume on this surface, from the results so far"""
    dec = []
    try:
        for l in open(RES):
            d = json.loads(l)
            if d.get("test") == "step" and d.get("surface", "") == surface and (d.get("cmd_speed") or 0) >= 0.8 and d.get("decel_90_10"):
                dec.append(float(d["decel_90_10"]))
    except (OSError, ValueError): pass
    return round(max(0.6, min(4.0, 0.7 * min(dec))), 2) if dec else float(os.environ.get("SOLO_BRAKE_FIRST", "0.8"))

stop = Stop(); init_rclpy(rclpy); stop.install(); n = Node('series'); pub = n.create_publisher(Twist, os.environ.get('SOLO_CMD_TOPIC', '/cmd_vel'), 10 if os.environ.get('SOLO_CMD_RELIABLE') else qos_profile_sensor_data)
hist = []
def on_ball(m):
    hist.append((time.time(), m.x, m.y)); del hist[:-400]
n.create_subscription(Pose2D, '/vision/poses/ball', on_ball, 20)

def check():
    if stop.sig is not None: raise Stopped()
def turn(w, T):
    t0 = time.time()
    while time.time() - t0 < T:
        check()
        m = Twist(); m.angular.z = float(w); pub.publish(m); rclpy.spin_once(n, timeout_sec=0.03)
def rest(T=0.4):
    t0 = time.time()
    while time.time() - t0 < T:
        check(); pub.publish(Twist()); rclpy.spin_once(n, timeout_sec=0.03)
def wiggle():
    turn(3.0, 0.3); turn(-3.0, 0.6); turn(3.0, 0.3); rest(0.5)
def listen(T):
    """wait without publishing anything (another program may be driving the robot)"""
    t0 = time.time()
    while time.time() - t0 < T: check(); rclpy.spin_once(n, timeout_sec=0.03)
def ball_ready(sx, sy, tol=0.08):
    """the ball within tol of the spot and still (< 1 cm of movement) over the last 1.5 s"""
    now = time.time(); h = [p for p in hist if now - p[0] < 1.5]
    if len(h) < 40 or now - h[-1][0] > 0.2 or h[-1][0] - h[0][0] < 1.2: return False
    xs = [p[1] for p in h]; ys = [p[2] for p in h]
    return max(xs) - min(xs) < 0.01 and max(ys) - min(ys) < 0.01 and math.hypot(xs[-1] - sx, ys[-1] - sy) < tol
def run(*a, env=None):
    check()
    rc, text, how = run_child(["python3", f"{HERE}/characterize.py", team, *[str(x) for x in a], "--out", OUT], lambda: stop.sig, env=env)
    for l in text.splitlines(): (print("    " + l, flush=True) if how != "done" and l.startswith("{") else None)
    if how != "done" and stop.sig is not None: raise Stopped()
    line = [l for l in text.splitlines() if l.startswith("{") or l.startswith("ABORT") or "not in vision" in l or "no ball" in l]
    return line[-1] if line else ((text.strip().splitlines() or ["no output"])[-1])

def goto_env(env):
    """the environment of the drive to a test's start: SOLO_GOTO_BOX as its allowed area when set, else the test's"""
    area = os.environ.get("SOLO_GOTO_BOX", "")
    return dict(env, SOLO_BOX=area) if area else env

def into_goto_box(p, d=0.05):
    """the point of SOLO_GOTO_BOX nearest to p, d inside its edges (the drive there stops at the edge of that box; the
    test then drives the rest of the way to its start inside its own box); None when SOLO_GOTO_BOX is not one rectangle
    and p lies outside it"""
    rs = [[float(v) for v in r.split(",")] for r in os.environ.get("SOLO_GOTO_BOX", "").split(";") if r.strip()]
    if any(r[0] + d <= p[0] <= r[1] - d and r[2] + d <= p[1] <= r[3] - d for r in rs): return p
    if len(rs) != 1: return None
    r = rs[0]
    return [round(min(max(p[0], r[0] + d), r[1] - d), 3), round(min(max(p[1], r[2] + d), r[3] - d), 3), p[2]]

def start_of(e, first):
    """[x, y, heading deg] to drive to before a test when SOLO_GOTO_BOX is set, or None: the start a step or a sweep
    drives to itself (it does that inside its own box, which the robot may be outside of after another test); for
    the first test of a run without a start of its own (a goto, a spin without "at"), the middle of its box"""
    a = e.get("args", [])
    if not os.environ.get("SOLO_GOTO_BOX") or "at" in e or e["test"] == "ram": return None
    if e["test"] == "step" and len(a) >= 6: return into_goto_box([float(a[2]), float(a[3]), float(a[4]) - (0.0 if a[0] == "fwd" else 90.0)])
    if e["test"] == "sweep" and len(a) >= 4: return into_goto_box([float(a[0]), float(a[2]), 0.0])
    if first and e.get("box"):
        r = box_of(e)[0]; return [round((r[0] + r[1]) / 2, 3), round((r[2] + r[3]) / 2, 3), 0.0]
    return None

def env_for(e):
    env = dict(os.environ)
    for k, name in (("box", "SOLO_BOX"), ("brake_box", "SOLO_BRAKE_BOX"), ("settle", "SOLO_SETTLE")):
        if k in e: env[name] = str(e[k])
    if AUTO: env["SOLO_BRAKE"] = str(brake_for(e.get("surface", "")))
    return env
def box_of(e):
    spec = e.get("box") or os.environ.get("SOLO_BOX")
    return [[float(v) for v in r.split(",")] for r in spec.split(";") if r.strip()] if spec else [[-2.0, 2.0, -1.35, 1.35]]
def ball_in_area(e):
    now = time.time(); h = [p for p in hist if now - p[0] < 1.5]
    return any(r[0] - 0.15 <= p[1] <= r[1] + 0.15 and r[2] - 0.15 <= p[2] <= r[3] + 0.15 for p in h for r in box_of(e))
def record(d, label, i, e):
    d.update(label=label, n=i, clock=time.strftime("%H:%M:%S"), surface=e.get("surface", ""))
    open(RES, "a").write(json.dumps(d) + "\n")

def main():
  done = 0
  rest(1.0)
  for i, e in enumerate(plan, 1):
      if i < FIRST: continue
      if MAXN and done >= MAXN: print(f"stopped after {done} tests (--max); continue with --resume", flush=True); break
      done += 1
      label, test, a = e.get("label", ""), e["test"], e.get("args", [])
      env = env_for(e)
      print(f"[{i}/{len(plan)}] {label}: {test} {a}" + (f"  (brake guard {env['SOLO_BRAKE']} m/s2)" if AUTO else ""), flush=True)
      if test == "ram":
          sx, sy = e["spot"]; dirn = float(a[1]) if len(a) > 1 else 0.0; runup = float(a[2]) if len(a) > 2 else 0.75
          ux, uy = math.cos(math.radians(dirn)), math.sin(math.radians(dirn))
          out = run("goto", round(sx - runup * ux, 3), round(sy - runup * uy, 3), dirn, env=goto_env(env))
          if not out.startswith("{"): print("goto failed:", out, flush=True); sys.exit(1)
          wiggle(); t_ask = time.time()
          print(f"    waiting for the ball at ({sx:+.2f}, {sy:+.2f}) ...", flush=True)
          while not ball_ready(sx, sy):
              rest(0.2)
              if time.time() - t_ask > 3.0: wiggle(); t_ask = time.time()
          turn(-3.0, 2.1); rest(2.0)
      else:
          told = False
          while ball_in_area(e):
              if not told: print("    REMOVE THE BALL from the test area ...", flush=True); told = True
              rest(0.3)
          at = e.get("at") or start_of(e, done == 1)
          if at:
              out = run("goto", *at, env=goto_env(env))
              if not out.startswith("{"): print("goto failed:", out, flush=True); sys.exit(1)
      listen(0.3)
      if "speeds" in e:                                   # escalation along a line, back and forth
          prev = None
          for k, sp in enumerate(e["speeds"]):
              aa = list(a); aa[1 if test == "step" else 0] = sp
              if k % 2 == 1 and "back" in e: aa[2:5] = e["back"]
              env = env_for(e)
              out = run(test, *aa, env=env)
              print(f"    v={sp}: " + out, flush=True)
              if not out.startswith("{"): sys.exit(1)
              d = json.loads(out); record(d, f"{label} {sp}", i, e)
              pk = d.get("peak_speed") or d.get("steady_rate") or 0.0
              if prev is not None and pk < 1.04 * prev:
                  print(f"    peak speed {pk} vs {prev} before: no longer rising, escalation stops", flush=True); break
              prev = pk if prev is None else max(prev, pk)
              rest(0.3)
      else:
          out = run(test, *a, env=env)
          print("    " + out, flush=True)
          if not out.startswith("{"): sys.exit(1)
          record(json.loads(out), label, i, e)
      pr = progress(); pr[KEY] = i; json.dump(pr, open(PROG, "w"))
      rest(0.5)
  print("series done", flush=True)

try:
    main()
except Stopped:
    try: Stop.zeros(lambda *a: pub.publish(Twist()), rclpy.ok, lambda dt: rclpy.spin_once(n, timeout_sec=dt), time.monotonic)
    except Exception: pass
    print("stopped by " + Stop.NAMES.get(stop.sig, str(stop.sig)), flush=True); sys.exit(130)

