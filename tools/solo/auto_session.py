#!/usr/bin/env python3
"""Run a list of field tests with our robot and the ball alone on the field, nobody next to it.

OPERATOR NOTE
  Before: vision connector, recorders and the field page (hq.py, locked) running; the strategy stack started with
  run_real.sh and switched OFF (so $SOLO_RUN_DIR/solo.cmd holds the node's command line); the ball anywhere inside
  the field; no other robot on the field; SOLO_NOTIFY_CMD set to an executable called as `CMD "message" "title"`.
  Start (after `source tools/solo/env.sh`):
      python3 tools/solo/auto_session.py <team> tools/solo/sessions/short_10.jsonl --budget 10
      python3 tools/solo/auto_session.py <team> SESSION --dry-run       # prints every command and the decision table
  Options: --budget MIN (default: the session's budget_min), --var NAME=VALUE (e.g. P, the folder of the series plans;
  OUT, the folder of the logs),
  --hq URL (field page, default http://127.0.0.1:8700), --ball-lost S (5), --robot-lost S (2), --robot-wait S (10),
  --busy-wait S (20), --realert S (180).
      python3 tools/solo/auto_session.py <team> SESSION --sim corner     # offline: fakes for every step, simulated clock
                                    (scenarios: ok, corner, goal, stuck, timeout; log in $SOLO_DATA/auto/*_sim_*.jsonl)
  Stop: Ctrl-C once (or SIGTERM) = finish the current step, strategy off, end marker, exit. Twice = also stop the
  running step now: its process group gets SIGINT, after 1.5 s SIGTERM, after 1.5 s more SIGKILL. place_ball.py
  sends zeros for 0.3 s on SIGINT and SIGTERM; after SIGKILL nothing is sent, and the robot's own command timeout
  (0.3 s) stops it. The budget ends the session the same way as one Ctrl-C.
  What the notification says (title "field session"), and what the person does:
    "ball in the +x goal at (x, y): put it anywhere inside the field"            -> put the ball back inside the lines
    "ball in the (+x, -y) corner at ..." / "ball against the +y wall at ..."      -> the same
    "ball not seen since 12:01:02 (last at (x, y)): put a ball anywhere inside the field"
    "robot stuck against the +y wall at (x, y): free it and put it inside the field" -> lift it 0.5 m into the field
    "robot not seen since 12:01:02, last at (x, y): ..."                          -> check the robot / its tag
    "another program commands the robot (solo_node): the session waits"          -> stop that program
    "step 7 (ram 1.5 >) failed: ABORT: ...; the session goes on"                  -> nothing (steps with on_fail alert)
    "session ended: budget used (12 steps done, 2 skipped)"                       -> collect the robot
  The session goes on by itself as soon as the ball is seen resting inside the field again / the robot is seen
  0.25 m clear of every wall. Reminders every --realert s, at most 3 messages per incident.
  Log: $SOLO_DATA/auto/<YYYYmmdd_HHMMSS>_session.jsonl (every step's command, times, exit status, result line, the
  world after it and every decision); markers go to $SOLO_DATA/events_<YYYYmmdd>.jsonl (one line per marker: its
  text, and "data": "use" when the recordings from there on are test data, "skip" when they are not).

SESSION FILE: JSON lines, one step per line ('#' lines are comments). An optional first line
    {"kind": "session", "name": "...", "vars": {"P": "${SESSION_DIR}/plans"}, "env": {"SOLO_BRAKE": "auto"}}
sets variables (${NAME} in any string; TEAM, T (this folder), SOLO_DATA, SESSION_DIR (the session file's folder) and
the environment are always there; --var overrides) and the environment of every step.
Steps (fields common to all: label, needs_ball, on_fail = skip | retry | stop | alert (default skip), timeout s,
group (a failed step skips the rest of its group), env {}, final (run at the end even when the budget is used;
marker / cmd / wait only), keep_ball_out "x0,x1,y0,y1" (the ball must be 0.15 m clear of this box before the step;
with park [x, y] the ball is placed there first)):
    {"kind": "place", "x": -0.85, "y": 0.65, "opts": ["--carry", "0.35"]}               place_ball.py; needs_ball
    {"kind": "test", "test": "ram", "args": [1.2, 0, 1.0, 0.25], "out": "${OUT}/logs", "tee": "${OUT}/rams.log",
     "env": {"SOLO_RAMP": "1.7", "SOLO_BOX": "-2.05,2.05,0.25,1.05"}}                   characterize.py
    {"kind": "series", "plan": "${P}/steps_ramped.jsonl", "from": 1, "max": 8, "args": ["--out", "${OUT}/logs"]}  series.py
    {"kind": "strategy", "on": true, "seconds": 10}       start the strategy node, wait, stop it ("on": false = stop)
    {"kind": "goto", "x": 1.2, "y": -0.8, "v": 0.4}                                      goto.py
    {"kind": "marker", "text": "short session start", "data": "use"}                    a line in the events file
    {"kind": "wait", "seconds": 5}
    {"kind": "cmd", "argv": ["${T}/robot_stack.sh", "save"], "final": true}              any command, no motion
needs_ball defaults to true for place, false otherwise.

DECISIONS (after every step the world is read from the field page's /state: robot pose, ball, their ages):
  ball inside the field (or in a defense area, or past a line within the placer's margin) -> ball steps run
  ball in a goal / against a goal frame / in a corner / against a wall / not seen > --ball-lost s:
      the placer is asked (offline, BallPlacer) whether it can push the ball toward the nearest point 0.3 m inside the
      lines; if so, one place_ball.py run; otherwise (or if that fails) one alert; the steps with needs_ball false go
      on; ball steps resume when a ball is seen resting inside the field; nothing ball-free left: wait, re-alert
  place_ball "unreachable: ball ..." -> the same as above, the place step stays at the head of the ball steps
  place_ball "failed: robot stuck ..." or a step that aborted / timed out with the robot within 0.10 m of a wall or
      a goal frame -> one slow goto 0.5 m inside the field, straight away from it; moved < 0.1 m: motion stops,
      alert, wait until the robot is seen clear
  place_ball failed otherwise (timeout, left the area) -> once more with SOLO_BOX 0.15 m wider and --carry 0.25;
      a second failure follows on_fail
  a test's ABORT line or a non-zero exit -> on_fail
  robot not seen > --robot-lost s before a motion step -> wait --robot-wait s, then alert and wait
  another commanding program running (procs.COMMANDERS: a python process whose script is solo_node, characterize.py,
      series.py, ram_series.py, place_ball.py, goto.py, nudge.py or frame_check.py; this process, its ancestors and
      the step it runs itself excluded; a command line that only mentions a name does not count), or the field page
      driving (a click target, a placement) -> no motion step starts, and no placement the supervisor adds (parking,
      rescue); after --busy-wait s an alert, and the session waits
  budget used / Ctrl-C -> the current step ends, strategy off, final steps, end marker, exit

Tests (no robot, no ROS): uv run --with pytest pytest -q tools/solo/tests/test_auto_session.py
"""
import json, math, os, re, signal, subprocess, sys, threading, time, urllib.request
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import procs as solo_procs                                 # noqa: E402  the commanding programs, the node's command line
from procs import is_instance, is_launcher                # noqa: E402,F401
REPO = os.path.dirname(os.path.dirname(HERE))
FX, FY = 2.25, 1.5                  # half field to the lines (m)
WALL = 0.30                         # boundary wall beyond the lines
GOAL_HW, GOAL_D = 0.40, 0.18        # goal half width, depth behind the goal line
DA_D, DA_HW = 0.5, 0.675            # defense area depth, half width
ROBOT_R = 0.09
PLACER_MARGIN = 0.25                # place_ball.py --margin default: how far past the lines it may push from
NEAR_WALL = 0.10                    # robot body within this of a wall / frame = "against it"
CLEAR_WALL = 0.25                   # body clearance that ends a stuck incident
FREE_IN = 0.5                       # the freeing goto ends this far inside the lines
COMMANDERS = solo_procs.COMMANDERS
MOTION = ("place", "test", "series", "goto", "strategy")
is_motion = lambda st: st["kind"] in MOTION and not (st["kind"] == "strategy" and not st.get("on", True))
TIMEOUTS = dict(place=90, test=60, series=900, goto=35, cmd=120, marker=5, wait=0)
clip = lambda v, lo, hi: lo if v < lo else hi if v > hi else v
sgn = lambda v, a, b: a if v >= 0 else b


# ------------------------------------------------------------------ geometry
def _seg(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0; L2 = dx * dx + dy * dy
    k = clip(((px - x0) * dx + (py - y0) * dy) / L2, 0.0, 1.0) if L2 else 0.0
    return math.hypot(px - x0 - k * dx, py - y0 - k * dy)

def obstacles(x, y):
    """[(clearance of the robot's body in m, name)] to the four boundary walls and the two goal frames, nearest first"""
    out = [(FX + WALL - x - ROBOT_R, "+x wall"), (x + FX + WALL - ROBOT_R, "-x wall"),
           (FY + WALL - y - ROBOT_R, "+y wall"), (y + FY + WALL - ROBOT_R, "-y wall")]
    for s, name in ((1, "+x goal frame"), (-1, "-x goal frame")):
        segs = [(s * FX, GOAL_HW, s * (FX + GOAL_D), GOAL_HW), (s * FX, -GOAL_HW, s * (FX + GOAL_D), -GOAL_HW),
                (s * (FX + GOAL_D), -GOAL_HW, s * (FX + GOAL_D), GOAL_HW)]
        out.append((min(_seg(x, y, *g) for g in segs) - ROBOT_R, name))
    return sorted(out)

def free_point(x, y):
    """the nearest point FREE_IN inside the lines: straight away from the wall or frame the robot is against"""
    return clip(x, -(FX - FREE_IN), FX - FREE_IN), clip(y, -(FY - FREE_IN), FY - FREE_IN)

def ball_class(b, margin=PLACER_MARGIN):
    """-> (class, where): field / defense / outside (placeable) or goal / frame / corner / wall / lost"""
    if b is None: return "lost", "ball not seen"
    x, y = b; ax, ay = abs(x), abs(y); sx, sy = sgn(x, "+x", "-x"), sgn(y, "+y", "-y"); at = f"at ({x:+.2f}, {y:+.2f})"
    if ax > FX and ay < GOAL_HW + 0.02: return "goal", f"ball in the {sx} goal {at}"
    if ax > FX - 0.03 and ay < GOAL_HW + 0.07: return "frame", f"ball against the {sx} goal frame {at}"
    if ax > FX and ay > FY: return "corner", f"ball in the ({sx}, {sy}) corner {at}"
    if ax > FX + margin: return "wall", f"ball against the {sx} wall {at}"
    if ay > FY + margin: return "wall", f"ball against the {sy} wall {at}"
    if ax > FX or ay > FY: return "outside", f"ball past the lines {at}"
    if ax > FX - DA_D and ay < DA_HW: return "defense", f"ball in the {sx} defense area {at}"
    return "field", f"ball {at}"

PLACEABLE = ("field", "defense", "outside")
IN_FIELD = ("field", "defense")

def reach_point(b):
    """the nearest point 0.3 m inside the lines, outside the defense areas"""
    x, y = clip(b[0], -(FX - 0.3), FX - 0.3), clip(b[1], -(FY - 0.3), FY - 0.3)
    if abs(x) > FX - DA_D - 0.1 and abs(y) < DA_HW + 0.1: x = math.copysign(FX - DA_D - 0.1, x)
    return round(x, 2), round(y, 2)

def placer_can(b, target):
    """ask BallPlacer (no ROS in it) whether it can push the ball from b toward target; None = cannot tell"""
    try:
        if HERE not in sys.path: sys.path.insert(0, HERE)
        from place_ball import BallPlacer
        p = BallPlacer(target)
        why = p.check_target() or p.check_ball(b)
        if why: return False
        plan = p.make_plan(b)
        return not isinstance(plan, str)
    except Exception:
        return None

def rects(spec):
    return [[float(v) for v in r.split(",")] for r in spec.split(";") if r.strip()] if spec else []

def widen(spec, d=0.15):
    """a SOLO_BOX spec d wider on every side, kept 0.12 m off the walls"""
    lx, ly = FX + WALL - 0.12, FY + WALL - 0.12
    rs = rects(spec) or [[-FX - PLACER_MARGIN, FX + PLACER_MARGIN, -FY - PLACER_MARGIN, FY + PLACER_MARGIN]]
    return ";".join(",".join(f"{v:.2f}" for v in (max(-lx, r[0] - d), min(lx, r[1] + d), max(-ly, r[2] - d), min(ly, r[3] + d))) for r in rs)

def ball_in_box(b, spec, pad=0.15):
    return b is not None and any(r[0] - pad <= b[0] <= r[1] + pad and r[2] - pad <= b[1] <= r[3] + pad for r in rects(spec))


# ------------------------------------------------------------------ processes (the checks are in procs.py)
class Procs:
    """the process table from /proc"""
    def table(self):
        return solo_procs.process_table()
    def mine(self):
        """this process and its ancestors"""
        return solo_procs.ancestors()
    def group(self, pgid):
        """the pids of the process group pgid (a step's process and what it started)"""
        out = set()
        for pid, _ in self.table():
            try:
                with open(f"/proc/{pid}/stat") as f:
                    if int(f.read().rsplit(")", 1)[1].split()[3]) == pgid: out.add(pid)
            except (OSError, ValueError, IndexError): pass
        return out

def busy(procs, names=COMMANDERS, exclude=()):
    """the commanding programs that run now, other than this process, its ancestors and the pids in exclude"""
    return solo_procs.running(names, procs.table(), set(procs.mine()) | set(exclude))


class Strategy:
    """switch the strategy node off and on the way hq.py does: SIGINT it (its command line is kept in
    $SOLO_RUN_DIR/solo.cmd) and start it again from that line; the policy server keeps running"""
    def __init__(self, procs, run_dir=None):
        self.procs = procs; self.run = run_dir or os.environ.get("SOLO_RUN_DIR", "/tmp/solo_run")
        self.cmdf = os.path.join(self.run, "solo.cmd"); self.child = None
    def pids(self):
        return [pid for pid, a in self.procs.table() if is_instance(a, "solo_node") or is_launcher(a)]
    def running(self):
        return any(is_instance(a, "solo_node") for _, a in self.procs.table())
    def can_start(self):
        return self.running() or os.path.exists(self.cmdf)
    def off(self):
        for pid, a in self.procs.table():
            if is_instance(a, "solo_node") and len(a) > 1 and os.path.basename(a[1]) == "solo_node":
                try:
                    os.makedirs(self.run, exist_ok=True)
                    with open(self.cmdf, "w") as f: json.dump(a, f)
                except OSError: pass
        for sig, wait in ((signal.SIGINT, 2.5), (signal.SIGTERM, 1.5)):
            for pid in self.pids():
                try: os.kill(pid, sig)
                except OSError: pass
            t0 = time.time()
            while self.pids() and time.time() - t0 < wait:
                if self.child: self.child.poll()
                time.sleep(0.1)
        if self.child: self.child.poll()
        return not self.running()
    def on(self):
        """start the node from the saved line, only when procs.check_node_cmd accepts it; else self.why says why"""
        self.why = ""
        if self.running(): return True
        try:
            with open(self.cmdf) as f: argv = json.load(f)
        except (OSError, ValueError) as e: self.why = f"no saved command line ({type(e).__name__})"; return False
        why = solo_procs.check_node_cmd(argv)
        if why: self.why = "saved command line refused: " + why; return False
        try:
            log = open(os.path.join(self.run, "solo.log"), "a")
            self.child = subprocess.Popen(argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e: self.why = f"cannot start: {e}"; return False
        try:
            with open(os.path.join(self.run, "solo.pid"), "w") as f: f.write(str(self.child.pid))
        except OSError: pass
        return True


# ------------------------------------------------------------------ the world
class PageWorld:
    """robot and ball from the field page's /state, polled in the background (4 Hz); ages kept here. A plain /state
    read (without ?page=1) does not refresh the page's dead man's switch: only the browser page keeps a target alive"""
    def __init__(self, url, clock=time.time):
        self.url = url.rstrip("/") + "/state"; self.clock = clock; self.L = threading.Lock()
        self.robot = None; self.robot_t = -1e9; self.ball = None; self.ball_t = -1e9; self.page_t = -1e9
        self.hist = deque(maxlen=40); self.strategy = None; self.page_driving = False; self.halt = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()
    def stop(self, wait=2.0):
        """end the polling thread"""
        self.halt.set(); self.thread.join(wait)
    def _loop(self):
        while not self.halt.is_set():
            try:
                d = json.loads(urllib.request.urlopen(self.url, timeout=1.0).read())
                t = self.clock()
                with self.L:
                    self.page_t = t; self.strategy = d.get("strategy")
                    self.page_driving = d.get("target") is not None or bool((d.get("place") or {}).get("active"))
                    if d.get("us") is not None and d.get("us_age") is not None:
                        self.robot = tuple(d["us"]); self.robot_t = t - float(d["us_age"])
                    if d.get("ball") is not None:
                        self.ball = tuple(d["ball"]); self.ball_t = t; self.hist.append((t, *self.ball))
            except Exception:
                pass
            self.halt.wait(0.25)
    def snapshot(self):
        t = self.clock()
        with self.L:
            h = [p for p in self.hist if t - p[0] < 1.5]
            still = len(h) >= 4 and h[-1][0] - h[0][0] > 1.0 and max(math.hypot(p[1] - h[-1][1], p[2] - h[-1][2]) for p in h) < 0.02
            return dict(robot=self.robot, robot_age=round(t - self.robot_t, 2), ball=self.ball, ball_age=round(t - self.ball_t, 2),
                        ball_still=still, page_age=round(t - self.page_t, 2), page_driving=self.page_driving)


class ScriptedWorld:
    """for tests: robot and ball set by hand or by a timeline [(t, {"robot": .., "ball": ..})] on a simulated clock;
    None hides an object (its age then grows)"""
    def __init__(self, clock, robot=(0.0, 0.6, 0.0), ball=(1.2, -0.8), timeline=()):
        self.clock = clock; self.robot = robot; self.ball = ball; self.robot_t = self.ball_t = clock()
        self.robot_on = robot is not None; self.ball_on = ball is not None; self.timeline = sorted(timeline, key=lambda e: e[0])
    def set(self, **kw):
        t = self.clock()
        if "robot" in kw:
            self.robot_on = kw["robot"] is not None
            if self.robot_on: self.robot = tuple(kw["robot"]); self.robot_t = t
        if "ball" in kw:
            self.ball_on = kw["ball"] is not None
            if self.ball_on: self.ball = tuple(kw["ball"]); self.ball_t = t
    def snapshot(self):
        while self.timeline and self.timeline[0][0] <= self.clock(): self.set(**self.timeline.pop(0)[1])
        t = self.clock()
        if self.robot_on: self.robot_t = t
        if self.ball_on: self.ball_t = t
        return dict(robot=self.robot, robot_age=round(t - self.robot_t, 2), ball=self.ball, ball_age=round(t - self.ball_t, 2),
                    ball_still=self.ball_on, page_age=0.0)


# ------------------------------------------------------------------ running a step
def result_of(lines):
    """the result line: the last JSON line, else the last ABORT / failure line, else the last line"""
    for l in reversed(lines):
        if l.startswith("{"):
            try: return json.loads(l), l
            except ValueError: pass
    keys = ("ABORT", "refusing", "failed", "not in vision", "no ball", "arrived:", "timeout:", "vision lost", "series done", "stopped after", "Error")
    for l in reversed(lines):
        if any(k in l for k in keys): return None, l
    return None, (lines[-1] if lines else "")

class Runner:
    """each step a subprocess in its own process group with a hard time limit"""
    def __init__(self, echo=True):
        self.echo = echo; self.p = None; self.kill_now = False
    def run(self, argv, env, timeout, tee=None):
        t0 = time.time(); lines = []; self.kill_now = False
        try:
            p = self.p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                          text=True, bufsize=1, start_new_session=True)
        except OSError as e:
            return dict(rc=127, timed_out=False, line=f"cannot start: {e}", data=None, t0=t0, t1=time.time(), tail=[])
        def read():
            for l in p.stdout:
                l = l.rstrip("\n"); lines.append(l)
                if self.echo: print("    | " + l, flush=True)
        th = threading.Thread(target=read, daemon=True); th.start()
        timed_out = False
        while p.poll() is None:
            if time.time() - t0 > timeout or self.kill_now:
                timed_out = True; self.stop(p); break
            time.sleep(0.1)
        th.join(2.0); self.p = None
        if tee:
            try:
                with open(tee, "a") as f: f.write("\n".join(lines) + "\n")
            except OSError: pass
        data, line = result_of(lines)
        return dict(rc=p.returncode, timed_out=timed_out, line=line, data=data, t0=t0, t1=time.time(), tail=lines[-5:])
    def stop(self, p):
        """the step's process group: SIGINT, 1.5 s to end; SIGTERM, 1.5 s; then SIGKILL. place_ball.py sends zeros for
        0.3 s on SIGINT and on SIGTERM; after SIGKILL nothing is sent, and the robot's own command timeout stops it.
        Then 0.5 s for that timeout"""
        for sig, wait in ((signal.SIGINT, 1.5), (signal.SIGTERM, 1.5), (signal.SIGKILL, 1.0)):
            try: os.killpg(p.pid, sig)
            except OSError: break
            try: p.wait(wait); break
            except subprocess.TimeoutExpired: pass
        time.sleep(0.5)
    def abort(self):
        self.kill_now = True
    def child_pid(self):
        """the pid (and process group) of the running step, or None"""
        p = self.p
        return p.pid if p is not None and p.poll() is None else None


class Notifier:
    def __init__(self, cmd=None):
        self.cmd = cmd if cmd is not None else os.environ.get("SOLO_NOTIFY_CMD", "")
    def send(self, msg, title):
        if not self.cmd:
            print(f"\n*** ALERT (SOLO_NOTIFY_CMD not set): {title}: {msg}\n", flush=True); return "not set"
        try:
            r = subprocess.run([self.cmd, msg, title], capture_output=True, text=True, timeout=30)
            return f"rc {r.returncode}"
        except Exception as e:
            return f"error {e}"


# ------------------------------------------------------------------ the session
def load_session(path, team, extra_vars=None):
    head, steps = {}, []
    for k, l in enumerate(open(path), 1):
        l = l.strip()
        if not l or l.startswith("#"): continue
        d = json.loads(l)
        if d.get("kind") == "session": head = d; continue
        d.setdefault("n", len(steps) + 1); d["line"] = k; steps.append(d)
    vs = dict(os.environ); vs.update(TEAM=team, T=HERE, SOLO_DATA=data_dir(), SESSION_DIR=os.path.dirname(os.path.abspath(path)))
    for k, v in head.get("vars", {}).items(): vs[k] = expand(v, vs)
    vs.update(extra_vars or {})
    return head, steps, vs

def data_dir():
    return os.environ.get("SOLO_DATA") or os.path.join(REPO, "log", "solo")

def expand(v, vs):
    if isinstance(v, str): return re.sub(r"\$\{(\w+)\}", lambda m: str(vs.get(m.group(1), m.group(0))), v)
    if isinstance(v, list): return [expand(x, vs) for x in v]
    if isinstance(v, dict): return {k: expand(x, vs) for k, x in v.items()}
    return v


class Session:
    def __init__(self, team, steps, head=None, vs=None, world=None, runner=None, notifier=None, procs=None, strategy=None,
                 log_path=None, events_path=None, budget_s=600.0, clock=time.time, sleep=time.sleep, python="python3",
                 ball_lost=5.0, robot_lost=2.0, robot_wait=10.0, busy_wait=20.0, realert=180.0, max_alerts=3, echo=True):
        self.team = team; self.steps = steps; self.head = head or {}; self.vs = vs or dict(TEAM=team, T=HERE, SOLO_DATA=data_dir())
        self.world = world; self.runner = runner; self.notifier = notifier; self.procs = procs; self.strat = strategy
        self.clock, self.sleep = clock, sleep; self.py = python; self.echo = echo
        self.ball_lost, self.robot_lost, self.robot_wait, self.busy_wait = ball_lost, robot_lost, robot_wait, busy_wait
        self.realert, self.max_alerts = realert, max_alerts
        self.t_start = clock(); self.t_end = self.t_start + budget_s
        self.log_path = log_path; self.events_path = events_path
        self.stop_req = 0; self.stop_why = None; self.motion_halted = False
        self.ball_incident = None; self.skipped_groups = set(); self.alerts = []; self.done = []; self.skipped = []
        self.retried_place = set(); self.bad_ball = None    # where the placer said "unreachable: ball"

    # ---- records
    def log(self, kind, **kw):
        d = dict(t=round(self.clock(), 3), clock=time.strftime("%H:%M:%S", time.localtime(self.clock())), kind=kind, **kw)
        if self.echo: print(f"[{d['clock']}] {kind}: " + json.dumps(kw)[:300], flush=True)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a") as f: f.write(json.dumps(d) + "\n")
        return d
    def alert(self, msg, title="field session"):
        self.alerts.append(msg); r = self.notifier.send(msg, title) if self.notifier else "no notifier"
        self.log("alert", message=msg, title=title, sent=r)
    def marker(self, text, data):
        """a line in the events file: data "use" (the recordings from here on are test data) or "skip" (they are not)"""
        line = dict(to=None, state=text, data=data, source="auto_session")
        line = {"from": time.strftime("%H:%M:%S", time.localtime(self.clock())), **line}
        if self.events_path:
            with open(self.events_path, "a") as f: f.write(json.dumps(line) + "\n")
        self.log("marker", text=text, data=data)

    # ---- time and stop
    def out_of_time(self):
        return self.stop_req > 0 or self.clock() >= self.t_end
    def request_stop(self, *_):
        self.stop_req += 1
        if self.stop_req == 1: self.stop_why = self.stop_why or "Ctrl-C"; print("\nstopping after the current step (Ctrl-C again: now)", flush=True)
        elif self.runner: self.runner.abort()

    # ---- the world
    def look(self, why=""):
        s = self.world.snapshot()
        b = s["ball"] if s["ball_age"] <= self.ball_lost else None
        s["ball_class"], s["ball_where"] = ball_class(b)
        if s["robot"] is not None:
            c, name = obstacles(*s["robot"][:2])[0]; s["robot_clear"], s["robot_near"] = round(c, 3), name
        return s
    def wait_ball_rest(self, s, T=6.0):
        """a ball seen but rolling, or briefly hidden: up to T s for it to rest or to count as lost"""
        t0 = self.clock()
        while self.clock() - t0 < T and not self.out_of_time():
            if s["ball_class"] == "lost" or (s["ball_still"] and s["ball_age"] < 1.0): return s
            self.sleep(0.25); s = self.look()
        return s

    # ---- the ball
    def at_bad(self, s):
        return self.bad_ball is not None and s["ball"] is not None and math.hypot(s["ball"][0] - self.bad_ball[0], s["ball"][1] - self.bad_ball[1]) < 0.05
    def ball_ready(self, s):
        return s["ball_class"] in PLACEABLE and not self.at_bad(s)
    def ball_back(self, s):
        return s["ball_class"] in IN_FIELD and s["ball_still"] and not self.at_bad(s)
    def ball_problem(self, s):
        """the ball cannot be used: try one placement toward the field if the placer can, else alert once"""
        s = self.wait_ball_rest(s)
        if self.ball_ready(s):
            if self.ball_incident: self.log("ball_back", where=s["ball_where"]); self.ball_incident = None
            self.bad_ball = None
            return True
        if self.ball_incident is None:
            self.ball_incident = dict(cls=s["ball_class"], since=self.clock(), alerts=0, tried=self.at_bad(s))
            self.log("ball_unusable", cls=s["ball_class"], where=s["ball_where"])
        inc = self.ball_incident
        if not inc["tried"] and s["ball_class"] != "lost" and not self.motion_halted and self.robot_seen(s):
            inc["tried"] = True
            tgt = reach_point(s["ball"]); can = placer_can(s["ball"], tgt)
            self.log("ball_rescue", target=tgt, placer_can=can)
            if can and not self.busy_now():
                self.side_place(tgt[0], tgt[1], "ball rescue")
                s = self.wait_ball_rest(self.look())
                if self.ball_ready(s): self.log("ball_back", where=s["ball_where"]); self.ball_incident = None; return True
        if inc["alerts"] == 0: self.ball_alert(s)
        return False
    def ball_alert(self, s):
        inc = self.ball_incident; inc["alerts"] += 1; inc["t_alert"] = self.clock()
        if s["ball_class"] == "lost":
            last = self.world.snapshot()["ball"]
            since = time.strftime("%H:%M:%S", time.localtime(self.clock() - s["ball_age"]))
            msg = f"ball not seen since {since}" + (f" (last at ({last[0]:+.2f}, {last[1]:+.2f}))" if last else "") + ": put a ball anywhere inside the field"
        elif self.at_bad(s):
            msg = f"{s['ball_where']}: the robot cannot push it from there: put it anywhere inside the field"
        else:
            msg = f"{s['ball_where']}: put it anywhere inside the field"
        if inc["alerts"] > 1: msg += f" (reminder {inc['alerts']}/{self.max_alerts})"
        self.alert(msg)
    def wait_for_ball(self):
        """nothing ball-free left: wait for a ball resting inside the field, re-alerting"""
        self.log("waiting", for_="ball")
        while not self.out_of_time():
            s = self.look()
            if self.ball_back(s): self.log("ball_back", where=s["ball_where"]); self.ball_incident = None; self.bad_ball = None; return True
            inc = self.ball_incident
            if inc and inc["alerts"] < self.max_alerts and self.clock() - inc.get("t_alert", 0) >= self.realert: self.ball_alert(s)
            self.sleep(1.0)
        return False

    # ---- the robot
    def hold(self, what, msg, cleared):
        """stop motion, alert, wait until cleared(snapshot) (re-alerts), resume"""
        self.motion_halted = True; self.alert(msg); n = 1; ta = self.clock()
        while not self.out_of_time():
            s = self.look()
            if cleared(s): self.log("hold_cleared", what=what); self.motion_halted = False; return True
            if n < self.max_alerts and self.clock() - ta >= self.realert:
                n += 1; ta = self.clock(); self.alert(msg + f" (reminder {n}/{self.max_alerts})")
            self.sleep(1.0)
        return False
    def robot_seen(self, s):
        return s["robot"] is not None and s["robot_age"] <= self.robot_lost
    def ensure_robot(self):
        s = self.look(); t0 = self.clock()
        while not self.robot_seen(s) and self.clock() - t0 < self.robot_wait and not self.out_of_time():
            self.sleep(0.5); s = self.look()
        if self.robot_seen(s): return True
        since = time.strftime("%H:%M:%S", time.localtime(self.clock() - s["robot_age"])) if s["robot"] else "the session started"
        last = f", last at ({s['robot'][0]:+.2f}, {s['robot'][1]:+.2f})" if s["robot"] else ""
        if s["robot"] and s["robot_clear"] < NEAR_WALL: last += f" against the {s['robot_near']}"
        if s.get("page_age", 0) > 3: msg = f"the field page does not answer (robot last seen {since}{last}): the session waits; restart hq.py"
        else: msg = f"robot not seen since {since}{last}: check it is on the field with its tag up"
        return self.hold("robot", msg, self.robot_seen)
    def robot_clear(self, s):
        return self.robot_seen(s) and s["robot_clear"] > CLEAR_WALL and abs(s["robot"][0]) < FX + 0.05 and abs(s["robot"][1]) < FY + 0.05
    def free_robot(self, s, why):
        """one slow goto FREE_IN inside the field, straight away from the wall; not moved 0.1 m: alert and wait"""
        x, y = s["robot"][:2]; tgt = free_point(x, y); name = s["robot_near"]
        self.log("robot_stuck", at=[round(x, 3), round(y, 3)], near=name, clearance=s["robot_clear"], why=why, free_to=tgt)
        st = dict(kind="goto", x=tgt[0], y=tgt[1], v=0.25, label="free the robot", on_fail="skip", n=0, timeout=30)
        if not self.busy_now(): self.run_step(st)
        s2 = self.look()
        if self.robot_seen(s2) and math.hypot(s2["robot"][0] - x, s2["robot"][1] - y) >= 0.1:
            self.log("robot_freed", at=[round(v, 3) for v in s2["robot"][:2]]); return True
        return self.hold("stuck", f"robot stuck against the {name} at ({x:+.2f}, {y:+.2f}): free it and put it inside the field",
                         lambda q: self.robot_clear(q) and math.hypot(q["robot"][0] - x, q["robot"][1] - y) >= 0.1)

    # ---- other programs
    def own_step(self):
        """the pids of the step this session runs now (its process group): not another program"""
        pid = self.runner.child_pid() if self.runner is not None and hasattr(self.runner, "child_pid") else None
        if pid is None or not self.procs: return set()
        return ({pid} | set(self.procs.group(pid))) if hasattr(self.procs, "group") else {pid}
    def busy_now(self, allow=()):
        b = [n for n in busy(self.procs, exclude=self.own_step()) if n not in allow] if self.procs else []
        if self.world is not None and self.world.snapshot().get("page_driving"): b.append("the field page (a click target or a placement)")
        return b
    def ensure_idle(self, allow=()):
        t0 = self.clock(); b = self.busy_now(allow)
        while b and self.clock() - t0 < self.busy_wait and not self.out_of_time():
            self.sleep(1.0); b = self.busy_now(allow)
        if not b: return True
        self.log("busy", programs=b)
        return self.hold("busy", f"another program commands the robot ({', '.join(b)}): the session waits", lambda q: not self.busy_now(allow))

    # ---- one step
    def command(self, st, retry=False):
        k = st["kind"]; py, T = self.py, HERE; e = lambda v: expand(v, self.vs)
        if k == "place":
            a = [py, f"{T}/place_ball.py", self.team, str(st["x"]), str(st["y"])] + [str(v) for v in e(st.get("opts", []))]
            if retry: a += ["--carry", "0.25"]
            return a
        if k == "test":
            return [py, f"{T}/characterize.py", self.team, st["test"]] + [str(v) for v in st.get("args", [])] + ["--out", e(st.get("out", "${SOLO_DATA}/logs"))]
        if k == "series":
            a = [py, f"{T}/series.py", self.team, e(st["plan"])] + [str(v) for v in e(st.get("args", []))]
            if st.get("from"): a += ["--from", str(st["from"])]
            if st.get("max"): a += ["--max", str(st["max"])]
            return a
        if k == "goto":
            return [py, f"{T}/goto.py", self.team, str(st["x"]), str(st["y"]), "--v", str(st.get("v", 0.4))]
        if k == "cmd":
            return [str(v) for v in e(st["argv"])]
        return None
    def env_for(self, st, retry=False):
        env = dict(os.environ)
        for src in (self.head.get("env", {}), st.get("env", {})):
            for k, v in expand(src, self.vs).items(): env[k] = str(v)
        if retry: env["SOLO_BOX"] = widen(env.get("SOLO_BOX", ""))
        return env
    def timeout_of(self, st):
        if "timeout" in st: return float(st["timeout"])
        if st["kind"] == "strategy": return float(st.get("seconds", 0)) + 15.0
        if st["kind"] == "place":
            o = st.get("opts", []); return (float(o[o.index("--timeout") + 1]) if "--timeout" in o else 60.0) + 20.0
        return float(TIMEOUTS.get(st["kind"], 60))

    def run_step(self, st, retry=False):
        """run one step; -> outcome dict(ok, line, data, rc, timed_out, stuck)"""
        k = st["kind"]; t0 = self.clock()
        if k == "marker":
            self.marker(expand(st["text"], self.vs), st.get("data", "")); return dict(ok=True, line="marker")
        if k == "wait":
            end = self.clock() + float(st["seconds"])
            while self.clock() < end and self.stop_req < 2: self.sleep(min(0.5, end - self.clock()))
            return dict(ok=True, line="waited")
        if k == "strategy":
            return self.run_strategy(st)
        argv = self.command(st, retry); env = self.env_for(st, retry)
        tee = expand(st["tee"], self.vs) if st.get("tee") else None
        r = self.runner.run(argv, env, self.timeout_of(st), tee=tee)
        r["ok"] = self.judge(st, r); r["stuck"] = "robot stuck" in (r.get("line") or "")
        self.log("step", n=st.get("n"), label=st.get("label", ""), step_kind=k, cmd=argv,
                 env={v: env[v] for v in ("SOLO_BOX", "SOLO_RAMP", "SOLO_BRAKE") if v in env}, start=round(r.get("t0", t0), 3),
                 end=round(r.get("t1", self.clock()), 3), rc=r.get("rc"), timed_out=r.get("timed_out"), ok=r["ok"], result=r.get("line"), retry=retry)
        return r
    def judge(self, st, r):
        k, d, line = st["kind"], r.get("data"), r.get("line") or ""
        if r.get("timed_out"): return False
        if k == "place": return bool(d and d.get("ok"))
        if k == "test": return r.get("rc") == 0 and isinstance(d, dict)
        if k == "goto": return line.startswith("arrived")
        return r.get("rc") == 0
    def run_strategy(self, st):
        t0 = self.clock()
        if not st.get("on", True):
            ok = self.strat.off(); self.log("step", n=st.get("n"), label=st.get("label", "strategy off"), step_kind="strategy", on=False, ok=ok, start=t0, end=self.clock())
            return dict(ok=ok, line="strategy off" if ok else "strategy still running")
        if not self.strat.can_start():
            self.log("step", n=st.get("n"), label=st.get("label", ""), step_kind="strategy", ok=False, result="no strategy command line (start it with run_real.sh, then off)")
            return dict(ok=False, line="strategy cannot start")
        on = self.strat.on(); t_on = self.clock(); end = t_on + float(st.get("seconds", 10))
        if not on: self.log("strategy_refused", n=st.get("n"), why=getattr(self.strat, "why", ""))
        track = []
        while on and self.clock() < end and self.stop_req < 2:
            s = self.world.snapshot(); track.append((round(self.clock() - t_on, 2), s["robot"] and [round(v, 3) for v in s["robot"][:2]], s["ball"] and [round(v, 3) for v in s["ball"]]))
            self.sleep(0.5)
        off = self.strat.off()
        self.log("step", n=st.get("n"), label=st.get("label", ""), step_kind="strategy", on=True, seconds=st.get("seconds"), started=on, stopped=off,
                 ok=on and off, start=round(t0, 3), end=round(self.clock(), 3), track=track[::2])
        if not off: self.stop_why = "the strategy node did not stop"; self.alert("the strategy node did not stop: the session ends; stop it on the field page"); self.stop_req = max(self.stop_req, 1)
        return dict(ok=on and off, line=f"strategy on {st.get('seconds')} s" + ("" if on and off else " FAILED"))

    def after(self, st, r):
        """look at the world after a step and handle the robot; -> True when the robot can go on"""
        s = self.look(); s = self.wait_ball_rest(s) if is_motion(st) else s
        self.log("world", after=st.get("n"), robot=s["robot"] and [round(v, 3) for v in s["robot"]], robot_age=s["robot_age"],
                 ball=s["ball"] and [round(v, 3) for v in s["ball"]], ball_age=s["ball_age"], ball_where=s["ball_where"], ball_class=s["ball_class"],
                 robot_clear=s.get("robot_clear"), robot_near=s.get("robot_near"))
        if not is_motion(st): return True
        if not self.robot_seen(s): return self.ensure_robot()
        failed = not r.get("ok")
        if r.get("stuck") or (failed and s["robot_clear"] < NEAR_WALL):
            return self.free_robot(s, r.get("line") or "")
        return True

    def handle_fail(self, st, r, pending):
        how = st.get("on_fail", "skip")
        if how == "retry" and not st.get("_retried"):
            st["_retried"] = True; pending.insert(0, st); self.log("decision", n=st.get("n"), action="retry"); return
        if how == "stop":
            self.stop_why = f"step {st.get('n')} failed (on_fail stop)"; self.stop_req = max(self.stop_req, 1)
        if how == "alert":
            self.alert(f"step {st.get('n')} ({st.get('label', st['kind'])}) failed: {(r.get('line') or '')[:160]}; the session goes on")
        if st.get("group"): self.skipped_groups.add(st["group"])
        self.skipped.append(st.get("n")); self.log("decision", n=st.get("n"), action=how, group_skipped=st.get("group"))

    def motion_ok(self, st):
        """the guard before every motion step: the robot seen, no other commanding program, the page not driving"""
        return self.ensure_robot() and self.ensure_idle(allow=("solo_node",) if st["kind"] == "strategy" else ())
    def execute(self, st, pending):
        k = st["kind"]
        if is_motion(st):
            if self.motion_halted: self.skipped.append(st.get("n")); self.log("decision", n=st.get("n"), action="skipped: motion stopped"); return
            if not self.motion_ok(st): pending.insert(0, st); return
        if self.out_of_time() and not st.get("final"): pending.insert(0, st); return
        r = self.run_step(st)
        if k == "place" and not r["ok"]:
            status = (r.get("data") or {}).get("status", "")
            if status.startswith("unreachable") and "spot" not in status:      # the ball's position, not the target
                st["_requeued"] = st.get("_requeued", 0) + 1
                b = (r.get("data") or {}).get("ball") or self.look()["ball"]
                if b: self.bad_ball = tuple(b[:2])
                if st["_requeued"] <= 2:
                    self.after(st, r); pending.insert(0, st); self.log("decision", n=st.get("n"), action="ball unreachable: the step waits for the ball"); return
            elif (r.get("timed_out") or status.startswith("failed: timeout") or status.startswith("failed: robot left")) and not r.get("stuck") and id(st) not in self.retried_place:
                self.retried_place.add(id(st)); self.log("decision", n=st.get("n"), action="placement retry: wider box, carry 0.25")
                if self.after(st, r) and not self.out_of_time():
                    r = self.run_step(st, retry=True)
        ok_go = self.after(st, r)
        if r["ok"]: self.done.append(st.get("n"))
        else: self.handle_fail(st, r, pending)
        return ok_go

    def needs(self, st):
        return st.get("needs_ball", st["kind"] == "place")
    def ball_blocks(self, st, s):
        """the ball lies inside the keep_ball_out box this ball-free step must keep clear"""
        return bool(st.get("keep_ball_out")) and ball_in_box(s["ball"] if s["ball_age"] < 1.5 else None, st["keep_ball_out"])
    def side_place(self, x, y, label):
        """a placement the supervisor adds (park the ball, rescue it): the same guard as every motion step before it,
        the robot checked after it; None when the guard did not let it start"""
        st = dict(kind="place", x=x, y=y, label=label, n=0)
        if label == "ball rescue": st["env"] = {"SOLO_BOX": ""}           # the placer's own area: field + margin
        if self.motion_halted or not self.motion_ok(st):
            self.log("decision", action=f"{label}: not started (motion stopped, robot not seen or another program)"); return None
        r = self.run_step(st); self.after(st, r); return r
    def pick(self, pending):
        """-> (step, None), or (None, blocked step), or (None, None) when only ball steps are left and the ball is unusable.
        Ball steps keep their order; ball-free steps go ahead of them while the ball cannot be used."""
        for st in [p for p in pending if p.get("group") in self.skipped_groups]:
            pending.remove(st); self.skipped.append(st.get("n")); self.log("decision", n=st.get("n"), action="skipped: its group failed")
        s = self.look(); ball_ok = None; blocked = None
        for st in pending:
            if self.needs(st):
                if ball_ok is None:
                    ball_ok = self.ball_ready(s) or self.ball_problem(s)
                    s = self.look()
                if ball_ok: return st, None
                continue
            if self.ball_blocks(st, s):
                if st.get("park") and self.ball_ready(s) and not self.motion_halted:
                    self.log("decision", n=st.get("n"), action=f"ball inside the test box: parked at {st['park']} first")
                    if self.side_place(st["park"][0], st["park"][1], "park the ball") is None: return None, None
                    s = self.look()
                    if not self.ball_blocks(st, s): return st, None
                blocked = blocked or st
                continue
            if st is not pending[0]: self.log("decision", n=st.get("n"), action="ball-free step taken ahead of the ball steps")
            return st, None
        return None, (blocked if ball_ok is not False else None)

    def run(self):
        main = [st for st in self.steps if not st.get("final")]; final = [st for st in self.steps if st.get("final")]
        self.log("session_start", name=self.head.get("name", ""), steps=len(self.steps), budget_s=round(self.t_end - self.t_start))
        pending = list(main)
        try:
            self.loop(pending)
        except Exception as e:                              # a fault of the supervisor itself: end cleanly
            import traceback
            self.log("error", error=repr(e), trace=traceback.format_exc()[-800:]); self.stop_why = f"supervisor error {e!r}"
        self.stop_why = self.stop_why or ("budget used" if self.clock() >= self.t_end else ("all steps done" if not pending else "stopped"))
        self.finish(final, pending)
        return self

    def loop(self, pending):
        while pending and not self.out_of_time():
            st, blocked = self.pick(pending)
            if st is None and blocked is not None:          # the ball sits in its box and cannot be parked: on_fail
                pending.remove(blocked); self.log("step", n=blocked.get("n"), label=blocked.get("label", ""), ok=False, result="ball inside the test box")
                self.handle_fail(blocked, dict(line=f"the ball lies inside the test box {blocked['keep_ball_out']}"), pending); continue
            if st is None:
                if not pending or not self.wait_for_ball(): break
                continue
            pending.remove(st)
            self.execute(st, pending)

    def finish(self, final, pending):
        try:
            if self.strat and self.strat.running(): self.log("end", action="strategy off", ok=self.strat.off())
        except Exception as e:
            self.log("end", action="strategy off", error=str(e))
        for st in final:
            if st["kind"] in ("marker", "cmd", "wait"): self.run_step(st)
        if not any(st["kind"] == "marker" for st in final): self.marker("auto session end", "skip")
        summary = f"session ended: {self.stop_why} ({len(self.done)} steps done, {len(self.skipped)} skipped, {len(pending)} not run)"
        self.log("session_end", why=self.stop_why, done=self.done, skipped=self.skipped, not_run=[s.get("n") for s in pending], alerts=len(self.alerts))
        self.alert(summary)


# ------------------------------------------------------------------ simulation (tests and --sim): no process, no network
class SimClock:
    def __init__(self): self.t = time.time()
    def __call__(self): return self.t
    def sleep(self, dt): self.t += max(dt, 0.01)


class SimProcs:
    """a process table: [(pid, argv)]; this process is pid 1"""
    def __init__(self, table=()): self.rows = list(table)
    def table(self): return list(self.rows)
    def mine(self): return {1}


class SimStrategy:
    def __init__(self, procs, can=True):
        self.procs = procs; self.can = can; self.ons = 0; self.offs = 0; self.cmdf = "solo.cmd"
    def running(self): return any(is_instance(a, "solo_node") for _, a in self.procs.table())
    def can_start(self): return self.can or self.running()
    def on(self):
        self.ons += 1
        if not self.running(): self.procs.rows.append((999, ["python3", "/x/lib/solo_strategy/solo_node", "--ros-args"]))
        return True
    def off(self):
        self.offs += 1; self.procs.rows = [r for r in self.procs.rows if r[0] != 999]; return True


class SimNotifier:
    def __init__(self, echo=False): self.sent = []; self.echo = echo
    def send(self, msg, title):
        self.sent.append(msg)
        if self.echo: print(f"*** NOTIFY: {title}: {msg}", flush=True)
        return "ok"


class SimRunner:
    """script: [dict(match="substring of the command", line=..., rc=0, dur=s, world={...}, timed_out=False, call=f)],
    consumed in order when the command matches the head; otherwise a default result: placements succeed and move
    the ball to the spot, gotos arrive, tests and series end well"""
    def __init__(self, clock, world, script=()):
        self.clock, self.world, self.script, self.calls = clock, world, list(script), []
    def run(self, argv, env, timeout, tee=None):
        cmd = " ".join(argv); self.calls.append(dict(cmd=cmd, env=env, timeout=timeout)); t0 = self.clock()
        if self.script and self.script[0]["match"] in cmd:
            e = self.script.pop(0)
            if e.get("call"): e["call"]()
            self.clock.t += e.get("dur", 5.0)
            if "world" in e: self.world.set(**e["world"])
            line = e["line"] if isinstance(e["line"], str) else json.dumps(e["line"])
            data, line = result_of([line])
            return dict(rc=e.get("rc", 0), timed_out=e.get("timed_out", False), line=line, data=data, t0=t0, t1=self.clock())
        self.clock.t += 5.0
        if "place_ball.py" in cmd:
            x, y = float(argv[3]), float(argv[4]); self.world.set(ball=(x, y))
            line = json.dumps(dict(ok=True, status="done", ball=[x, y], error=0.02))
        elif "goto.py" in cmd:
            x, y = float(argv[3]), float(argv[4]); self.world.set(robot=(x, y, 0.0)); line = f"arrived: robot at ({x:+.2f}, {y:+.2f}) after 3.0 s"
        elif "characterize.py" in cmd:
            line = json.dumps(dict(test=argv[3], args=argv[4:]))
        elif "series.py" in cmd:
            line = "series done"
        else:
            line = "ok"
        data, line = result_of([line])
        return dict(rc=0, timed_out=False, line=line, data=data, t0=t0, t1=self.clock())
    def abort(self): pass
    def cmds(self, key=""):
        return [c["cmd"] for c in self.calls if key in c["cmd"]]


SIM = {   # --sim scenarios: scripted results [match, line, world after] on top of the default "everything works"
    "ok": [],
    "corner": [dict(match="ram 1.5 0", line=dict(test="ram"), world=dict(ball=(2.52, 1.73)))],
    "goal": [dict(match="ram 1.2 0", line=dict(test="ram"), world=dict(ball=(2.33, 0.05)))],
    "stuck": [dict(match="place_ball.py blue 0.85", line=dict(ok=False, status="failed: robot stuck at (1.00, 1.65)"), world=dict(robot=(1.0, 1.65, 0.0))),
              dict(match="goto.py", line="vision lost the robot", world=dict(robot=(1.0, 1.65, 0.0)))],
    "timeout": [dict(match="place_ball.py", line=dict(ok=False, status="failed: timeout (60 s)")),
                dict(match="place_ball.py", line=dict(ok=False, status="failed: timeout (60 s)"))],
}
SIM_FIX = {"corner": (150, dict(ball=(0.2, -0.3))), "goal": (150, dict(ball=(0.2, -0.3))), "stuck": (150, dict(robot=(0.0, 0.0, 0.0)))}


# ------------------------------------------------------------------ dry run
DECISIONS = [
    ("ball inside the field / in a defense area / past a line within %.2f m" % PLACER_MARGIN, "ball steps run", "-"),
    ("ball in a goal, against a goal frame, in a corner, against a wall", "one place_ball run toward the nearest point 0.3 m inside, if BallPlacer can; ball-free steps go on", "'<where>: put it anywhere inside the field' (once; reminders only when nothing ball-free is left)"),
    ("ball not seen > ball-lost s", "ball-free steps go on; ball steps wait", "'ball not seen since T (last at ...): put a ball anywhere inside the field'"),
    ("place_ball: unreachable (ball)", "as the line above; the place step stays first among the ball steps (twice at most)", "as above"),
    ("place_ball: failed timeout / left the area", "once more: box +0.15 m, --carry 0.25; then on_fail", "only with on_fail alert"),
    ("place_ball: robot stuck, or abort/timeout with the robot within %.2f m of a wall or frame" % NEAR_WALL, "one goto.py at 0.25 m/s to 0.5 m inside the field; moved < 0.1 m: motion stops, wait", "'robot stuck against the <wall> at (x, y): free it and put it inside the field'"),
    ("robot not seen before a motion step", "wait robot-wait s, then motion stops and the session waits", "'robot not seen since T, last at (x, y): ...'"),
    ("another commanding program runs", "no motion step; after busy-wait s the session waits", "'another program commands the robot (...): the session waits'"),
    ("a test ABORT / non-zero exit / step time limit", "on_fail: skip (and the rest of its group) / retry once / stop / alert", "only with on_fail alert or stop"),
    ("budget used / Ctrl-C", "current step ends, strategy off, final steps, end marker", "'session ended: ...'"),
]

def dry_run(sess, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p(f"session {sess.head.get('name', '')}: {len(sess.steps)} steps, budget {round(sess.t_end - sess.t_start)} s; log {sess.log_path}; events {sess.events_path}")
    p(f"notify command: {'set' if os.environ.get('SOLO_NOTIFY_CMD') else 'NOT SET (alerts only printed)'}")
    if sess.procs: p(f"commanding programs running now: {busy(sess.procs) or 'none'}")
    if sess.strat: p(f"strategy: {'running' if sess.strat.running() else 'off'}; restart line {'present' if os.path.exists(sess.strat.cmdf) else 'MISSING'} ({sess.strat.cmdf})")
    for st in sess.steps:
        k = st["kind"]; needs = st.get("needs_ball", k == "place")
        flags = f"needs_ball={needs} on_fail={st.get('on_fail', 'skip')} limit={sess.timeout_of(st):.0f}s" + (f" group={st['group']}" if st.get("group") else "") + (" FINAL" if st.get("final") else "")
        if k == "marker": what = f"marker {expand(st['text'], sess.vs)!r} {st.get('data', '')}"
        elif k == "wait": what = f"wait {st['seconds']} s"
        elif k == "strategy": what = f"strategy on {st.get('seconds')} s, then off" if st.get("on", True) else "strategy off"
        else:
            env = sess.env_for(st); keys = sorted(set(sess.head.get("env", {})) | set(st.get("env", {})))
            what = " ".join(f"{v}={env.get(v, '')!r}" for v in keys) + " " + " ".join(sess.command(st))
            if k == "place": what += f"\n        retry: SOLO_BOX={widen(env.get('SOLO_BOX', ''))!r} ... --carry 0.25"
        p(f"  {st.get('n'):>3} [{k}] {st.get('label', '')}  ({flags})\n        {what}")
        if st.get("keep_ball_out"): p(f"        ball kept 0.15 m clear of {st['keep_ball_out']}" + (f", parked at {st['park']} if not" if st.get("park") else ""))
    p("\nsituation -> what the supervisor does -> what the person receives")
    for a, b, c in DECISIONS: p(f"  {a}\n      -> {b}\n      -> {c}")


def main():
    args = sys.argv[1:]
    def opt(name, default):
        if name in args:
            i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
        return default
    budget = float(opt("--budget", "0")); hq = opt("--hq", "http://127.0.0.1:8700")
    kw = dict(ball_lost=float(opt("--ball-lost", "5")), robot_lost=float(opt("--robot-lost", "2")), robot_wait=float(opt("--robot-wait", "10")),
              busy_wait=float(opt("--busy-wait", "20")), realert=float(opt("--realert", "180")))
    sim = opt("--sim", None)
    dry = "--dry-run" in args; args = [a for a in args if a != "--dry-run"]
    extra = {}
    while "--var" in args:
        k, _, v = opt("--var", "").partition("="); extra[k] = v
    if len(args) < 2 or args[0] not in ("blue", "yellow"): sys.exit(__doc__)
    team, path = args[0], args[1]
    head, steps, vs = load_session(path, team, extra)
    budget_s = 60.0 * (budget or float(head.get("budget_min", 10)))
    stamp = time.strftime("%Y%m%d_%H%M%S"); dd = data_dir()
    log_path = os.path.join(dd, "auto", f"{stamp}_session.jsonl"); events = os.path.join(dd, time.strftime("events_%Y%m%d.jsonl"))
    procs = Procs()
    if dry:
        dry_run(Session(team, steps, head, vs, procs=procs, strategy=Strategy(procs), log_path=log_path, events_path=events, budget_s=budget_s, **kw)); return
    if sim is not None:                                # the whole session on a simulated clock, fakes for every step
        clk = SimClock(); fix = SIM_FIX.get(sim)
        world = ScriptedWorld(clk, robot=(0.0, 0.6, 0.0), ball=(0.3, -0.9), timeline=[(clk.t + fix[0], fix[1])] if fix else ())
        procs = SimProcs(); sess = Session(team, steps, head, vs, world=world, runner=SimRunner(clk, world, SIM[sim]), notifier=SimNotifier(echo=True),
                                           procs=procs, strategy=SimStrategy(procs), clock=clk, sleep=clk.sleep, budget_s=budget_s,
                                           log_path=os.path.join(dd, "auto", f"{stamp}_sim_{sim}.jsonl"), **kw)
        sess.run(); print(f"simulated session log {sess.log_path}"); return
    sess = Session(team, steps, head, vs, world=PageWorld(hq), runner=Runner(), notifier=Notifier(), procs=procs, strategy=Strategy(procs),
                   log_path=log_path, events_path=events, budget_s=budget_s, python=os.environ.get("SOLO_PYTHON", "python3"), **kw)
    signal.signal(signal.SIGINT, sess.request_stop); signal.signal(signal.SIGTERM, sess.request_stop)
    time.sleep(1.5)                                    # the first /state polls
    if sess.world.snapshot()["page_age"] > 5: print(f"the field page does not answer at {hq}/state: start hq.py first", flush=True); sys.exit(1)
    print(f"session log {log_path}", flush=True)
    sess.run()


if __name__ == "__main__":
    main()
