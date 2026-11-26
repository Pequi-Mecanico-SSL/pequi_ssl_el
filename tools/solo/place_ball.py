#!/usr/bin/env python3
"""Put the ball on a spot with the robot alone on the field: the robot pushes it with its flat front (no kicker, no
dribbler) until it rests within a tolerance of the spot, then backs away from it.
    source tools/solo/env.sh; python3 tools/solo/place_ball.py <team> x y [--precise | --tol 0.15] [--timeout 60] [--speed 0.6]
        [--carry 0.35] [--margin 0.25] [--mode auto|carry|dose] [--keep-out-defense] [--out DIR] [--watch FILE]
Rough by default (tol 0.15 m: usually one push); --precise is tol 0.05. A test should start from where the ball
really rests: the JSON line printed at the end has it ("ball", measured with the robot backed away, and its distance
"error" from the spot), with ok, status, time, pushes, the roll gain it learned and the log file.
The strategy node must be stopped; the tool refuses to run while it (or another test) is running on this computer.

How it works (BallPlacer, no ROS in it: state in, body-frame command out, so it can be simulated and reused):
  1. settle: wait until the ball is seen and still (a rolling ball is never chased). Within tol: back away and stop.
  2. plan: push direction from the ball to the spot. The robot needs a line-up point 0.16-0.30 m behind the ball
     inside the allowed area (the field plus `margin` beyond the lines, minus the goals and their frames, minus the
     defense areas with keep_out_defense). If there is none, a push at up to 80 deg off the direct line toward a
     nearer intermediate point is used, and the spot is finished from there. A ball in a goal, against a goal frame
     or too far beyond the lines for any line-up point is reported as "unreachable: ...", as is a spot outside the field.
  3. line-up: turn toward the push direction while driving to the line-up point, round the ball on a circle of
     0.32 m when the straight way passes near it (the side whose arc stays in the area), hold still 0.25 s.
  4. push, by distance s to the spot:
       s < 0.20 m   creep: touch at 0.12 m/s and carry; the ball rolls about 3 cm after the stop
       s < dose_min carry: touch at <= 0.3 m/s, carry at up to `carry` m/s on the front, slow to 0.2 m/s and stop where
                    the roll that follows (model x gain) ends on the spot
       longer       dose: a push at the contact speed whose roll stops the ball 0.3 m SHORT even if it rolls 35 %
                    farther than the model; the next push continues from behind it without rounding it
     While the ball sits on the front it may not be seen: its position is then the robot's front plus the last seen
     sideways offset. The robot slides sideways to keep the ball centred; a ball that leaves the front sideways ends
     the push and the next one is planned from where it rests.
  5. after each push: wait until the ball rests, compare the roll with the model and update the roll gain (geometric
     mean of the old gain and the last ratio, within 0.6..1.7) and, for doses, the exit-angle bias. Repeat from 1.
  6. done: back 0.35 m straight away from the ball (or the free direction that most increases the distance), stop.
  Ball not seen for 1 s (vision does not report a ball touching the robot): last seen within 0.3 m of the robot, or
  never seen, -> back 0.35 m away from it (from the front when the robot stands on its last seen point); last seen
  farther -> drive to 0.5 m short of it; look again; after 3 tries "failed: ball not seen (last seen at ...)".
  Stuck: more than 0.15 m/s commanded for over 1 s while the robot moved < 4 cm (a wall, a goal frame) -> back 0.3 m
  along the reverse of the command at 0.15 m/s and re-plan; the second time "failed: robot stuck at (x, y)" with
  zeros; the result has "stuck" (count) and "stuck_failed" for a supervisor to alert a person.
Roll model: deceleration c0 + c1 v = 0.157 + 0.116 v m/s2. After a carry the ball travels latency x v + gain x
roll(release_k x v) from where it is at the stop command (v the commanded speed): release_k 0.68 with latency 0.23 s,
fitted on 9 carries of the real robot at 0.2-0.45 m/s (rms 0.035 m; release_k 0.9 gives 0.12 m on them). The robot
stops 0.03-0.13 m after the command, the ball leaves its front at 0.4-0.9 x the robot's speed. Tools: tests/fit_release.py.
During a carry the heading is turned toward the push line by up to 7 deg (1.5 rad/m of the ball's offset from it): on
the real robot a ball sliding on the front pulled the robot 5 cm off the line, as the robot keeps the ball centred.

Safety in the command-line tool: best-effort publisher (as characterize.py), zeros whenever the pose is older than
0.2 s (as hq.py), SOLO_BOX (the area the robot's centre must stay in, as in characterize.py; leaving it aborts with
zeros). It refuses to start while another commanding program runs (procs.COMMANDERS: solo_node, characterize.py,
series.py, ram_series.py, another place_ball.py, goto.py, nudge.py, frame_check.py); the check skips this process and
all its ancestors (series.py or a wrapper script that started it is not "another program") and counts a program only
when it is the executable of a process or the script an interpreter runs (python3 x.py, not an editor open on x.py).
Zeros for 0.3 s on every exit it can handle: the end, an abort, SIGINT (Ctrl-C) and SIGTERM. Both signals only set a
flag; the loop then ends the placement ("failed: stopped by SIGTERM"), sends the zeros, prints the result line and
saves the log. rclpy is started without its own signal handlers (SignalHandlerOptions.NO, ROS 2 Humble and later) so
that it cannot shut the context down before the zeros are sent. On an older distribution without that option rclpy
keeps its own SIGINT handler, which may shut the context down first: the zeros after a Ctrl-C are then not sent
(SIGTERM is not affected), and the robot's own command timeout stops it. After SIGKILL nothing is sent.
--watch FILE is a dead man's switch: the caller touches FILE (e.g. every 0.5 s) and the tool stops with zeros
("failed: dead man's file ...") once its modification time is more than 2 s old or it is missing; without the option
there is no such check. --speed, --carry, --tol, --margin, --wall and --timeout must be finite positive numbers.

Integration:
  series.py, for a "spot" test: instead of the wiggle-and-wait loop, run
      python3 place_ball.py <team> sx sy --tol 0.05 --timeout 60
  before the goto to the run-up point; parse the last stdout line (JSON), and on "ok": false fall back to the person
  (wiggle and wait, as series.py does now). The spot tolerance of ball_ready (0.08) is wider than the default tol, so a placed ball
  passes it. The placer leaves the robot >= 0.3 m from the ball, and the goto that follows rounds the ball.
  Or in-process: BallPlacer(...).step(t, pose_or_None, ball_or_None) at 50 Hz, publish the command, until
  placer.finished; placer.result() is the JSON dict.
  hq.py button: a "place ball" mode in which a click sets the spot; the control loop then calls placer.step() instead
  of the go-to controller (same 0.2 s pose freshness, same dead man's switch: the page polling stops -> placer dropped,
  zeros), shows placer.status in the note, draws the spot and the line-up point (placer.debug()), and drops the
  placer when another program starts.
"""
import json, math, os, signal, sys, time
from collections import deque
if os.path.dirname(os.path.abspath(__file__)) not in sys.path: sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import procs  # noqa: E402  the commanding programs (shared with hq.py and auto_session.py)

FIELD_X, FIELD_Y = 2.25, 1.5
GOAL_HW, GOAL_D = 0.4, 0.18           # goal half width, depth behind the goal line
DA_D, DA_HW = 0.5, 0.675              # defense area depth and half width
ROBOT_R = 0.09
FRONT = 0.09                          # flat front at 0.09 m from the centre, about 0.14 m wide
FRONT_HW = 0.07
BALL_R = 0.0215
HOLD = FRONT + BALL_R                 # ball centre in front of the robot centre while touching
GOAL_FRAME = [((FIELD_X, GOAL_HW), (FIELD_X + GOAL_D, GOAL_HW)), ((FIELD_X + GOAL_D, GOAL_HW), (FIELD_X + GOAL_D, 0.0))]
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
clip = lambda v, lo, hi: lo if v < lo else hi if v > hi else v


def seg_dist(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0; L2 = dx * dx + dy * dy
    k = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / L2)) if L2 > 0 else 0.0
    return math.hypot(px - x0 - k * dx, py - y0 - k * dy)


def roll(v, c0=0.157, c1=0.116):
    """distance (m) a ball rolls from speed v with deceleration c0 + c1 v"""
    if v <= 0: return 0.0
    return v / c1 - c0 / c1 ** 2 * math.log1p(c1 * v / c0)


def roll_speed(s, c0=0.157, c1=0.116):
    """the speed from which a ball rolls s m"""
    lo, hi = 0.0, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if roll(mid, c0, c1) > s: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


class BallPlacer:
    """step(t, robot, ball) -> (vx, vy, w, status). robot: (x, y, theta) or None (no fresh pose); ball: (x, y) of a
    detection that arrived since the last step, or None (none new). Field frame, metres, radians; t in seconds (monotonic). The command is body-frame (x forward, y left,
    w counter-clockwise). `finished` is True once status starts with done / unreachable / failed."""

    def __init__(self, target, tol=0.15, speed=0.6, carry=0.35, margin=0.25, mode="auto", keep_out_defense=False,
                 timeout=60.0, max_pushes=15, dose_min=1.2, release_k=0.68, c0=0.157, c1=0.116, latency=0.23,
                 backoff=0.35, accel=1.5, wall=0.30):
        self.target = (float(target[0]), float(target[1])); self.tol = tol; self.vmax = speed; self.carry = carry
        self.margin = margin; self.wall = wall; self.mode = mode; self.keep_da = keep_out_defense; self.timeout = timeout
        self.max_pushes = max_pushes; self.dose_min = dose_min; self.k = release_k; self.c0, self.c1 = c0, c1
        self.latency = latency; self.backoff_d = backoff; self.acc = accel
        self.gain = 1.0; self.bias = 0.0          # learned: roll gain, exit angle of a dose relative to the heading
        self.state = "settle"; self.status = "waiting: ball"; self.t0 = None; self.t_state = 0.0
        self.balls = deque(); self.ball_last = None; self.ball_t = -1e9
        self.pushes = 0; self.history = []; self.vf = [0.0, 0.0]; self.t_prev = None
        self.plan = None; self.lat = 0.0; self.rel = None; self.bumps = 0; self.reveal = 0; self.finished = False
        self.final = None; self.t_end = None; self.remeasured = None; self.stuck = 0; self.t_move = None

    # ---- geometry --------------------------------------------------------------------------------------------
    def allowed(self, x, y):
        """may the robot's centre be here?"""
        m = self.lim()                         # the robot's centre: margin, and its body clear of the wall
        if abs(x) > FIELD_X + m or abs(y) > FIELD_Y + m: return False
        ax, ay = abs(x), abs(y)
        if ax > FIELD_X - 0.02 and ay < GOAL_HW: return False                       # into a goal
        if ax > FIELD_X - ROBOT_R - 0.05 and ay < GOAL_HW + ROBOT_R + 0.05:         # near a goal frame
            for (x0, y0), (x1, y1) in GOAL_FRAME:
                if seg_dist(ax, ay, x0, y0, x1, y1) < ROBOT_R + 0.03: return False
        if self.keep_da and abs(x) > FIELD_X - DA_D - ROBOT_R and abs(y) < DA_HW + ROBOT_R: return False
        return True

    def lim(self):
        return min(self.margin, self.wall - ROBOT_R - 0.03) if self.wall else self.margin

    def clamp(self, p):
        m = self.lim()
        return (clip(p[0], -FIELD_X - m, FIELD_X + m), clip(p[1], -FIELD_Y - m, FIELD_Y + m))

    def path_ok(self, a, b, step=0.03):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        return all(self.allowed(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1))

    def check_target(self):
        x, y = self.target
        if abs(x) > FIELD_X or abs(y) > FIELD_Y: return "unreachable: spot outside the field"
        if self.keep_da and abs(x) > FIELD_X - DA_D and abs(y) < DA_HW: return "unreachable: spot in a defense area"
        return None

    def check_ball(self, b):
        x, y = b
        if abs(x) > FIELD_X and abs(y) < GOAL_HW + 0.02: return "unreachable: ball in a goal"
        if abs(x) > FIELD_X - 0.03 and abs(y) < GOAL_HW + 0.02 + 0.05: return "unreachable: ball against a goal frame"
        if abs(x) > FIELD_X + self.margin or abs(y) > FIELD_Y + self.margin:
            return f"unreachable: ball beyond the margin ({self.margin:.2f} m past the lines)"
        if self.keep_da and abs(x) > FIELD_X - DA_D and abs(y) < DA_HW: return "unreachable: ball in a defense area"
        return None

    def make_plan(self, b, target=None, depth=0):
        """-> dict(u, aim, stand, kind, v) or an 'unreachable: ...' string"""
        tx, ty = target or self.target; dx, dy = tx - b[0], ty - b[1]; s = math.hypot(dx, dy)
        base = math.atan2(dy, dx); best = None
        for off in (0, 10, -10, 20, -20, 35, -35, 50, -50, 65, -65, 80, -80):
            a = base + math.radians(off); u = (math.cos(a), math.sin(a))
            sp = s * math.cos(math.radians(off))
            if off and sp < 0.15: continue
            aim = (b[0] + u[0] * sp, b[1] + u[1] * sp)
            if abs(aim[0]) > FIELD_X or abs(aim[1]) > FIELD_Y: continue
            stand = None
            for L in (0.30, 0.26, 0.22, 0.19, 0.16):
                p = (b[0] - u[0] * L, b[1] - u[1] * L)
                end = (aim[0] - u[0] * HOLD, aim[1] - u[1] * HOLD)
                if self.allowed(*p) and self.path_ok(p, end): stand = (p, L); break
            if stand is None: continue
            left = s * abs(math.sin(math.radians(off)))
            best = dict(u=u, ang=a, aim=aim, stand=stand[0], run=stand[1], dist=sp, left=left, ball=b); break
        if best is None and depth == 0:
            # no push toward the spot: move the ball first to a point from which there is one (e.g. a ball that stopped
            # in front of a goal, a spot that can only be reached from one side)
            cands = []
            for k in range(16):
                a = 2 * math.pi * k / 16
                for dist in (0.3, 0.5, 0.8):
                    q = (b[0] + dist * math.cos(a), b[1] + dist * math.sin(a))
                    if abs(q[0]) > FIELD_X - 0.1 or abs(q[1]) > FIELD_Y - 0.1: continue
                    if self.keep_da and abs(q[0]) > FIELD_X - DA_D and abs(q[1]) < DA_HW: continue
                    cands.append((math.hypot(q[0] - tx, q[1] - ty) + 0.3 * dist, q))
            for _, q in sorted(cands):
                p1 = self.make_plan(b, q, 1)
                if isinstance(p1, dict) and isinstance(self.make_plan(q, None, 1), dict):
                    p1["reposition"] = True; best = p1; break
            if best is not None: return best
        if best is None:
            return "unreachable: no line-up point behind the ball inside the allowed area (margin %.2f m)" % self.margin
        sp = best["dist"]; mode = self.mode
        u = best["u"]; a = best["aim"]       # a goal or the wall just beyond the aim: stop it short of the model
        best["careful"] = any(not self.allowed(a[0] + u[0] * d, a[1] + u[1] * d) or
                              (abs(a[0] + u[0] * d) > FIELD_X and abs(a[1] + u[1] * d) < GOAL_HW + 0.03) or
                              (self.keep_da and abs(a[0] + u[0] * d) > FIELD_X - DA_D and abs(a[1] + u[1] * d) < DA_HW)
                              for d in (0.1, 0.2, 0.3, 0.45))
        if sp < 0.2:
            best.update(kind="creep", v=0.12)
        elif mode == "dose" or (mode == "auto" and sp > self.dose_min):
            want = sp - 0.3 if sp > 0.6 else 0.5 * sp                              # stop short of the spot
            if (sp + self.tol) / 1.35 >= (sp - self.tol) / 0.8 and not best["careful"]:
                want = (sp + self.tol) * 0.98                                       # the whole scatter band within tol
            vb = roll_speed(want / (1.35 * self.gain), self.c0, self.c1)
            v = clip(vb / self.k, 0.15, min(0.8, self.vmax)); best.update(kind="dose", v=v)
        else:
            best.update(kind="carry", v=min(0.3, self.carry))
        return best

    def release_dist(self, v):
        """ball travel still to come after the stop command at robot speed v (pose age + reaction, then the roll);
        a careful push (a goal or a wall behind the spot) assumes the long end of the roll scatter (+35 %)"""
        f = 1.3 if isinstance(self.plan, dict) and self.plan.get("careful") else 1.0
        return v * self.latency + f * self.gain * roll(self.k * v, self.c0, self.c1)

    # ---- helpers ---------------------------------------------------------------------------------------------
    def set(self, state, t, status=None):
        self.state = state; self.t_state = t
        if status: self.status = status

    def ball_still(self, t, win=0.5, tol=0.006, need=8, spread=0.015):
        """the ball's resting position (median of the last `win` s) or None while it moves. Vision reports a resting
        ball with a few mm of jitter (sd up to 5 mm, 2 cm peak to peak on the real field), so stillness is judged by
        the drift between the medians of the two halves of the window (< tol: under about 2.5 cm/s) and a robust
        spread (sd from the median absolute deviation < spread), not by the peak-to-peak range."""
        h = [p for p in self.balls if t - p[0] <= win]
        if len(h) < need or t - h[-1][0] > 0.2 or h[-1][0] - h[0][0] < 0.6 * win: return None
        med = lambda v: sorted(v)[len(v) // 2]
        k = len(h) // 2; out = []
        for i in (1, 2):
            a = [p[i] for p in h[:k]]; b = [p[i] for p in h[k:]]
            if abs(med(b) - med(a)) > tol: return None
            m = med([p[i] for p in h])
            if 1.4826 * med([abs(p[i] - m) for p in h]) > spread: return None
            out.append(m)
        return tuple(out)

    def drive(self, r, goal, heading, vmax, dt, tol=0.02):
        """field velocity toward goal (ramped), turning to heading -> body command and distance left"""
        x, y, th = r; d = math.hypot(goal[0] - x, goal[1] - y); eth = wrap(heading - th)
        v = min(vmax, 1.6 * d + 0.02) * (0.4 if abs(eth) > 0.7 else 1.0)
        want = ((goal[0] - x) / d * v, (goal[1] - y) / d * v) if d > tol * 0.5 else (0.0, 0.0)
        want = self.guard(r, want)
        self.ramp(want, dt)
        w = clip(2.5 * eth, -1.5, 1.5)
        return self.body(th, w) + (w,), d, eth

    def guard(self, r, want):
        """slow the motion toward the outer limit so that the robot stops inside it (0.35 s of delay, 1.2 m/s2)"""
        m = self.lim(); out = list(want)
        for k, L in ((0, FIELD_X + m), (1, FIELD_Y + m)):
            v = out[k]
            if v == 0: continue
            d = L - (r[k] if v > 0 else -r[k])
            vm = max(0.0, -0.84 + math.sqrt(0.7056 + 2.4 * max(0.0, d - 0.01)))   # v*0.35 + v^2/2.4 = d
            if abs(v) > vm: out[k] = math.copysign(vm, v)
        return tuple(out)

    def ramp(self, want, dt):
        for k in (0, 1):                       # ramp up, never delay slowing down
            if abs(want[k]) > abs(self.vf[k]) and want[k] * self.vf[k] >= 0:
                self.vf[k] += clip(want[k] - self.vf[k], -self.acc * dt, self.acc * dt)
            else: self.vf[k] = want[k]

    def body(self, th, w):
        thp = th + w * 0.1                     # the heading when the command acts
        c, s = math.cos(thp), math.sin(thp)
        return (c * self.vf[0] + s * self.vf[1], -s * self.vf[0] + c * self.vf[1])

    def zero(self):
        self.vf = [0.0, 0.0]; return (0.0, 0.0, 0.0)

    def end_on_estimate(self, t, why):
        """end with the best resting estimate: "done" when it is within tol, else "failed: <why>" (why "done": the
        placement was judged on the spot before backing away)"""
        b = self.final or self.ball_still(t, win=1.0, tol=0.02, spread=0.03) or self.ball_last
        if b is not None and self.final is None: self.final = b
        if b is not None and math.hypot(b[0] - self.target[0], b[1] - self.target[1]) <= self.tol:
            return self.stop_with(t, "done")
        return self.stop_with(t, why if why.startswith("failed") else "failed: ball not on the spot")

    def stop_with(self, t, status):
        self.status = status; self.finished = True; self.t_end = t; self.state = "end"
        return (0.0, 0.0, 0.0, status)

    def debug(self):
        p = self.plan if isinstance(self.plan, dict) else {}
        return dict(state=self.state, target=self.target, stand=p.get("stand"), aim=p.get("aim"), kind=p.get("kind"),
                    gain=round(self.gain, 3), pushes=self.pushes)

    def result(self):
        b = self.final or self.ball_last
        err = math.hypot(b[0] - self.target[0], b[1] - self.target[1]) if b else None
        return dict(ok=self.status == "done", status=self.status, target=[round(v, 3) for v in self.target], tol=self.tol,
                    ball=[round(v, 3) for v in b] if b else None, ball_seen_still=self.final is not None,
                    remeasured=self.remeasured, error=round(err, 3) if err is not None else None,
                    time=round((self.t_end or 0.0) - (self.t0 or 0.0), 1), pushes=self.pushes, gain=round(self.gain, 3),
                    bias_deg=round(math.degrees(self.bias), 1), bumps=self.bumps, stuck=self.stuck,
                    stuck_failed=self.status.startswith("failed: robot stuck"), history=self.history)

    # ---- the loop --------------------------------------------------------------------------------------------
    def step(self, t, robot, ball):
        if self.finished: return (0.0, 0.0, 0.0, self.status)
        if self.t0 is None:
            self.t0 = t; self.t_prev = t
            why = self.check_target()
            if why: return self.stop_with(t, why)
        dt = clip(t - self.t_prev, 1e-3, 0.1); self.t_prev = t
        if ball is not None:
            self.balls.append((t, ball[0], ball[1])); self.ball_last = (ball[0], ball[1]); self.ball_t = t
        while self.balls and t - self.balls[0][0] > 1.5: self.balls.popleft()
        if t - self.t0 > self.timeout and self.state not in ("backoff", "confirm"):    # those two end within 4 + 2 s
            b = self.ball_still(t) or self.ball_still(t, win=1.0, tol=0.02, spread=0.03)
            self.final = b or self.ball_last
            if b is not None and math.hypot(b[0] - self.target[0], b[1] - self.target[1]) <= self.tol:
                self.remeasured = False; return self.stop_with(t, "done")       # resting on the spot: ok
            return self.stop_with(t, f"failed: timeout ({self.status})")
        if t - self.t0 > self.timeout + 8.0:
            return self.end_on_estimate(t, f"failed: timeout ({self.status})")
        if robot is None:                      # never drive blind
            self.vf = [0.0, 0.0]
            return (0.0, 0.0, 0.0, self.status)
        fn = getattr(self, "s_" + self.state)
        out = fn(t, robot, ball, dt)
        if not self.finished:
            out = self.stuck_check(t, robot, out)
        return out if len(out) == 4 else out + (self.status,)

    def stuck_check(self, t, r, out):
        """commanded > 0.15 m/s for over 1 s while the measured position moved < 4 cm: against a wall or a goal frame.
        Back off 0.3 m along the reverse of the command at 0.15 m/s and re-plan; stuck again: end, zeros."""
        v = math.hypot(*self.vf)
        if v < 0.15 or self.state == "unstick":
            self.t_move = None; return out
        if self.t_move is None: self.t_move = (t, r[0], r[1])
        elif t - self.t_move[0] > 1.0:
            if math.hypot(r[0] - self.t_move[1], r[1] - self.t_move[2]) < 0.04:
                self.stuck += 1; self.t_move = None
                if self.stuck > 1:
                    self.zero(); self.stop_with(t, f"failed: robot stuck at ({r[0]:.2f}, {r[1]:.2f})"); return (0.0, 0.0, 0.0)
                u = (-self.vf[0] / v, -self.vf[1] / v)
                self.back = self.clamp((r[0] + 0.3 * u[0], r[1] + 0.3 * u[1]))
                self.set("unstick", t, "stuck: backing off"); return self.zero()
            self.t_move = (t, r[0], r[1])
        return out

    def s_unstick(self, t, r, ball, dt):
        cmd, d, _ = self.drive(r, self.back, r[2], 0.15, dt)
        if d < 0.03 or t - self.t_state > 3.0:
            self.set("settle", t, "re-planning after being stuck"); return self.zero()
        return cmd

    def s_settle(self, t, r, ball, dt):
        b = self.ball_still(t)
        if b is None and t - self.t_state > 3.0: b = self.ball_still(t, win=1.0, tol=0.02, spread=0.03)
        if b is None:
            seen = t - self.ball_t < 0.3
            self.status = "waiting: ball moving" if seen else "waiting: ball not seen"
            if not seen and t - self.ball_t > 1.0 and t - self.t_state > 1.0:
                return self.search(t, r)
            return self.zero()
        self.reveal = 0
        why = self.check_ball(b)
        if why:
            self.final = b; return self.stop_with(t, why)
        err = math.hypot(b[0] - self.target[0], b[1] - self.target[1])
        if err <= self.tol - 0.004:                # 4 mm inside: the estimate is 1-2 mm noisy
            self.final = b
            if math.hypot(b[0] - r[0], b[1] - r[1]) < self.backoff_d + HOLD:
                self.back = self.back_point(r, b, self.backoff_d)
                self.set("backoff", t, "backing away"); return self.zero()
            self.set("confirm", t, "measuring where the ball rests"); return self.zero()
        if self.pushes >= self.max_pushes:
            self.final = b; return self.stop_with(t, "failed: too many pushes")
        p = self.make_plan(b)
        if isinstance(p, str):
            self.final = b; return self.stop_with(t, p)
        self.plan = p; self.orbit = None
        self.set("lineup", t, f"lining up for a {p['kind']} push of {p['dist']:.2f} m")
        return self.zero()

    def search(self, t, r):
        """the ball is not seen: vision does not report a ball touching the robot. Last seen near the robot (or never
        seen): it is probably on the front or under its edge, so back 0.35 m away from it and look again; last seen
        far away: drive to a point 0.5 m short of it. After 3 tries: give up with the last known position."""
        if self.reveal >= 3:
            b = self.ball_last
            return self.stop_with(t, "failed: ball not seen" + (f" (last seen at ({b[0]:.2f}, {b[1]:.2f}))" if b else ""))
        self.reveal += 1
        c, s = math.cos(r[2]), math.sin(r[2]); front = (r[0] + c * HOLD, r[1] + s * HOLD)
        b = self.ball_last
        if b is None or math.hypot(b[0] - r[0], b[1] - r[1]) < 0.3:
            fw = ((b[0] - r[0]) * c + (b[1] - r[1]) * s) if b else 0.0
            est = b if b is not None and fw > HOLD - 0.02 else front       # driven over its last seen point: on the front
            self.back = self.back_point(r, est, 0.35)
            self.set("reveal", t, "ball not seen: backing off to look for it")
        else:
            d = math.hypot(b[0] - r[0], b[1] - r[1]); f = (d - 0.5) / d
            q = self.clamp((r[0] + (b[0] - r[0]) * f, r[1] + (b[1] - r[1]) * f))
            self.back = q if self.allowed(*q) and self.path_ok(r[:2], q) else r[:2]
            self.set("reveal", t, "ball not seen: driving closer to look for it")
        return self.zero()

    def back_point(self, r, b, dist):
        """a point dist m farther from the ball than the robot is now, away from it, inside the area"""
        best, far = None, -1.0
        for frac in (1.0, 0.6):
            for off in (0, 30, -30, 60, -60, 90, -90, 120, -120):
                a = math.atan2(r[1] - b[1], r[0] - b[0]) + math.radians(off)
                p = (r[0] + frac * dist * math.cos(a), r[1] + frac * dist * math.sin(a))
                if not (self.allowed(*p) and self.path_ok(r[:2], p)): continue
                db = math.hypot(p[0] - b[0], p[1] - b[1])
                if db > math.hypot(r[0] - b[0], r[1] - b[1]) + 0.5 * frac * dist and off in (0, 30, -30):
                    return p
                if db > far: best, far = p, db
            if best is not None: return best
        return r[:2]

    def s_reveal(self, t, r, ball, dt):
        cmd, d, _ = self.drive(r, self.back, r[2], 0.25, dt)
        if d < 0.025 or t - self.t_state > 4.0:
            self.set("settle", t); return self.zero()
        return cmd

    def s_backoff(self, t, r, ball, dt):
        cmd, d, _ = self.drive(r, self.back, r[2], min(0.4, self.vmax), dt)
        if d < 0.03 or t - self.t_state > 4.0:
            self.set("confirm", t, "measuring where the ball rests"); return self.zero()
        return cmd

    def s_confirm(self, t, r, ball, dt):
        """standing back: measure the resting ball again (it was maybe hidden by the robot before)"""
        b = self.ball_still(t)
        if b is None:
            if t - self.t_state > 2.0:                                        # not measured again within 2 s:
                self.remeasured = False                                       # keep the last resting estimate
                return self.end_on_estimate(t, "done")
            return self.zero()
        self.final = b; self.remeasured = True
        if math.hypot(b[0] - self.target[0], b[1] - self.target[1]) > self.tol and self.pushes < self.max_pushes:
            self.set("settle", t, "ball not on the spot after all"); return self.zero()
        return self.stop_with(t, "done")

    def s_lineup(self, t, r, ball, dt):
        p = self.plan; b = p["ball"]; u = p["u"]; S = p["stand"]
        if ball is not None and math.hypot(ball[0] - b[0], ball[1] - b[1]) > 0.03:     # bumped or moved: re-plan
            self.bumps += 1; self.set("settle", t, "ball moved during line-up"); return self.zero()
        if t - self.t_state > 20.0:
            self.set("settle", t); return self.zero()
        x, y, th = r
        R = 0.32; rx, ry = x - b[0], y - b[1]; dr = math.hypot(rx, ry)
        # is the straight way to the stand point clear of the ball?
        sx, sy = S[0] - x, S[1] - y; L2 = sx * sx + sy * sy
        sp = clip(((b[0] - x) * sx + (b[1] - y) * sy) / L2, 0.0, 1.0) if L2 > 1e-6 else 0.0
        clear = math.hypot(x + sp * sx - b[0], y + sp * sy - b[1])
        along = rx * u[0] + ry * u[1]
        goal = S
        if clear < 0.22 and not (along < -0.12 and abs(-rx * u[1] + ry * u[0]) < 0.06):
            phi_r = math.atan2(ry, rx); phi_s = math.atan2(-u[1], -u[0]); dphi = wrap(phi_s - phi_r)
            if self.orbit is None:
                sgn = 1.0 if dphi > 0 else -1.0
                for sg in (sgn, -sgn):        # the side whose arc stays inside the area
                    n = max(2, int(abs(dphi if sg == sgn else dphi - sgn * 2 * math.pi) / 0.2))
                    span = dphi if sg == sgn else dphi - sgn * 2 * math.pi
                    if all(self.allowed(b[0] + R * math.cos(phi_r + span * i / n), b[1] + R * math.sin(phi_r + span * i / n)) for i in range(n + 1)):
                        self.orbit = sg; break
                else:
                    self.orbit = sgn
            step = self.orbit * min(0.6, abs(dphi) if (dphi > 0) == (self.orbit > 0) else 0.6)
            rad = max(R, min(dr, R + 0.1)) if dr > R else R
            goal = (b[0] + rad * math.cos(phi_r + step), b[1] + rad * math.sin(phi_r + step))
            for rr in (R, 0.27, 0.23):        # near a line: a tighter circle, then the nearest point inside
                if self.allowed(*goal): break
                goal = (b[0] + rr * math.cos(phi_r + step), b[1] + rr * math.sin(phi_r + step))
            goal = self.clamp(goal)
        heading = p["ang"] + (self.bias if p["kind"] == "dose" else 0.0)
        cmd, d, eth = self.drive(r, goal, heading, self.vmax, dt)
        if goal is S and d < 0.02 and abs(eth) < 0.04:
            if not hasattr(self, "t_ok") or self.t_ok is None: self.t_ok = t
            if t - self.t_ok > 0.25:
                self.t_ok = None; self.lat = 0.0; self.contact = None; self.heading = heading
                self.set("push", t, f"{p['kind']} push at {p['v']:.2f} m/s"); return self.zero()
        else:
            self.t_ok = None
        return cmd

    def frame(self, r, q):
        c, s = math.cos(r[2]), math.sin(r[2]); dx, dy = q[0] - r[0], q[1] - r[1]
        return c * dx + s * dy, -s * dx + c * dy

    def s_push(self, t, r, ball, dt):
        p = self.plan; b = p["ball"]; u = p["u"]
        x, y, th = r
        along = (x - b[0]) * u[0] + (y - b[1]) * u[1]; lat = -(x - b[0]) * u[1] + (y - b[1]) * u[0]
        moved = ball is not None and math.hypot(ball[0] - b[0], ball[1] - b[1]) > 0.02
        if -along <= HOLD + 0.005 or moved:
            self.t_contact = t; self.b_contact = b
            if p["kind"] == "dose":
                return self.do_release(t, r, self.vf_speed())
            self.set("carry", t, f"{p['kind']}: carrying"); return self.carry_cmd(t, r, ball, dt)
        if t - self.t_state > 5.0 or along > 0.05:
            self.set("settle", t); return self.zero()
        v = p["v"]
        vy_f = clip(-5.0 * lat, -0.25, 0.25)
        want = (u[0] * v - u[1] * vy_f, u[1] * v + u[0] * vy_f)
        self.ramp(want, dt)
        w = clip(3.0 * wrap(self.heading - th), -1.0, 1.0)
        return self.body(th, w) + (w,)

    def vf_speed(self):
        return math.hypot(*self.vf)

    def carry_cmd(self, t, r, ball, dt):
        p = self.plan; u = p["u"]; x, y, th = r
        fw_l = None
        if ball is not None:
            fw, lt = self.frame(r, ball)
            if 0.06 < fw < 0.16 and abs(lt) < 0.11:
                self.lat = 0.6 * self.lat + 0.4 * lt; fw_l = fw
                if abs(lt) > FRONT_HW + 0.01:             # sliding off the front sideways
                    return self.do_release(t, r, self.vf_speed(), why="ball left the front sideways")
            elif abs(lt) > 0.09 and fw > -0.05 and math.hypot(ball[0] - x, ball[1] - y) < 0.2:
                return self.do_release(t, r, self.vf_speed(), why="ball left the front sideways")
            est = ball
        else:
            c, s = math.cos(th), math.sin(th)
            est = (x + c * HOLD - s * self.lat, y + s * HOLD + c * self.lat)
        rem = (p["aim"][0] - est[0]) * u[0] + (p["aim"][1] - est[1]) * u[1]
        v_now = self.vf_speed()
        v_rel = 0.12 if p["kind"] == "creep" else 0.2
        if rem <= self.release_dist(max(v_now, 0.05)) or t - self.t_state > 15.0:
            return self.do_release(t, r, v_now)
        vmax = p["v"] if p["kind"] == "creep" else self.carry
        v = clip(v_rel + 0.7 * (rem - self.release_dist(v_rel)), v_rel, vmax)
        vy = clip(2.0 * self.lat, -0.15, 0.15)                     # keep the ball centred on the front
        e_line = -(est[0] - p["ball"][0]) * u[1] + (est[1] - p["ball"][1]) * u[0]     # the ball left of the line
        corr = clip(-1.5 * e_line, -0.12, 0.12) if p["kind"] != "creep" else 0.0     # steer it back to the line
        a = p["ang"] + corr; ua = (math.cos(a), math.sin(a))
        vy_f = vy
        want = (ua[0] * v - ua[1] * vy_f, ua[1] * v + ua[0] * vy_f)
        self.ramp(want, dt)
        w = clip(3.0 * wrap(self.heading + corr - th), -0.6, 0.6)
        return self.body(th, w) + (w,)

    def s_carry(self, t, r, ball, dt):
        return self.carry_cmd(t, r, ball, dt)

    def do_release(self, t, r, v, why=None):
        p = self.plan; c, s = math.cos(r[2]), math.sin(r[2])
        est = (r[0] + c * HOLD - s * self.lat, r[1] + s * HOLD + c * self.lat)
        self.rel = dict(t=t, at=est, v=v, kind=p["kind"], heading=r[2], pred=self.release_dist(v), why=why,
                        start=p["ball"], aim=p["aim"])
        self.pushes += 1
        self.set("rolling", t, "ball rolling" + (f" ({why})" if why else "")); return self.zero()

    def s_rolling(self, t, r, ball, dt):
        b = self.ball_still(t)
        if b is None and t - self.t_state > 3.0: b = self.ball_still(t, win=1.0, tol=0.02, spread=0.03)
        if b is None or t - self.t_state < 0.6:
            if t - self.t_state > 12.0: self.set("settle", t)
            return self.zero()
        rel = self.rel; u = self.plan["u"]
        travel = (b[0] - rel["at"][0]) * u[0] + (b[1] - rel["at"][1]) * u[1]
        pr = rel["pred"]; ratio = None
        if self.plan.get("careful"): pr = extra_pr = rel["v"] * self.latency + (pr - rel["v"] * self.latency) / 1.3
        extra = rel["v"] * self.latency
        model = pr - extra
        if rel["why"] is None and model > 0.04 and travel > 0.0:
            ratio = clip((travel - extra) / (model / self.gain), 0.4, 2.5)
            self.gain = clip(math.sqrt(self.gain * ratio), 0.6, 1.7)
        if rel["kind"] == "dose":
            dx, dy = b[0] - rel["start"][0], b[1] - rel["start"][1]
            if math.hypot(dx, dy) > 0.3:
                e = wrap(math.atan2(dy, dx) - rel["heading"])
                if abs(e) < math.radians(20): self.bias = clip(0.5 * self.bias - 0.5 * e, -0.17, 0.17)
        self.history.append(dict(kind=rel["kind"], v=round(rel["v"], 3), from_=[round(v, 3) for v in rel["start"]],
                                 rest=[round(v, 3) for v in b], err=round(math.hypot(b[0] - self.target[0], b[1] - self.target[1]), 3),
                                 ratio=round(ratio, 3) if ratio else None, why=rel["why"]))
        self.set("settle", t); return self.zero()


# ---- command-line tool (rclpy) -------------------------------------------------------------------------------
BLOCKERS = procs.COMMANDERS
ancestors = procs.ancestors


def busy_programs(table, mine):
    """names in BLOCKERS run by a process of table [(pid, argv)] outside `mine`, as its executable or as the script
    of an interpreter"""
    return procs.running(BLOCKERS, table, mine)


def positive(name, v):
    """a command-line value that must be a finite positive number -> float, else exit with the reason"""
    try: f = float(v)
    except ValueError: f = float("nan")
    if not (math.isfinite(f) and f > 0): sys.exit(f"{name} must be a finite positive number, not {v!r}")
    return f


class Stop:
    """SIGINT and SIGTERM only set `sig`; the loop checks it and ends the placement itself (end()), then sends zeros
    for 0.3 s (zeros()) while ROS is still up"""
    NAMES = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}
    def __init__(self):
        self.sig = None
    def __call__(self, signum, frame=None):
        if self.sig is None: self.sig = signum
    def install(self, sigmod=signal):
        for s in (sigmod.SIGINT, sigmod.SIGTERM): sigmod.signal(s, self)
    def end(self, pl, t):
        """end the placement when a signal came -> True when it did"""
        if self.sig is None: return False
        if not pl.finished: pl.stop_with(t, "failed: stopped by " + self.NAMES.get(self.sig, str(self.sig)))
        return True
    @staticmethod
    def zeros(send, ok, wait, clock, seconds=0.3, period=0.02):
        """send(0, 0, 0) every `period` s for `seconds` s, as long as ok() (the ROS context is valid) -> how many"""
        t0 = clock(); k = 0
        while ok() and (k == 0 or clock() - t0 < seconds):
            send(0.0, 0.0, 0.0); k += 1; wait(period)
        return k


def init_rclpy(rclpy):
    """rclpy.init without rclpy's own signal handlers when the distribution has the option -> True if it has"""
    try:
        from rclpy.signals import SignalHandlerOptions
    except ImportError:
        rclpy.init(); return False
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO); return True


def main():
    args = sys.argv[1:]
    def opt(name, default):
        if name in args:
            i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
        return default
    precise = "--precise" in args
    if precise: args.remove("--precise")
    tol = positive("--tol", opt("--tol", "0.05" if precise else "0.15")); T = positive("--timeout", opt("--timeout", "60"))
    speed = positive("--speed", opt("--speed", "0.6")); wall = positive("--wall", opt("--wall", "0.30"))
    carry = positive("--carry", opt("--carry", "0.35")); margin = positive("--margin", opt("--margin", "0.25")); mode = opt("--mode", "auto")
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.environ.get("SOLO_DATA") or os.path.join(os.path.dirname(os.path.dirname(here)), "log", "solo")
    out = opt("--out", data)
    watch = opt("--watch", None)
    keep = "--keep-out-defense" in args
    if keep: args.remove("--keep-out-defense")
    if len(args) != 3 or args[0] not in ("blue", "yellow"): sys.exit(__doc__)
    team, tx, ty = args[0], float(args[1]), float(args[2])
    speed = clip(speed, 0.1, 1.0); carry = clip(carry, 0.1, 0.5)

    busy = busy_programs(procs.process_table(), ancestors())
    if busy: sys.exit("refusing: running now: " + ", ".join(busy))

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Pose2D, Twist
    idx = os.environ.get("SOLO_ROBOT_INDEX", "0")
    def rects(spec):
        return [[float(v) for v in r.split(",")] for r in spec.split(";") if r.strip()] if spec else None
    BOX = rects(os.environ.get("SOLO_BOX")) or [[-FIELD_X - margin, FIELD_X + margin, -FIELD_Y - margin, FIELD_Y + margin]]
    inside = lambda x, y: any(r[0] <= x <= r[1] and r[2] <= y <= r[3] for r in BOX)
    init_rclpy(rclpy); n = Node("place_ball")
    stop = Stop(); stop.install()
    pub = n.create_publisher(Twist, os.environ.get("SOLO_CMD_TOPIC", "/cmd_vel"), 10 if os.environ.get("SOLO_CMD_RELIABLE") else qos_profile_sensor_data)
    pose = [None, -1e9]; ball = [None, -1e9]; log = []; cmd = [0.0, 0.0, 0.0]
    T0 = time.monotonic(); T0_WALL = time.time(); now = lambda: time.monotonic() - T0
    def on_pose(m): pose[0], pose[1] = (m.x, m.y, m.theta), now(); log.append((now(), "r", m.x, m.y, m.theta, *cmd))
    def on_ball(m): ball[0], ball[1] = (m.x, m.y), now(); log.append((now(), "b", m.x, m.y, 0.0, *cmd))
    n.create_subscription(Pose2D, f"/vision/poses/{team}/robot{idx}", on_pose, 20)
    n.create_subscription(Pose2D, "/vision/poses/ball", on_ball, 20)
    def send(vx, vy, w):
        cmd[:] = [float(vx), float(vy), float(w)]; m = Twist(); m.linear.x, m.linear.y, m.angular.z = cmd; pub.publish(m)
    def zeros():
        Stop.zeros(send, rclpy.ok, lambda dt: rclpy.spin_once(n, timeout_sec=dt), time.monotonic)
    def save():
        os.makedirs(out, exist_ok=True)
        p = os.path.join(out, f"{time.strftime('%H%M%S')}_place_{tx:+.2f}_{ty:+.2f}.csv")
        with open(p, "w") as f:
            f.write("t,kind,x,y,th,cmd_vx,cmd_vy,cmd_w\n")
            for r in log: f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")
        return p
    b_used = [-1e9]
    pl = BallPlacer((tx, ty), tol=tol, speed=speed, carry=carry, margin=margin, mode=mode, keep_out_defense=keep, timeout=T,
                    wall=wall)
    res = None
    try:
        t_w = now()
        while pose[0] is None and now() - t_w < 6 and stop.sig is None: rclpy.spin_once(n, timeout_sec=0.05)
        if pose[0] is None and not stop.end(pl, now()): res = dict(ok=False, status="failed: robot not in vision")
        while res is None and not pl.finished:
            rclpy.spin_once(n, timeout_sec=0.02)
            if stop.end(pl, now()): break
            t = now()
            rp = pose[0] if t - pose[1] < 0.2 else None
            bp = ball[0] if ball[1] > b_used[0] else None; b_used[0] = ball[1]      # new detections only
            if pose[0] is not None and not inside(pose[0][0], pose[0][1]):
                zeros(); pl.stop_with(t, f"failed: robot left the allowed area at ({pose[0][0]:.2f}, {pose[0][1]:.2f})"); break
            if watch:
                try: age = time.time() - os.path.getmtime(watch)
                except OSError: age = 1e9
                if age > 2.0:
                    zeros(); pl.stop_with(t, f"failed: dead man's file not touched for {min(age, 999):.1f} s"); break
            vx, vy, w, st = pl.step(t, rp, bp)
            send(vx, vy, w) if rp is not None else send(0, 0, 0)
            if pl.finished: break
    except KeyboardInterrupt:                     # only when the handlers could not be installed
        pl.stop_with(now(), "failed: interrupted")
    finally:
        try: zeros()
        except Exception: pass
    res = res or pl.result()
    res["t0_wall"] = round(T0_WALL, 3)
    try: res["log"] = save()
    except OSError as e: res["log"] = None; res["log_error"] = str(e)
    try: n.destroy_node(); rclpy.shutdown()
    except Exception: pass
    print(json.dumps(res))


if __name__ == "__main__":
    main()
