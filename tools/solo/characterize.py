"""Characterize the real robot with vision: step responses (forward, sideways, turn), and a ram of the ball
(ball speed off the front, rolling deceleration). The strategy node must be paused.
    source tools/solo/env.sh; python3 tools/solo/characterize.py <team> <test> [args] [--out DIR]
tests:  step fwd|side <speed m/s> [x0 y0 dir_deg length]
                                      from (x0, y0) along the direction dir_deg, command off after `length` m or 3 s
                                      (default: from x = -1.3 on the lane y = LANE along +x, 2.0 m)
        spin <rate rad/s> [seconds]   turn on the spot (default 1.5 s)
        ram <speed m/s> [dir_deg runup past]
                                      from `runup` m behind the ball (0.75), straight through it along dir_deg, command
                                      off `past` m beyond the ball's spot (0.25)
        goto x y th_deg
        sweep x0 x1 y0 y1 rows [speed]
                                      rows along x from y0 to y1 (a coverage pass; 0.5 m/s)
Every test logs t, pose, ball and command to a CSV in --out (default $SOLO_DATA/logs; SOLO_DATA
defaults to log/solo in this repository; created if missing) and prints a summary.
A small test area: SOLO_BOX="x0,x1,y0,y1" (m) is the rectangle the robot's centre must stay in. Leaving it aborts the
test, and a moving robot is braked early enough to stop inside it, assuming it can brake at SOLO_BRAKE m/s2 (1.5).
SOLO_BOX may list several rectangles separated by ';' (the area is their union). SOLO_BRAKE_BOX (same format,
default SOLO_BOX) is the area the early braking keeps a running test in, e.g. the part of the floor of one surface.
SOLO_SETTLE (s) overrides the time the robot is held still after a step's command (1.5).
SIGINT / SIGTERM: zero velocity for 0.2 s, the log saved, exit 130."""
import json, math, sys, time
import os
import rclpy
IDX = os.environ.get('SOLO_ROBOT_INDEX', '0')     # our robot's index among its colour in the vision connector
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Twist

args = [a for a in sys.argv[1:]]
out = os.path.join(os.environ.get("SOLO_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "log", "solo")), "logs")
if "--out" in args:
    i = args.index("--out"); out = args[i + 1]; del args[i:i + 2]
os.makedirs(out, exist_ok=True)
team, test, rest = args[0], args[1], args[2:]
LANE = 0.75
def rects(spec):
    return [[float(v) for v in r.split(",")] for r in spec.split(";") if r.strip()] if spec else None
BOX = rects(os.environ.get("SOLO_BOX"))
AREA = BOX or [[-2.0, 2.0, -1.35, 1.35]]
BRAKE_AREA = rects(os.environ.get("SOLO_BRAKE_BOX")) or BOX
SETTLE = float(os.environ.get("SOLO_SETTLE", "1.5"))
def inside(x, y, area=None):
    return any(r[0] <= x <= r[1] and r[2] <= y <= r[3] for r in (area or AREA))
A_BRAKE = float(os.environ.get("SOLO_BRAKE", "1.5"))
RAMP = float(os.environ.get("SOLO_RAMP", "0"))     # m/s2: ramp the commanded speed of step tests instead of a jump (0 = jump)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from place_ball import init_rclpy  # noqa: E402  rclpy without its own signal handlers when the option exists: ours below stays
init_rclpy(rclpy); n = Node('characterize'); pub = n.create_publisher(Twist, os.environ.get('SOLO_CMD_TOPIC', '/cmd_vel'), 10 if os.environ.get('SOLO_CMD_RELIABLE') else qos_profile_sensor_data)
pose = [None, 0.0]; ball = [None, 0.0]; log = []; cmd = [0.0, 0.0, 0.0]
T0 = time.monotonic(); T0_WALL = time.time()          # log times are seconds after T0 (wall clock T0_WALL)
now = lambda: time.monotonic() - T0
def on_pose(m):
    pose[0], pose[1] = (m.x, m.y, m.theta), now(); log.append((now(), 'r', m.x, m.y, m.theta, *cmd))
def on_ball(m):
    ball[0], ball[1] = (m.x, m.y), now(); log.append((now(), 'b', m.x, m.y, 0.0, *cmd))
n.create_subscription(Pose2D, f'/vision/poses/{team}/robot{IDX}', on_pose, 20)
n.create_subscription(Pose2D, '/vision/poses/ball', on_ball, 20)
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi

def send(vx, vy, w):
    vx, vy, w = float(vx), float(vy), float(w); cmd[:] = [vx, vy, w]; m = Twist(); m.linear.x, m.linear.y, m.angular.z = vx, vy, w; pub.publish(m)

STOPPING = []
def on_stop(*_):
    """SIGINT / SIGTERM (a supervisor stopping the test, Ctrl-C): zeros, the log, a result line marked interrupted, exit"""
    if STOPPING: return                                # a second signal (the group's and the one series.py forwards): once
    STOPPING.append(1)
    for _ in range(10):
        try: send(0, 0, 0); time.sleep(0.02)
        except Exception: break
    p = None
    try: p = save()
    except Exception: pass
    try: print(json.dumps(dict(test=test, args=rest, interrupted=True, signal=_[0] if _ else None, log=p)), flush=True)
    except Exception: pass
    os._exit(130)
import signal
signal.signal(signal.SIGINT, on_stop); signal.signal(signal.SIGTERM, on_stop)

def tick(dt=0.02):
    rclpy.spin_once(n, timeout_sec=dt)
    if pose[0] is not None and (now() - pose[1] > 0.5 or not inside(pose[0][0], pose[0][1])):
        why = "vision lost the robot" if now() - pose[1] > 0.5 else "robot at the edge of the allowed area"
        for _ in range(10): send(0, 0, 0); rclpy.spin_once(n, timeout_sec=0.02)
        save(); sys.exit(f"ABORT: {why} (last at {pose[0]})")

def save():
    p = f"{out}/{time.strftime('%H%M%S')}_{test}_{'_'.join(rest)}.csv"
    with open(p, "w") as f:
        f.write("t,kind,x,y,th,cmd_vx,cmd_vy,cmd_w\n")
        for r in log: f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")
    return p

def goto(tx, ty, tth, vmax=0.6, tol=0.03, T=25, keep_ball=0.35):
    """turn on the spot to the heading tth, then drive to (tx, ty) holding it; round the ball by a waypoint when the
    straight line passes within keep_ball of it."""
    t0 = now(); wl = 0.0
    while now() - t0 < 5:                                   # 1. turn on the spot
        tick()
        if now() - pose[1] > 0.2: send(0, 0, 0); continue
        eth = wrap(tth - pose[0][2])
        if abs(eth) < 0.08: break
        send(0, 0, max(-2.5, min(2.5, 3.0 * eth)))
    for _ in range(8): send(0, 0, 0); tick()
    pts = [(tx, ty)]
    if ball[0] is not None and now() - ball[1] < 1.0:       # 2. a waypoint round the ball
        x, y, _ = pose[0]; bx, by = ball[0]; dx, dy = tx - x, ty - y; L2 = dx * dx + dy * dy
        if L2 > 1e-4 and math.hypot(tx - bx, ty - by) > keep_ball:
            sp = max(0.0, min(1.0, ((bx - x) * dx + (by - y) * dy) / L2)); px, py = x + sp * dx, y + sp * dy
            dist = math.hypot(px - bx, py - by)
            if dist < keep_ball and 0.0 < sp < 1.0:
                L = math.sqrt(L2); nx, ny = ((px - bx) / dist, (py - by) / dist) if dist > 0.02 else (-dy / L, dx / L)
                pts.insert(0, (bx + nx * (keep_ball + 0.2), by + ny * (keep_ball + 0.2)))
    for k, (wx, wy) in enumerate(pts):                      # 3. drive, holding the heading
        last = k == len(pts) - 1
        while now() - t0 < T:
            tick()
            if now() - pose[1] > 0.2:                     # no fresh pose: stand still rather than drive blind
                send(0, 0, 0); wl = 0.0; continue
            x, y, th = pose[0]; d = math.hypot(wx - x, wy - y); eth = wrap(tth - th)
            if d < (tol if last else 0.12) and (abs(eth) < 0.05 or not last): break
            sp = min(vmax, 1.8 * d + 0.03); fx, fy = ((wx - x) / d * sp, (wy - y) / d * sp) if d > 1e-3 else (0.0, 0.0)
            if ball[0] is not None and now() - ball[1] < 1.0:
                bx, by = ball[0]; od = math.hypot(bx - x, by - y); gap = od - 0.25
                if 1e-3 < od and gap < 0.3:
                    nx, ny = (bx - x) / od, (by - y) / od; closing = fx * nx + fy * ny
                    if closing > 1.5 * gap: ex = closing - 1.5 * gap; fx -= nx * ex; fy -= ny * ex
            wl = max(-1.2, min(1.2, 2.5 * eth))
            thp = th + wl * 0.1                             # the heading it will have when the command acts
            c, s = math.cos(thp), math.sin(thp)
            send(c * fx + s * fy, -s * fx + c * fy, wl)
    for _ in range(15): send(0, 0, 0); tick()
    return pose[0]

def vel_now(win=0.06):
    """(vx, vy) of the robot in the field frame from the vision samples of the last `win` s"""
    rs = [r for r in log[-40:] if r[1] == 'r']
    if len(rs) < 2: return 0.0, 0.0
    a = next((r for r in rs if rs[-1][0] - r[0] <= win), rs[-2]); b = rs[-1]
    if a is b: a = rs[-2]
    dt = max(b[0] - a[0], 1e-3)
    return (b[2] - a[2]) / dt, (b[3] - a[3]) / dt

def must_brake():
    """in a small test area: True when the robot, at its present velocity, needs the rest of the area to stop"""
    if BOX is None or pose[0] is None: return False
    vx, vy = vel_now(); v = math.hypot(vx, vy)
    if v < 0.15: return False
    x, y, _ = pose[0]; ux, uy = vx / v, vy / v
    need = v * v / (2 * A_BRAKE) + 0.25 * v + 0.08       # braking distance + 0.25 s of reaction + margin
    # free distance along the velocity inside the brake area; a start a few cm outside it (goto tolerance) is
    # allowed when the robot is moving into it
    d = next((k * 0.01 for k in range(6) if inside(x + k * 0.01 * ux, y + k * 0.01 * uy, BRAKE_AREA)), None)
    if d is None: return True
    while d < need and inside(x + (d + 0.01) * ux, y + (d + 0.01) * uy, BRAKE_AREA): d += 0.01
    return d < need

def run(vx, vy, w, T, stop_if=lambda: False, settle=1.2):
    """open loop: the command for T s (or until stop_if), then zero for settle s. -> (t_start, t_stop)"""
    ts = now(); v = math.hypot(vx, vy)
    while now() - ts < T and not stop_if():
        k = min(1.0, RAMP * (now() - ts) / v) if RAMP > 0 and v > 1e-6 else 1.0      # SOLO_RAMP: raise the command at this rate
        send(k * vx, k * vy, w); tick()
    te = now()
    while now() - te < settle:
        send(0, 0, 0); tick()
    return ts, te

def series(kind, key, t_from, t_to):
    """[(t, value)] of vision samples between the times; key picks the value from (x, y, th)"""
    return [(r[0], key(r[2], r[3], r[4])) for r in log if r[1] == kind and t_from <= r[0] <= t_to]

def speed_profile(s, win=0.06):
    """central-difference velocity [(t, v)] over a window of about `win` s"""
    v = []
    for i in range(len(s)):
        j = i
        while j + 1 < len(s) and s[j + 1][0] - s[i][0] < win: j += 1
        if j > i and s[j][0] - s[i][0] > 0.6 * win:
            v.append((0.5 * (s[i][0] + s[j][0]), (s[j][1] - s[i][1]) / (s[j][0] - s[i][0])))
    return v

def first_cross(v, level, t_min):
    for t, x in v:
        if t >= t_min and x >= level: return t
    return float('nan')

t0 = now()
while (pose[0] is None or now() - t0 < 1.0) and now() - t0 < 6: rclpy.spin_once(n, timeout_sec=0.05)
if pose[0] is None: sys.exit("robot not in vision")
res = dict(test=test, args=rest)
if test == "goto":
    p = goto(float(rest[0]), float(rest[1]), math.radians(float(rest[2])))
    res["end"] = [round(v, 3) for v in p]
elif test == "sweep":                        # sweep x0 x1 y0 y1 rows [speed]: where does vision see the robot
    x0, x1, y0, y1 = (float(v) for v in rest[:4]); rows = int(rest[4]); vm = float(rest[5]) if len(rest) > 5 else 0.5
    n0 = len(log); t_s = now()
    for k in range(rows):
        y = y0 + (y1 - y0) * k / max(rows - 1, 1)
        a_, b_ = (x0, x1) if k % 2 == 0 else (x1, x0)
        goto(a_, y, 0.0, vmax=vm, tol=0.08, T=12); goto(b_, y, 0.0, vmax=vm, tol=0.08, T=12)
    rs = [r for r in log[n0:] if r[1] == 'r']; gaps = [b_[0] - a_[0] for a_, b_ in zip(rs, rs[1:])]
    res.update(seconds=round(now() - t_s, 1), samples=len(rs), rate_hz=round(len(rs) / max(now() - t_s, 1e-3), 1),
               gaps_over_100ms=sum(1 for g in gaps if g > 0.1), longest_gap_ms=round(1000 * max(gaps, default=0)))
elif test == "step":
    axis, sp = rest[0], float(rest[1])
    x0, y0, dirn, length = (float(v) for v in rest[2:6]) if len(rest) >= 6 else (-1.3, LANE, 0.0, 2.0)
    dirn = math.radians(dirn); ux, uy = math.cos(dirn), math.sin(dirn)
    th0 = dirn if axis == "fwd" else dirn - math.pi / 2      # body x (fwd) or body +y (side) along the run direction
    goto(x0, y0, th0)
    for _ in range(25): send(0, 0, 0); tick()
    p0 = pose[0]
    along = lambda x, y: (x - p0[0]) * ux + (y - p0[1]) * uy
    early = [False]
    def stop():
        if must_brake(): early[0] = True
        return early[0] or along(pose[0][0], pose[0][1]) > length
    ts, te = run(sp if axis == "fwd" else 0.0, sp if axis == "side" else 0.0, 0.0, 3.0 if len(rest) >= 6 else 2.2, stop_if=stop, settle=SETTLE)
    p1 = pose[0]
    xs = series('r', lambda x, y, th: along(x, y), ts - 0.3, te + SETTLE); v = speed_profile(xs)
    plateau = [x for t, x in v if ts + 0.6 * (te - ts) <= t <= te]
    vss = sum(plateau) / max(len(plateau), 1)
    t10, t50, t90 = (first_cross(v, f * vss, ts) for f in (0.1, 0.5, 0.9))
    vb = [(t, x) for t, x in v if t >= te]
    tb90 = next((t for t, x in vb if x <= 0.9 * vss), float('nan')); tb10 = next((t for t, x in vb if x <= 0.1 * vss), float('nan'))
    res.update(cmd_speed=sp, steady_speed=round(vss, 3), ratio=round(vss / sp, 3), peak_speed=round(max(x for _, x in v), 3),
               delay_to_10pct_ms=round((t10 - ts) * 1000), accel_10_90=round(0.8 * vss / (t90 - t10), 2) if t90 > t10 else None,
               brake_delay_ms=round((tb90 - te) * 1000), decel_90_10=round(0.8 * vss / (tb10 - tb90), 2) if tb10 > tb90 else None,
               run_time=round(te - ts, 2), travelled=round(along(p1[0], p1[1]), 3),
               stop_dist_after_zero=round(along(p1[0], p1[1]) - next((x for t, x in xs if t >= te), along(p1[0], p1[1])), 3),
               side_drift_m=round(-(p1[0] - p0[0]) * uy + (p1[1] - p0[1]) * ux, 3), heading_drift_deg=round(math.degrees(wrap(p1[2] - p0[2])), 1),
               speed_at_command_off=round(next((x for t, x in v if t >= te), float('nan')), 3), braked_early=early[0],
               start=[round(p0[0], 3), round(p0[1], 3)], end=[round(p1[0], 3), round(p1[1], 3)], brake_assumed=A_BRAKE,
               t_cmd_on=round(ts, 3), t_cmd_off=round(te, 3))
elif test == "spin":
    w = float(rest[0])
    for _ in range(25): send(0, 0, 0); tick()
    p0 = pose[0]; ts, te = run(0.0, 0.0, w, float(rest[1]) if len(rest) > 1 else 1.5); p1 = pose[0]
    raw = series('r', lambda x, y, th: th, ts - 0.3, te + 1.2); un = [raw[0]]
    for t, a in raw[1:]: un.append((t, un[-1][1] + wrap(a - un[-1][1])))
    v = speed_profile(un); sgn = 1.0 if w > 0 else -1.0; v = [(t, sgn * x) for t, x in v]
    plateau = [x for t, x in v if ts + 0.5 * (te - ts) <= t <= te]; wss = sum(plateau) / max(len(plateau), 1)
    t10, t90 = first_cross(v, 0.1 * wss, ts), first_cross(v, 0.9 * wss, ts)
    res.update(cmd_rate=w, steady_rate=round(wss, 2), ratio=round(wss / abs(w), 3), delay_to_10pct_ms=round((t10 - ts) * 1000),
               ang_accel_10_90=round(0.8 * wss / (t90 - t10), 1) if t90 > t10 else None,
               centre_drift_m=round(math.hypot(p1[0] - p0[0], p1[1] - p0[1]), 3))
elif test == "ram":
    sp = float(rest[0]); ang = math.radians(float(rest[1])) if len(rest) > 1 else 0.0     # ram direction in the field frame
    runup = float(rest[2]) if len(rest) > 2 else 0.75; past = float(rest[3]) if len(rest) > 3 else 0.25
    if ball[0] is None: sys.exit("no ball in vision")
    ux, uy = math.cos(ang), math.sin(ang)
    bx, by = ball[0]
    sx, sy = bx - runup * ux, by - runup * uy
    if not inside(sx, sy): sys.exit(f"ABORT: the run-up point ({sx:.2f}, {sy:.2f}) is outside the allowed area")
    if math.hypot(pose[0][0] - sx, pose[0][1] - sy) > 0.06 or abs(wrap(ang - pose[0][2])) > 0.1:
        goto(sx, sy, ang)
    for _ in range(25): send(0, 0, 0); tick()
    b0 = ball[0]; p0 = pose[0]
    along = lambda x, y: (x - b0[0]) * ux + (y - b0[1]) * uy            # coordinate along the ram line, 0 at the ball's spot
    across = lambda x, y: -(x - b0[0]) * uy + (y - b0[1]) * ux
    ts = now()
    while now() - ts < 3.0 and along(pose[0][0], pose[0][1]) < past and not must_brake():    # straight through the ball's spot, holding the line
        eh = wrap(ang - pose[0][2]); e = -across(pose[0][0], pose[0][1])
        k = min(1.0, RAMP * (now() - ts) / sp) if RAMP > 0 and sp > 1e-6 else 1.0     # SOLO_RAMP: raise the command at this rate
        send(k * sp, max(-0.5, min(0.5, 6.0 * e)) if abs(eh) < 0.4 else 0.0, max(-3.0, min(3.0, 5.0 * eh))); tick()
    te = now()
    while now() - te < (2.5 if BOX else 5.0):
        send(0, 0, 0); tick()
    rx = speed_profile(series('r', lambda x, y, th: along(x, y), ts, te + 0.5)); bxs = series('b', lambda x, y, th: along(x, y), ts, te + 5.0)
    bys = series('b', lambda x, y, th: across(x, y), ts, te + 5.0); bv = speed_profile(bxs, win=0.08)
    t_hit = next((t for t, x in bxs if x > 0.02), float('nan'))
    v_rob = max((x for t, x in rx if t <= t_hit + 0.02), default=float('nan'))
    after = [(t, x) for t, x in bv if t > t_hit + 0.05]
    v_ball = max((x for t, x in after[:12]), default=float('nan'))
    # rolling deceleration: least-squares slope of the ball speed between 0.3 s after its peak and 0.25 m/s
    tpk = next((t for t, x in after if x == v_ball), t_hit)
    seg = [(t, x) for t, x in after if t > tpk + 0.3 and x > 0.25]
    dec = float('nan')
    if len(seg) > 8:
        mt = sum(t for t, _ in seg) / len(seg); mx = sum(x for _, x in seg) / len(seg)
        dec = -sum((t - mt) * (x - mx) for t, x in seg) / sum((t - mt) ** 2 for t, _ in seg)
    res.update(cmd_speed=sp, dir_deg=round(math.degrees(ang)), runup=runup, past=past, ball_start=[round(b0[0], 3), round(b0[1], 3)],
               robot_speed_at_contact=round(v_rob, 3), ball_speed_after=round(v_ball, 3),
               ball_over_robot=round(v_ball / v_rob, 2) if v_rob > 0.1 else None, ball_roll_decel=round(dec, 3),
               ball_travel=round(bxs[-1][1], 2) if bxs else None, ball_side_drift=round(bys[-1][1], 2) if bys else None,
               ball_end=[round(ball[0][0], 2), round(ball[0][1], 2)], line_error_at_contact_m=round(across(pose[0][0], pose[0][1]), 3))
for _ in range(10): send(0, 0, 0); tick()
res["t0_wall"] = round(T0_WALL, 3); res["log"] = save()
print(json.dumps(res))
