"""Spin-kick tests on the real robot (built like tools/solo/characterize.py: same topics, publisher, box guard, CSV).
THE ROBOT SPINS FAST AND THROWS THE BALL. Nothing else may publish the command topic.
    source tools/solo/env.sh; SOLO_BOX=x0,x1,y0,y1 python3 tools/solo/spinkick.py <team> <test> [numbers] [--out DIR]
(tools/solo/spinkick_go.sh runs a session of these tests with the boxes and logs set up)
tests:
  sling <rate rad/s> [seconds 0.5] [accel rad/s2 0] [turn_deg 0]
        The robot stays where it stands. It waits (zeros) until the ball rests within 0.14 m of its centre for 1 s
        (a person puts it in the pocket; SPIN_NOWAIT=1: no wait for the ball), then waits for the operator's Enter
        (hands away), holds 2 s more, then spins:
        command = rate (accel 0) or a ramp accel*t up to rate; ends after `seconds`, or as soon as the robot has
        turned turn_deg (turn_deg > 0). Then zeros for 3 s while the ball is tracked.
  carry <speed m/s> <rate rad/s> [carry_m 0.25] [seconds 0.4] [keep 0] [latency_s 0.10]
        The ball rests on the carpet 0.25-1.2 m from the robot. The robot turns to face it, drives through it at
        `speed` (heading and line held as characterize.py's ram), carries it carry_m (at most 0.6 m; contact is
        taken as the robot's centre 0.08 m behind the ball's spot), then spins at `rate` for `seconds`:
        keep 0 = translation command zero during the spin; keep 1 = the field-frame velocity is held (as fspin).
        No spin if the test box guard braked the run. Then zeros for 3 s.
SOLO_BOX (required; several rectangles with ';'): the robot's centre must stay inside; leaving it, or no vision pose
for 0.15 s while a carry runs or spins (0.5 s otherwise), stops the test with zeros. A moving robot is braked early (SOLO_BRAKE m/s2, default 1.5).
Limits built in: |rate| <= 30 rad/s, speed <= 1.5 m/s. Commands go out at 50 Hz.
Every exit path sends zeros (normal end, abort, exception, SIGINT / SIGTERM / SIGHUP) and writes the CSV
t,kind,x,y,th,cmd_vx,cmd_vy,cmd_w,phase (kind r = robot, b = ball) to --out (default $SOLO_DATA/logs; SOLO_DATA defaults
to log/solo in this repository) plus one JSON result line on stdout. Arguments: every number finite; carry: speed
0.1..1.5 m/s, carry_m 0..0.6 m, latency_s 0..0.2 s, keep 1 at most 0.5 s. A carry refuses a ball more than 20 deg off
the robot's heading or outside the test box. Both tests refuse to move while a node other than this one and the robot's
teleop publishes the command topic."""
import json, math, os, select, signal, sys, time

MAX_W, MAX_V, MAX_CARRY, NEAR, SETTLE = 30.0, 1.5, 0.6, 0.14, 3.0
args = sys.argv[1:]
out = os.path.join(os.environ.get("SOLO_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "log", "solo")), "logs")
if "--out" in args:
    i = args.index("--out"); out = args[i + 1]; del args[i:i + 2]
if len(args) < 3 or args[1] not in ("sling", "carry"): sys.exit(__doc__)
team, test, num = args[0], args[1], [float(v) for v in args[2:]]
if not all(math.isfinite(v) for v in num): sys.exit("every number must be finite")       # a NaN would reach the robot
if test == "carry" and len(num) < 2: sys.exit("carry <speed m/s> <rate rad/s> [carry_m seconds keep latency_s]")
def arg(k, default): return num[k] if len(num) > k else default
def rects(spec): return [[float(v) for v in r.split(",")] for r in spec.split(";") if r.strip()]
BOX = rects(os.environ.get("SOLO_BOX", ""))
if not BOX: sys.exit("set SOLO_BOX=x0,x1,y0,y1: the rectangle the robot's centre must stay in")
A_BRAKE = float(os.environ.get("SOLO_BRAKE", "1.5"))
if test == "sling":
    rate, seconds, accel, turn = num[0], arg(1, 0.5), arg(2, 0.0), math.radians(arg(3, 0.0)); speed = 0.0
else:
    speed, rate, carry_m, seconds, keep, lat = num[0], num[1], arg(2, 0.25), arg(3, 0.4), arg(4, 0.0) > 0, arg(5, 0.10)
    if not (0.1 <= speed <= MAX_V and 0.0 <= carry_m <= MAX_CARRY): sys.exit("speed 0.1..1.5 m/s, carry_m 0..0.6 m")
    if not 0.0 <= lat <= 0.2: sys.exit("latency_s 0..0.2")
    if keep and seconds > 0.5: sys.exit("keep 1: seconds at most 0.5 (the speed is held only until the ball leaves)")
if not (0.5 <= abs(rate) <= MAX_W and 0.05 <= seconds <= 3.0): sys.exit("|rate| 0.5..30 rad/s, seconds 0.05..3")

import rclpy                                           # ROS only after the arguments are checked
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Twist
try:
    from rclpy.signals import SignalHandlerOptions     # rclpy without its own signal handlers: ours below stay
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
except ImportError:
    rclpy.init()
IDX = os.environ.get("SOLO_ROBOT_INDEX", "0")
n = Node("spinkick")
pub = n.create_publisher(Twist, os.environ.get("SOLO_CMD_TOPIC", "/cmd_vel"), 10 if os.environ.get("SOLO_CMD_RELIABLE") else qos_profile_sensor_data)
pose = [None, 0.0]; ball = [None, 0.0]; log = []; cmd = [0.0, 0.0, 0.0]; phase = ["wait"]; rot = [0.0]
T0 = time.monotonic(); T0_WALL = time.time()
now = lambda: time.monotonic() - T0
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
clip = lambda v, m: max(-m, min(m, v))
inside = lambda x, y: any(r[0] <= x <= r[1] and r[2] <= y <= r[3] for r in BOX)
def on_pose(m):
    if pose[0] is not None: rot[0] += wrap(m.theta - pose[0][2])          # total turn since the start (unwrapped)
    pose[0], pose[1] = (m.x, m.y, m.theta), now(); log.append((now(), "r", m.x, m.y, m.theta, *cmd, phase[0]))
def on_ball(m):
    ball[0], ball[1] = (m.x, m.y), now(); log.append((now(), "b", m.x, m.y, 0.0, *cmd, phase[0]))
n.create_subscription(Pose2D, f"/vision/poses/{team}/robot{IDX}", on_pose, 20)
n.create_subscription(Pose2D, "/vision/poses/ball", on_ball, 20)

def send(vx, vy, w):
    vx, vy, w = float(vx), float(vy), float(w); cmd[:] = [vx, vy, w]
    m = Twist(); m.linear.x, m.linear.y, m.angular.z = vx, vy, w; pub.publish(m)
def zeros(k=10):
    for _ in range(k):
        try: send(0, 0, 0); time.sleep(0.02)
        except Exception: break
def save():
    os.makedirs(out, exist_ok=True)
    p = f"{out}/{time.strftime('%H%M%S')}_{test}_{'_'.join(f'{v:g}' for v in num)}.csv"
    with open(p, "w") as f:
        f.write("t,kind,x,y,th,cmd_vx,cmd_vy,cmd_w,phase\n")
        for r in log: f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")
    return p
res = dict(test=test, args=num, ok=False)
ENDED = []
def end(why, code):
    """the one way out: zeros, the log, the result line, exit"""
    if ENDED: return
    ENDED.append(1); zeros(); p = None
    try: p = save()
    except Exception: pass
    res.update(why=why, t0_wall=round(T0_WALL, 3), log=p)
    try: print(json.dumps(res), flush=True)
    except Exception: pass
    os._exit(code)
signal.signal(signal.SIGINT, lambda *_: end("interrupted", 130)); signal.signal(signal.SIGTERM, lambda *_: end("interrupted", 130))
signal.signal(signal.SIGHUP, lambda *_: end("interrupted (hangup)", 129))      # a closed terminal also sends zeros
TOPIC = os.environ.get("SOLO_CMD_TOPIC", "/cmd_vel")
def others_publishing():
    """refuse when another node publishes the command topic (teleop on the robot is allowed: it is silent without
    a joystick). Discovery needs about a second; a check that cannot run says so and leaves the decision to the caller's process check."""
    try: names = sorted({i.node_name for i in n.get_publishers_info_by_topic(TOPIC)} - {"spinkick", "teleop"})
    except Exception as ex: print(f"WARNING: publisher check unavailable ({type(ex).__name__})", flush=True); return
    if names: end(f"REFUSED: other publishers on {TOPIC}: {names}", 1)
STALE = [0.5]                                     # pose age that aborts; 0.15 s while the robot translates

def tick(dt=0.005):
    rclpy.spin_once(n, timeout_sec=dt)
    if pose[0] is not None and now() - pose[1] > STALE[0]: end(f"ABORT: vision lost the robot for {STALE[0]} s (last at {pose[0]})", 1)
    if pose[0] is not None and not inside(pose[0][0], pose[0][1]): end(f"ABORT: robot at the edge of the test box ({pose[0]})", 1)
NEXT = [0.0]
def drive(vx, vy, w):
    """tick, and send the command at 50 Hz"""
    tick()
    if now() >= NEXT[0]: send(vx, vy, w); NEXT[0] = now() + 0.02
def hold(T):
    t = now()
    while now() - t < T: drive(0, 0, 0)
def must_brake():
    """True when the robot, at its present velocity, needs the rest of the test box to stop (as characterize.py)"""
    rs = [r for r in log[-40:] if r[1] == "r"]
    if len(rs) < 5: return False
    a, b = rs[-5], rs[-1]; dt = max(b[0] - a[0], 1e-3); vx, vy = (b[2] - a[2]) / dt, (b[3] - a[3]) / dt; v = math.hypot(vx, vy)
    if v < 0.15: return False
    need = v * v / (2 * A_BRAKE) + 0.25 * v + 0.08; d = 0.0
    while d < need:
        d += 0.01
        if not inside(b[2] + d * vx / v, b[3] + d * vy / v): return True
    return False
def ball_resting_near():
    bs = [r for r in log[-200:] if r[1] == "b" and now() - r[0] < 1.0]
    if len(bs) < 30 or now() - bs[-1][0] > 0.2: return False
    xs = [r[2] for r in bs]; ys = [r[3] for r in bs]
    return max(xs) - min(xs) < 0.006 and max(ys) - min(ys) < 0.006 and math.hypot(xs[-1] - pose[0][0], ys[-1] - pose[0][1]) < NEAR

def main():
    t = now()
    while (pose[0] is None or now() - t < 1.0) and now() - t < 6: rclpy.spin_once(n, timeout_sec=0.05)
    if pose[0] is None: end("robot not in vision", 1)
    others_publishing()
    if not inside(pose[0][0], pose[0][1]): end(f"ABORT: robot starts outside the test box ({pose[0]})", 1)
    sgn = 1.0 if rate > 0 else -1.0
    if test == "sling":
        if os.environ.get("SPIN_NOWAIT"): hold(1.0)
        else:
            print("put the ball in the pocket; when it rests the tool asks for Enter", flush=True); t = now()
            while not ball_resting_near():
                drive(0, 0, 0)
                if now() - t > 90: end("no resting ball near the robot within 90 s", 1)
        print("ball resting: hands away, then Enter (Ctrl-C = no spin)", flush=True); t = now()      # the operator's go
        try:
            import termios; termios.tcflush(sys.stdin, termios.TCIFLUSH)                     # an Enter typed earlier does not count
        except Exception: pass
        while True:
            drive(0, 0, 0)
            if now() - t > 60: end("no Enter within 60 s", 1)
            if select.select([sys.stdin], [], [], 0)[0]:
                if not sys.stdin.readline(): end("no operator at the terminal (stdin closed)", 1)
                break
        others_publishing()
        print("SPIN IN 2 s: hands away", flush=True); hold(2.0)
        res.update(start=[round(v, 4) for v in pose[0]], ball_start=[round(v, 4) for v in ball[0]] if ball[0] and now() - ball[1] < 0.3 else None)
    else:
        if ball[0] is None or now() - ball[1] > 0.5: end("no ball in vision", 1)
        (x, y, th), (bx, by) = pose[0], ball[0]; d = math.hypot(bx - x, by - y)
        if not 0.25 <= d <= 1.2: end(f"ball {d:.2f} m from the robot: must be 0.25..1.2 m", 1)
        ang = math.atan2(by - y, bx - x); ux, uy = math.cos(ang), math.sin(ang)
        if abs(wrap(ang - th)) > math.radians(20) or not inside(bx, by):                     # only the run line the operator set up
            end(f"ball at bearing {math.degrees(wrap(ang - th)):.0f} deg from the heading (at most 20) or outside the test box", 1)
        others_publishing(); t = now(); phase[0] = "turn"
        while now() - t < 5 and abs(wrap(ang - pose[0][2])) > 0.05: drive(0, 0, clip(3.0 * wrap(ang - pose[0][2]), 2.0))
        hold(0.5)
        along = lambda px, py: (px - bx) * ux + (py - by) * uy            # along the run line, 0 at the ball's spot
        across = lambda px, py: -(px - bx) * uy + (py - by) * ux
        res.update(start=[round(v, 4) for v in pose[0]], ball_start=[round(bx, 4), round(by, 4)], dir_deg=round(math.degrees(ang), 1))
        others_publishing()                                               # again after the turn: more discovery time
        phase[0] = "run"; t = now(); braked = False; STALE[0] = 0.15
        while along(pose[0][0], pose[0][1]) < -0.08 + carry_m:           # contact = centre 0.08 m behind the spot
            if now() - t > 4.0 or must_brake(): braked = True; break
            eh = wrap(ang - pose[0][2]); e = -across(pose[0][0], pose[0][1])
            drive(speed, clip(6.0 * e, 0.5) if abs(eh) < 0.4 else 0.0, clip(5.0 * eh, 2.0))
        if braked: end("run braked by the test box guard or timed out: no spin", 1)
    phase[0] = "spin"; rot0 = rot[0]; ts = now(); res["t_spin_on"] = round(ts, 4)
    while now() - ts < seconds:
        if test == "sling":
            if turn > 0 and abs(rot[0] - rot0) >= turn: break
            drive(0, 0, sgn * min(abs(rate), accel * (now() - ts)) if accel > 0 else rate)
        elif keep:
            if must_brake(): res["spin_cut"] = "box guard"; break
            thp = pose[0][2] + rate * lat; c, s = math.cos(thp), math.sin(thp)       # field velocity held, as fspin
            drive(c * speed * ux + s * speed * uy, -s * speed * ux + c * speed * uy, rate)
        else:
            if test == "carry" and must_brake(): res["spin_cut"] = "box guard"; break        # as keep 1
            drive(0, 0, rate)
    send(0, 0, 0); STALE[0] = 0.5; phase[0] = "after"; res.update(t_spin_off=round(now(), 4), turned_deg=round(math.degrees(rot[0] - rot0), 1))
    hold(SETTLE)
    res.update(ok=True, end=[round(v, 4) for v in pose[0]], ball_end=[round(v, 3) for v in ball[0]] if ball[0] else None)
    end("done", 0)

try:
    main()
except SystemExit:
    raise
except BaseException as ex:                             # any error: zeros, the log, a result line
    end(f"ERROR: {type(ex).__name__}: {ex}", 1)
