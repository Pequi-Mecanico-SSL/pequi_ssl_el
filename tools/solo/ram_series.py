"""A series of ram tests with a person replacing the ball. Signals by the robot:
    one wiggle  = please put the ball back at the centre (the robot waits at the side)
    one full clockwise spin = ball seen on the spot and still: the test starts in 2 s
    source tools/solo/env.sh; python3 tools/solo/ram_series.py <team> <speed> [<speed> ...]
Results are appended to $SOLO_DATA/results.jsonl (SOLO_DATA default: log/solo in this repository)."""
import json, math, os, subprocess, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Twist

team, speeds = sys.argv[1], [float(a) for a in sys.argv[2:]]
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.environ.get("SOLO_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "log", "solo")), "results.jsonl")
os.makedirs(os.path.dirname(RES), exist_ok=True)
rclpy.init(); n = Node('ram_series'); pub = n.create_publisher(Twist, '/cmd_vel', qos_profile_sensor_data)
ball = [None, 0.0]; hist = []
def on_ball(m):
    ball[0], ball[1] = (m.x, m.y), time.time(); hist.append((time.time(), m.x, m.y)); del hist[:-200]
n.create_subscription(Pose2D, '/vision/poses/ball', on_ball, 20)

def turn(w, T):
    t0 = time.time()
    while time.time() - t0 < T:
        m = Twist(); m.angular.z = float(w); pub.publish(m); rclpy.spin_once(n, timeout_sec=0.03)
def rest(T=0.4):
    t0 = time.time()
    while time.time() - t0 < T:
        pub.publish(Twist()); rclpy.spin_once(n, timeout_sec=0.03)
def wiggle(k):
    for _ in range(k):
        turn(3.0, 0.3); turn(-3.0, 0.6); turn(3.0, 0.3); rest(0.5)
def spin_alarm():
    turn(3.0, 2.1); rest(0.5)
def ball_ready(sx, sy):
    """the ball within 0.12 m of the spot and still (< 1 cm of movement) over the last 2 s"""
    now = time.time(); h = [p for p in hist if now - p[0] < 2.0]
    if len(h) < 60 or now - h[-1][0] > 0.2 or h[-1][0] - h[0][0] < 1.7: return False
    xs = [p[1] for p in h]; ys = [p[2] for p in h]
    return max(xs) - min(xs) < 0.01 and max(ys) - min(ys) < 0.01 and math.hypot(xs[-1] - sx, ys[-1] - sy) < 0.12
def run(*a):
    r = subprocess.run(["python3", f"{HERE}/characterize.py", team, *[str(x) for x in a]], capture_output=True, text=True, timeout=90)
    line = [l for l in r.stdout.splitlines() if l.startswith("{") or l.startswith("ABORT") or "not in vision" in l or "no ball" in l]
    return line[-1] if line else (r.stderr.strip().splitlines() or ["no output"])[-1]

# ball spots: the middles of the four quarter-arcs of the centre circle (radius 0.5 m); the ram goes away from the centre
# along the field's length. The robot asks for the ball from its run-up point 0.75 m before the spot, facing it: the spot
# is where the arc crosses the line straight ahead of the robot.
Q = 0.5 * math.sqrt(0.5)
SPOTS = [(Q, Q, 0.0), (-Q, Q, 180.0), (-Q, -Q, 180.0), (Q, -Q, 0.0)]            # x, y, ram direction (deg)
rest(1.5)
for i, sp in enumerate(speeds):
    sx, sy, dirn = SPOTS[i % 4]
    ux, uy = math.cos(math.radians(dirn)), math.sin(math.radians(dirn))
    out = run("goto", round(sx - 0.75 * ux, 3), round(sy - 0.75 * uy, 3), dirn)
    if not out.startswith("{"): print("goto failed:", out, flush=True); sys.exit(1)
    wiggle(1)
    print(f"[{i + 1}/{len(speeds)}] waiting for the ball on the arc at ({sx:+.2f}, {sy:+.2f}) (ram at {sp} m/s toward {dirn:.0f} deg) ...", flush=True)
    t_ask = time.time()
    while not ball_ready(sx, sy):
        rest(0.2)
        if time.time() - t_ask > 3.0:                  # ask again every 3 s
            wiggle(1); t_ask = time.time()
    turn(-3.0, 2.1); rest(2.0)                        # one full clockwise spin: ball seen, the ram starts in 2 s
    out = run("ram", sp, dirn)
    print(out, flush=True)
    if out.startswith("{"):
        open(RES, "a").write(out + "\n")
    else:
        sys.exit(1)
print("series done", flush=True)
