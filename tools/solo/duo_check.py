#!/usr/bin/env python3
"""Which robot listens on which ROS domain: a short turn on the spot is sent on each domain in turn, and vision says
which of our robots turned. Run it with both robots' stacks up and NO strategy node (it commands the robots).
    source tools/solo/env.sh; python3 tools/solo/duo_check.py <team> <id>:<domain> <id>:<domain> [--rate 1.5] [--time 0.7] [--topic /cmd_vel]
e.g. python3 tools/solo/duo_check.py yellow 1:77 3:78
For each pair: a turn at --rate rad/s for --time s on that domain's --topic only, then zeros; PASS when the robot with
that vision id turned (0.4 .. 1.6 x the commanded angle, the commanded way) and every other robot of our colour turned
less than 0.15 rad and moved less than 0.05 m. Any other result means a robot hears the wrong commands (or none): do
not start the strategy. Nothing moves across the field: each robot only turns where it stands.
Our robots are read by vision id: the vision connector must run with <team>_robot_count:=16 (run_duo.sh does it)."""
import math, os, signal, sys, threading, time
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Twist
if os.path.dirname(os.path.abspath(__file__)) not in sys.path: sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import procs                                             # noqa: E402

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
RATE, TIME, TOPIC = float(opt("--rate", "1.5")), float(opt("--time", "0.7")), opt("--topic", "/cmd_vel")
if len(args) < 2 or args[0] not in ("blue", "yellow"):
    sys.exit(__doc__)
TEAM = args[0]
try:
    PAIRS = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in args[1:]]
except (ValueError, IndexError):
    sys.exit("pairs are <vision id>:<ROS domain>, e.g. 1:77 3:78")
if len({d for _, d in PAIRS}) != len(PAIRS) or len({i for i, _ in PAIRS}) != len(PAIRS):
    sys.exit("each id and each domain once")
busy = procs.running(procs.COMMANDERS)
if busy:
    sys.exit(f"refused: {', '.join(busy)} is running (it commands a robot); stop it first")

from rclpy.signals import SignalHandlerOptions                       # noqa: E402
rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
own = int(os.environ.get("ROS_DOMAIN_ID", "0") or 0)
eye = Node("duo_check")
pose = {}
for i in range(16):
    eye.create_subscription(Pose2D, f"/vision/poses/{TEAM}/robot{i}", lambda m, i=i: pose.__setitem__(i, (m.x, m.y, m.theta, time.monotonic())), 10)
pubs, nodes = {}, []
for _, dom in PAIRS:
    if dom == own:
        node = eye
    else:
        ctx = rclpy.Context(); rclpy.init(context=ctx, domain_id=dom, signal_handler_options=SignalHandlerOptions.NO)
        node = Node(f"duo_check_d{dom}", context=ctx); nodes.append((node, ctx))
    pubs[dom] = node.create_publisher(Twist, TOPIC, qos_profile_sensor_data)
stop = [False]
def on_signal(*_): stop[0] = True
signal.signal(signal.SIGINT, on_signal); signal.signal(signal.SIGTERM, on_signal)


def zeros():
    for _ in range(5):
        for p in pubs.values(): p.publish(Twist())
        time.sleep(0.04)


def spin_for(T):
    t0 = time.monotonic()
    while time.monotonic() - t0 < T and not stop[0]:
        rclpy.spin_once(eye, timeout_sec=0.02)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def snapshot():
    t = time.monotonic()
    return {i: p for i, p in pose.items() if t - p[3] < 0.3}


spin_for(1.5)
ours = sorted(snapshot())
print(f"{TEAM} robots in vision now (ids): {ours}", flush=True)
missing = [i for i, _ in PAIRS if i not in ours]
ok_all = not missing
if missing:
    print(f"FAIL  id(s) {missing} not in vision: cannot check them", flush=True)
results = []
try:
    for k, (rid, dom) in enumerate(PAIRS):
        if stop[0] or rid in missing: continue
        sign = 1.0 if k % 2 == 0 else -1.0
        zeros(); spin_for(0.6); before = snapshot()
        m = Twist(); m.angular.z = sign * RATE; t0 = time.monotonic()
        while time.monotonic() - t0 < TIME and not stop[0]:
            pubs[dom].publish(m); spin_for(0.03)
        zeros(); spin_for(1.0); after = snapshot()
        turned = {i: wrap(after[i][2] - before[i][2]) for i in before if i in after}
        moved = {i: math.hypot(after[i][0] - before[i][0], after[i][1] - before[i][1]) for i in before if i in after}
        want = sign * RATE * TIME
        own_ok = rid in turned and 0.4 <= turned[rid] / want <= 1.6
        others_ok = all(abs(turned[i]) < 0.15 and moved[i] < 0.05 for i in turned if i != rid)
        ok = own_ok and others_ok
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  domain {dom} -> id {rid}: commanded {want:+.2f} rad; turned "
              + ", ".join(f"id {i} {turned[i]:+.2f} rad / moved {moved[i]:.2f} m" for i in sorted(turned)), flush=True)
finally:
    zeros()
    for n, ctx in nodes:
        n.destroy_node(); rclpy.shutdown(context=ctx)
print("robots and domains consistent" if ok_all and not stop[0] else "NOT CONSISTENT: do not start the strategy", flush=True)
sys.exit(0 if ok_all and not stop[0] else 1)
