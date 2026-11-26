"""Drive our robot to a field point using vision, going round the other team's robots and the ball
(the strategy node must be paused: both publish /cmd_vel).
    source tools/solo/env.sh; python3 tools/solo/goto.py <blue|yellow> x y [--v 0.5]
Stops when vision loses the robot for 0.4 s, on arrival, or after 25 s. Prints the smallest distance to each obstacle."""
import math, sys, time
import os
import rclpy
IDX = os.environ.get('SOLO_ROBOT_INDEX', '0')     # our robot's index among its colour in the vision connector
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Twist
team = sys.argv[1]; tx, ty = float(sys.argv[2]), float(sys.argv[3])
vmax = float(sys.argv[sys.argv.index("--v") + 1]) if "--v" in sys.argv else 0.5
other = 'yellow' if team == 'blue' else 'blue'
KEEP = {'robot': 0.32, 'ball': 0.25}                     # centre distances kept (robot radius 0.09, ball 0.02)
rclpy.init(); n = Node('goto'); pub = n.create_publisher(Twist, '/cmd_vel', qos_profile_sensor_data); pose = [None, 0.0]; obs = {}
n.create_subscription(Pose2D, f'/vision/poses/{team}/robot{IDX}', lambda m: pose.__setitem__(0, (m.x, m.y, m.theta)) or pose.__setitem__(1, time.time()), 10)
for i in range(3):
    n.create_subscription(Pose2D, f'/vision/poses/{other}/robot{i}', lambda m, i=i: obs.__setitem__(f'{other}{i}', (m.x, m.y, 'robot', time.time())), 10)
n.create_subscription(Pose2D, '/vision/poses/ball', lambda m: obs.__setitem__('ball', (m.x, m.y, 'ball', time.time())), 10)
t0 = time.time()
while (pose[0] is None or time.time() - t0 < 1.0) and time.time() - t0 < 5: rclpy.spin_once(n, timeout_sec=0.05)
if pose[0] is None: sys.exit("robot not in vision")
print(f"from ({pose[0][0]:+.2f}, {pose[0][1]:+.2f}) to ({tx:+.2f}, {ty:+.2f}); obstacles: " + ", ".join(f"{k} ({v[0]:+.2f}, {v[1]:+.2f})" for k, v in sorted(obs.items())))
t0 = time.time(); why = "timeout"; closest = {}
while time.time() - t0 < 25:
    rclpy.spin_once(n, timeout_sec=0.03)
    now = time.time()
    if now - pose[1] > 0.4: why = "vision lost the robot"; break
    x, y, th = pose[0]; d = math.hypot(tx - x, ty - y)
    if d < 0.04: why = "arrived"; break
    sp = min(vmax, 1.5 * d + 0.05); vx, vy = (tx - x) / d * sp, (ty - y) / d * sp
    for k, (ox, oy, kind, t) in obs.items():
        if now - t > 1.0: continue
        od = math.hypot(ox - x, oy - y); closest[k] = min(closest.get(k, 9), od)
        gap = od - KEEP[kind]
        if gap > 0.45 or od < 1e-3: continue
        nx, ny = (ox - x) / od, (oy - y) / od
        closing = vx * nx + vy * ny; lim = 1.5 * gap                 # allowed closing speed: 0 at the kept distance
        if closing > lim:
            ex = closing - lim; vx -= nx * ex; vy -= ny * ex
            sx, sy = -ny, nx                                         # sidestep on the target's side
            if sx * (tx - x) + sy * (ty - y) < 0: sx, sy = -sx, -sy
            k2 = min(0.9 * ex, vmax); vx += sx * k2; vy += sy * k2
    s = math.hypot(vx, vy)
    if s > vmax: vx, vy = vx * vmax / s, vy * vmax / s
    m = Twist(); c, sn = math.cos(th), math.sin(th); m.linear.x, m.linear.y = c * vx + sn * vy, -sn * vx + c * vy; pub.publish(m)
for _ in range(8): pub.publish(Twist()); rclpy.spin_once(n, timeout_sec=0.03)
print(f"{why}: robot at ({pose[0][0]:+.2f}, {pose[0][1]:+.2f}) after {time.time() - t0:.1f} s; closest approach: " + ", ".join(f"{k} {v:.2f} m" for k, v in sorted(closest.items())))
