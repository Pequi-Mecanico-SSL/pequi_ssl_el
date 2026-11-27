"""Open-loop nudge of the real robot: publishes a body-frame velocity on /cmd_vel for a few seconds, then zeros, and
prints the wheel speeds the robot reported. The robot MOVES.
    source tools/solo/env.sh; python3 tools/solo/nudge.py vx vy w seconds"""
import rclpy, time, sys
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
# usage: nudge.py vx vy w seconds   -> publishes on /cmd_vel at 20 Hz, then zeros; prints wheel speeds
vx, vy, w, T = (float(a) for a in sys.argv[1:5])
rclpy.init(); n = Node('nudge'); pub = n.create_publisher(Twist, '/cmd_vel', qos_profile_sensor_data); ws = []
n.create_subscription(JointState, '/wheels/state', lambda m: ws.append((time.time(), list(m.velocity), list(m.effort))), qos_profile_sensor_data)
t0 = time.time()
while time.time() - t0 < 0.6: rclpy.spin_once(n, timeout_sec=0.02)      # discovery
t0 = time.time(); peak = [0, 0, 0]; k0 = len(ws)
while time.time() - t0 < T:
    m = Twist(); m.linear.x, m.linear.y, m.angular.z = vx, vy, w; pub.publish(m)
    rclpy.spin_once(n, timeout_sec=0.05)
mid = [v for t, v, e in ws[k0:] if t - t0 > 0.5 * T]
for _ in range(10): pub.publish(Twist()); rclpy.spin_once(n, timeout_sec=0.05)
print(f"cmd vx {vx} vy {vy} w {w} for {T} s: {len(ws) - k0} wheel-state msgs; mean wheel speed (rad/s) in the 2nd half: "
      + (str([round(sum(c) / len(c), 2) for c in zip(*mid)]) if mid else "none received"))
