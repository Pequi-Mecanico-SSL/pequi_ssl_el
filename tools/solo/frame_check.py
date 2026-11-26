"""Checks the vision frame against the robot's own frame: drives forward, then turns, and compares with what vision saw.
    source tools/solo/env.sh; python3 tools/solo/frame_check.py <blue|yellow>
The vision connector must be running. The robot moves about 0.3 m forward and turns about 90 degrees: clear the space."""
import math, sys, time
import os
import rclpy
IDX = os.environ.get('SOLO_ROBOT_INDEX', '0')     # our robot's index among its colour in the vision connector
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Twist
team = sys.argv[1]
rclpy.init(); n = Node('frame_check'); pub = n.create_publisher(Twist, '/cmd_vel', qos_profile_sensor_data); pose = [None, 0.0]
n.create_subscription(Pose2D, f'/vision/poses/{team}/robot{IDX}', lambda m: pose.__setitem__(0, (m.x, m.y, m.theta)) or pose.__setitem__(1, time.time()), 10)
def spin(T, vx=0.0, vy=0.0, w=0.0, send=False):
    t0 = time.time()
    while time.time() - t0 < T:
        if send:
            m = Twist(); m.linear.x, m.linear.y, m.angular.z = vx, vy, w; pub.publish(m)
        rclpy.spin_once(n, timeout_sec=0.04)
spin(1.5)
if pose[0] is None or time.time() - pose[1] > 0.5:
    sys.exit(f"no pose on /vision/poses/{team}/robot{IDX}: is the connector running, is the robot in view?")
p0 = pose[0]; print(f"start pose x {p0[0]:+.2f} y {p0[1]:+.2f} theta {math.degrees(p0[2]):+.0f} deg")
spin(1.2, vx=0.25, send=True); spin(0.8, send=True); p1 = pose[0]
dx, dy = p1[0] - p0[0], p1[1] - p0[1]; d = math.hypot(dx, dy)
err = math.degrees((math.atan2(dy, dx) - p0[2] + math.pi) % (2 * math.pi) - math.pi) if d > 0.03 else float('nan')
print(f"forward 0.25 m/s x 1.2 s: moved {d:.2f} m, direction {err:+.0f} deg from the heading vision reports (0 = consistent, 180 = heading reversed)")
spin(1.0, w=1.5, send=True); spin(0.8, send=True); p2 = pose[0]
dth = math.degrees((p2[2] - p1[2] + math.pi) % (2 * math.pi) - math.pi)
print(f"turn +1.5 rad/s x 1 s (counter-clockwise for the robot): vision theta changed {dth:+.0f} deg (positive = consistent; negative = the connector's mirror setting is wrong, flip it)")
ok = d > 0.1 and abs(err) < 35 and dth > 20
print("RESULT:", "frames consistent" if ok else "NOT consistent: see the two lines above")
