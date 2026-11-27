"""Offline stand-in for the camera and the robot, to dry-run characterize.py / fspin_runs.sh with no robot.
Publishes /vision/poses/<team>/robot0 and /vision/poses/ball at 60 Hz, executes /cmd_vel with a crude plant:
  packet delay NET s, the body command held over the ground from its arrival (the drive node's behaviour), field
  velocity following it with a first-order lag TAU s (acceleration capped at ACC m/s2), camera pose VIS s old.
With this plant the straightest strategy latency is VIS + NET (the node's own aim-ahead is taken as exact).
Run ONLY on an isolated domain:  ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=231 python3 fake_field.py blue
"""
import math, os, sys, collections
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Twist

if os.environ.get('ROS_LOCALHOST_ONLY') != '1': sys.exit('refusing: set ROS_LOCALHOST_ONLY=1 (offline dry run only)')
team = sys.argv[1] if len(sys.argv) > 1 else 'blue'
NET, TAU, ACC, VIS = (float(os.environ.get(k, d)) for k, d in (('NET', '0.02'), ('TAU', '0.06'), ('ACC', '2.5'), ('VIS', '0.035')))
X0 = [float(v) for v in os.environ.get('START', '-1.3,0.75,0').split(',')]
BALL = [float(v) for v in os.environ.get('BALL', '1.8,-1.2').split(',')]

rclpy.init(); n = Node('fake_field')
pr = n.create_publisher(Pose2D, f'/vision/poses/{team}/robot0', 20)
pb = n.create_publisher(Pose2D, '/vision/poses/ball', 20)
st = dict(x=X0[0], y=X0[1], th=X0[2], vx=0.0, vy=0.0, w=0.0, t=0.0)
inbox = collections.deque(); hist = collections.deque()
held = [None, 0.0, 0.0]     # (field vx, vy) fixed at arrival, w ; None = zero
last_rx = [-1.0]
def on_cmd(m):
    inbox.append((st['t'] + NET, m.linear.x, m.linear.y, m.angular.z))
n.create_subscription(Twist, '/cmd_vel', on_cmd, qos_profile_sensor_data)
DT = 0.005; k_pub = [0]
def step():
    t = st['t'] = st['t'] + DT
    while inbox and inbox[0][0] <= t:
        _, bx, by, w = inbox.popleft(); c, s = math.cos(st['th']), math.sin(st['th'])
        held[:] = [(c * bx - s * by, s * bx + c * by), w, 0]; last_rx[0] = t
    if held[0] is None or t - last_rx[0] > 0.3: fx, fy, fw = 0.0, 0.0, 0.0    # node timeout 0.3 s
    else: (fx, fy), fw = held[0], held[1]
    a = min(1.0, DT / TAU)
    dvx, dvy = a * (fx - st['vx']), a * (fy - st['vy']); dv = math.hypot(dvx, dvy); cap = ACC * DT
    if dv > cap: dvx, dvy = dvx * cap / dv, dvy * cap / dv
    st['vx'] += dvx; st['vy'] += dvy; st['w'] += min(1.0, DT / 0.04) * (fw - st['w'])
    st['x'] += st['vx'] * DT; st['y'] += st['vy'] * DT; st['th'] += st['w'] * DT
    hist.append((t, st['x'], st['y'], st['th']))
    while hist and hist[0][0] < t - VIS - 0.05: hist.popleft()
    k_pub[0] += 1
    if k_pub[0] % 3 == 0:                      # ~67 Hz camera
        old = next((h for h in hist if h[0] >= t - VIS), hist[-1])
        m = Pose2D(); m.x, m.y = old[1], old[2]; m.theta = (old[3] + math.pi) % (2 * math.pi) - math.pi; pr.publish(m)
        b = Pose2D(); b.x, b.y = BALL; pb.publish(b)
n.create_timer(DT, step)
rclpy.spin(n)
