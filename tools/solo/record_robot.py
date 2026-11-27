"""Record the robot's own topics (commands, wheel telemetry, IMU) as JSON lines, gzip.
    source tools/solo/env.sh; python3 tools/solo/record_robot.py [--out DIR] [--robot | --laptop] [--tag NAME]
--out defaults to $SOLO_DATA/raw (SOLO_DATA default: log/solo in this repository; created if missing).
--robot: run ON the robot (nothing crosses the Wi-Fi): its topics without /solo/debug, flushed every 2 s (power cuts).
--laptop: only what the laptop publishes itself (/cmd_vel, /solo/debug, and the two-robot node's copies of its commands,
/duo/robot<id>/cmd_vel), the companion of a recorder on the robot.
--tag: added to the file name (robot_<date>_<time>_<tag>.jsonl.gz), e.g. the robot whose domain is recorded: two
recorders started in the same second then write two files. Two robots: one recorder per robot's ROS domain
(SOLO_DOMAIN=<n> source tools/solo/env.sh first), each with its own tag.
Lines: {"t": unix, "topic": ..., ...fields}. wheels/state: velocity rad/s, effort = Iq A, position rad.
imu: all three accelerations and rates (az shows the robot being lifted or set down, ax/ay a tilt while it is carried)."""
import gzip, json, os, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as BE
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray, String

out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(os.environ.get("SOLO_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "log", "solo")), "raw")
os.makedirs(out, exist_ok=True)
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
if tag and not tag.replace("_", "").replace("-", "").isalnum(): sys.exit(f"--tag: letters, digits, - and _ only, not {tag!r}")
path = f"{out}/robot_{time.strftime('%Y%m%d_%H%M%S')}{'_' + tag if tag else ''}.jsonl.gz"
f = gzip.open(path, "wt", compresslevel=5)
print("recording to", path, flush=True)
ON_ROBOT, ON_LAPTOP = '--robot' in sys.argv, '--laptop' in sys.argv; FLUSH = 2.0 if ON_ROBOT else 60.0
rclpy.init(); n = Node('record_robot' + ('_onboard' if ON_ROBOT else '')); cnt = {}
def w(topic, **kw):
    f.write(json.dumps(dict(t=round(time.time(), 4), topic=topic, **kw)) + "\n"); cnt[topic] = cnt.get(topic, 0) + 1
tw = lambda m: dict(vx=round(m.linear.x, 4), vy=round(m.linear.y, 4), w=round(m.angular.z, 4))
r4 = lambda a: [None if x != x else round(float(x), 4) for x in a]
n.create_subscription(Twist, '/cmd_vel', lambda m: w('cmd_vel', **tw(m)), BE)
if not ON_LAPTOP:
  n.create_subscription(Twist, '/cmd_vel_corrected', lambda m: w('cmd_vel_corrected', **tw(m)), BE)
  n.create_subscription(Float32MultiArray, '/wheels/cmd', lambda m: w('wheels_cmd', v=r4(m.data)), BE)
  n.create_subscription(JointState, '/wheels/state', lambda m: w('wheels_state', vel=r4(m.velocity), iq=r4(m.effort), pos=r4(m.position)), BE)
  n.create_subscription(Imu, '/imu', lambda m: w('imu', gz=round(m.angular_velocity.z, 4), ax=round(m.linear_acceleration.x, 3), ay=round(m.linear_acceleration.y, 3), az=round(m.linear_acceleration.z, 3), gx=round(m.angular_velocity.x, 4), gy=round(m.angular_velocity.y, 4)), BE)
if not ON_ROBOT:
  n.create_subscription(String, '/solo/debug', lambda m: w('solo_debug', d=json.loads(m.data)), BE)
  for i in range(16):                       # duo_node's copy of each robot's command (vision id i), on the laptop's domain
    n.create_subscription(Twist, f'/duo/robot{i}/cmd_vel', lambda m, i=i: w(f'duo_cmd_vel{i}', **tw(m)), BE)
t = tp = time.time()
try:
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.5)
        if time.time() - t > FLUSH:
            f.flush(); t = time.time()
        if time.time() - tp > 60:
            print(time.strftime('%H:%M:%S'), cnt, flush=True); tp = time.time()
finally:
    f.close()
