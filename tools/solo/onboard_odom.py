#!/usr/bin/env python3
"""On-board odometry for the solo robot: runs ON THE ROBOT, reads the wheels and the gyro locally at full rate, keeps a
running pose (x, y, theta) in an "odom" frame that is zero where the program started, and publishes only that pose
across the network. The laptop (solo_node, recover_source:=robot) anchors it to vision. solo_node's recovery is off by
default (recover:=false) and has been checked offline on recorded data only, not yet on the robot.

    # on the robot, with a copy of this file in $HOME/solo_odom:
    docker run -d --name solo_odom --restart unless-stopped --net=host --ipc=host -e ROS_DOMAIN_ID=77 \
      -v $HOME/solo_odom:/odo pequi_ssl_el-robot \
      bash -c "source /opt/ros/iron/setup.bash && exec python3 -u /odo/onboard_odom.py"

In:  /wheels/state  sensor_msgs/JointState  position = integrated wheel angles (rad), velocity (rad/s); best effort
     /imu           sensor_msgs/Imu         angular_velocity.z, raw (the offset is not removed by the driver)
Out: /solo/odom         geometry_msgs/Pose2D  x, y (m), theta (rad, wrapped), odom frame, 30 Hz, best effort;
                                              not published while /wheels/state is older than WHEEL_STALE
     /solo/odom_status  std_msgs/String       one line, 1 Hz: gyro state (ok / stale / none), offset, rates, still

Position: body displacement = M @ (wheel position change) per /wheels/state sample, rotated by the mid-step heading.
Heading: the gyro (rate less the offset) integrated between wheel samples; when /imu is older than GYRO_STALE s, the
wheels' own heading (third row of M) instead, and the status line says "gyro stale". The gyro offset is the mean raw
rate over the time the wheels have been still (every wheel under STILL_SPEED rad/s) for STILL_TIME s or more, as the
robot's velocity controller does (robot_pid.py), but kept up to date at every still stretch, not only once.
While still the heading is held (the offset is being measured, the robot is not turning).
No ROS time is used: all timing is this computer's monotonic clock.

The ROS-free class OnboardOdom can be run offline over recordings of /wheels/state and /imu."""
import math
import time

# Wheels -> body motion: body (dx, dy, dth) = M @ (dpos0, dpos1, dpos2), wheel positions in rad, dx forward, dy left (m),
# dth (rad). Fitted by least squares against vision (body velocity vs wheel rates) on recorded field runs of the
# three-wheel robot on carpet (same as M_FIELD in solo_strategy/odometry.py).
# The nominal three-wheel kinematics (-60 / 60 / 180 deg, 81.4 mm, r 24.5 mm) give the same shape up to a sign on x and
# 10-20 % more travel than the fit (roller slip). The fit holds for the motor calibration the runs were recorded with: re-fit after
# recalibrating a motor.
M = ((-0.01293, 0.01128, -0.00072),
     (0.00802, 0.00801, -0.01342),
     (0.08944, 0.08871, 0.08843))

STILL_SPEED = 0.2      # rad/s: every wheel under this counts as still (encoder noise when standing: about 0.05)
STILL_TIME = 1.0       # s still before the gyro offset is measured
OFFSET_TAU = 2.0       # s: time constant of the running offset mean while still
OFFSET_MAX = 0.15      # rad/s: a raw rate beyond this while "still" is not offset (robot lifted and turned): ignored
GYRO_STALE = 0.2       # s without /imu: heading from the wheels
WHEEL_STALE = 0.25     # s without /wheels/state: /solo/odom is not published (the laptop then sees it stale and stands still)
WHEEL_JUMP = 80.0      # rad/s: a position change beyond this is the wheel driver restarting, not motion
PUB_HZ = 30.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class OnboardOdom:
    def __init__(self, m=M):
        self.m = [list(r) for r in m]
        self.x = self.y = self.th = 0.0
        self.pos, self.t_w, self.t_wheel_last = None, None, None
        self.g_t, self.g_acc, self.g_acc_w = None, 0.0, 0.0   # last gyro time, gyro angle (offset removed), its value at the last wheel sample
        self.offset, self.offset_n, self.offset_ok = 0.0, 0, False
        self.still_since = None
        self.n_w = self.n_g = self.resets = 0
        self.wheel_heading_steps = 0

    def still(self, t):
        return self.still_since is not None and t - self.still_since >= STILL_TIME

    def gyro_fresh(self, t):
        return self.g_t is not None and t - self.g_t < GYRO_STALE

    def gyro(self, t, gz):
        """one /imu sample: t local monotonic time (s), gz raw z rate (rad/s)"""
        if not math.isfinite(gz):
            return
        self.n_g += 1
        if self.still(t) and abs(gz) < OFFSET_MAX:
            if self.offset_n == 0:
                self.offset = gz
            else:
                a = min(1.0, max(t - self.g_t, 0.0) / OFFSET_TAU) if self.g_t is not None else 0.0
                self.offset += a * (gz - self.offset)
            self.offset_n += 1
            self.offset_ok = self.offset_ok or self.offset_n >= 100
        if self.g_t is not None and 0.0 < t - self.g_t < 0.1 and not self.still(t):
            self.g_acc += (gz - self.offset) * (t - self.g_t)
        self.g_t = t

    def wheels(self, t, pos, vel=None):
        """one /wheels/state sample: positions (rad), velocities (rad/s, for the stillness test)"""
        if len(pos) != len(self.m[0]) or not all(math.isfinite(p) for p in pos):
            return
        self.n_w += 1
        if vel is not None and len(vel) == len(pos) and all(math.isfinite(v) for v in vel):
            if max(abs(v) for v in vel) < STILL_SPEED:
                if self.still_since is None:
                    self.still_since = t
            else:
                self.still_since = None
        if self.pos is None:
            self.pos, self.t_w, self.g_acc_w = list(pos), t, self.g_acc
            return
        d = [p - q for p, q in zip(pos, self.pos)]
        dt = max(t - self.t_w, 1e-3)
        self.pos, self.t_w = list(pos), t
        g = self.g_acc
        dg, self.g_acc_w = g - self.g_acc_w, g
        if max(abs(v) for v in d) > WHEEL_JUMP * dt + 1.0:     # driver restarted (positions back to 0): no motion
            self.resets += 1
            return
        bx = sum(a * b for a, b in zip(self.m[0], d))
        by = sum(a * b for a, b in zip(self.m[1], d))
        if self.gyro_fresh(t):
            dth = dg
        else:
            dth = sum(a * b for a, b in zip(self.m[2], d))
            self.wheel_heading_steps += 1
        mid = self.th + 0.5 * dth
        c, s = math.cos(mid), math.sin(mid)
        self.x += c * bx - s * by
        self.y += s * bx + c * by
        self.th = wrap(self.th + dth)

    def pose(self):
        return self.x, self.y, self.th

    def gyro_state(self, t):
        if self.g_t is None:
            return 'none'
        return 'ok' if self.gyro_fresh(t) else 'stale'


def main():
    import rclpy
    from geometry_msgs.msg import Pose2D
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
    from sensor_msgs.msg import Imu, JointState
    from std_msgs.msg import String

    be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                    durability=QoSDurabilityPolicy.VOLATILE)

    class OdomNode(Node):
        def __init__(self):
            super().__init__('solo_onboard_odom')
            self.o = OnboardOdom()
            self.t0 = time.monotonic()
            self.pub = self.create_publisher(Pose2D, '/solo/odom', be)
            self.pub_s = self.create_publisher(String, '/solo/odom_status', be)
            self.create_subscription(JointState, '/wheels/state',
                                     lambda m: self.o.wheels(time.monotonic(), list(m.position), list(m.velocity)), be)
            self.create_subscription(Imu, '/imu', lambda m: self.o.gyro(time.monotonic(), m.angular_velocity.z), be)
            self.create_timer(1.0 / PUB_HZ, self.publish)
            self.create_timer(1.0, self.status)
            self.n_pub, self.last = 0, (0, 0, 0, time.monotonic())
            self.get_logger().info(f'onboard odometry up: /wheels/state + /imu -> /solo/odom at {PUB_HZ:g} Hz')

        def publish(self):
            if self.o.t_w is None or time.monotonic() - self.o.t_w > WHEEL_STALE:
                return                                  # no fresh wheel sample: say nothing rather than an old pose
            x, y, th = self.o.pose()
            self.pub.publish(Pose2D(x=float(x), y=float(y), theta=float(th)))
            self.n_pub += 1

        def status(self):
            t = time.monotonic()
            nw, ng, npub, t1 = self.last
            dt = max(t - t1, 1e-3)
            o = self.o
            wa = 'none' if o.t_w is None else f'{t - o.t_w:.2f}s'
            line = (f'gyro {o.gyro_state(t)} offset {o.offset:+.4f} rad/s ({"measured" if o.offset_ok else "not yet measured"}) '
                    f'heading {"gyro" if o.gyro_fresh(t) else "WHEELS"} still {"yes" if o.still(t) else "no"} '
                    f'rates wheels {(o.n_w - nw) / dt:.0f} Hz imu {(o.n_g - ng) / dt:.0f} Hz pub {(self.n_pub - npub) / dt:.0f} Hz '
                    f'wheel age {wa} resets {o.resets} pose {o.x:+.3f} {o.y:+.3f} {o.th:+.3f} up {t - self.t0:.0f}s')
            self.last = (o.n_w, o.n_g, self.n_pub, t)
            self.pub_s.publish(String(data=line))
            if o.gyro_state(t) != 'ok' or (o.t_w is None or t - o.t_w > 0.3):
                self.get_logger().warning(line)

    rclpy.init()
    n = OdomNode()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, Exception) as e:     # docker stop (SIGTERM) ends the spin
        if not isinstance(e, KeyboardInterrupt) and type(e).__name__ != 'ExternalShutdownException':
            raise
    finally:
        try:
            n.destroy_node()
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
