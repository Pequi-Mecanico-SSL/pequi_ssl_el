"""Measure what the vision filter is doing for one robot.

    ros2 run vision_filter vision_check --ros-args -p robot:=blue/robot0 -p seconds:=10.0

Collects raw (/vision/poses/*) and filtered (/vision/filtered/odom/*) data for
`seconds`, then prints a report. Two modes, picked automatically:

  ground truth (fake_vision running): RMS error of raw vs filtered pose,
      and of filtered velocity vs a naive finite difference of raw poses.
      Errors are measured at the moment each message arrives, so vision
      latency counts against raw poses the same way it hurts the strategy.

  no ground truth (real field): put the robot down and do not touch it.
      Reports raw vs filtered jitter (standard deviation) and the filtered
      speed, which should be close to zero.
"""
import bisect
import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float('nan')


def median(values):
    return float(np.median(np.abs(values))) if len(values) else float('nan')


class VisionCheck(Node):

    def __init__(self):
        super().__init__('vision_check')
        self.robot = self.declare_parameter('robot', 'blue/robot0').value
        self.duration = float(self.declare_parameter('seconds', 10.0).value)

        self.raw = []       # (t_receive, x, y, theta)
        self.filtered = []  # (t_stamp, x, y, theta, vx, vy, omega)
        self.truth = []     # (t_stamp, x, y, theta, vx, vy, omega)

        self.create_subscription(
            Pose2D, f'/vision/poses/{self.robot}', self.raw_cb, 50)
        self.create_subscription(
            Odometry, f'/vision/filtered/odom/{self.robot}',
            lambda m: self.filtered.append(self.odom_row(m)), 50)
        self.create_subscription(
            Odometry, f'/vision/ground_truth/{self.robot}',
            lambda m: self.truth.append(self.odom_row(m)), 200)
        self.start = None
        self.created = self.now_s()
        self.warned = False
        self.done = False
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'collecting {self.duration:.0f} s for {self.robot} ...')

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def odom_row(m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p, tw = m.pose.pose, m.twist.twist
        return (t, p.position.x, p.position.y, yaw(p.orientation),
                tw.linear.x, tw.linear.y, tw.angular.z)

    def raw_cb(self, m):
        self.raw.append((self.now_s(), m.x, m.y, m.theta))

    def tick(self):
        if self.start is None:
            if self.raw or self.filtered:
                self.start = self.now_s()
            elif not self.warned and self.now_s() - self.created > 5.0:
                self.warned = True
                self.get_logger().warn(
                    f'no data on /vision/poses/{self.robot} or /vision/filtered/odom/'
                    f'{self.robot} yet: are vision_connector and vision_filter running '
                    'and is the robot visible?')
            return
        if self.now_s() - self.start >= self.duration:
            self.report()
            self.done = True

    def truth_at(self, t):
        times = [r[0] for r in self.truth]
        k = bisect.bisect_left(times, t)
        if k == 0 or k == len(times):
            return None
        a, b = np.array(self.truth[k - 1]), np.array(self.truth[k])
        f = (t - a[0]) / (b[0] - a[0])
        out = a + f * (b - a)
        out[3] = wrap(a[3] + f * wrap(b[3] - a[3]))
        return out

    def report(self):
        span = self.duration
        print(f'\n=== vision_check {self.robot} ({span:.0f} s) ===')
        print(f'raw messages:      {len(self.raw):5d}  ({len(self.raw) / span:5.1f} Hz)')
        print(f'filtered messages: {len(self.filtered):5d}  ({len(self.filtered) / span:5.1f} Hz)')
        if not self.raw or not self.filtered:
            print('missing data: is the connector (raw) and vision_filter (filtered) running, '
                  'and is the robot visible?')
            return
        if self.truth:
            self.report_truth()
        else:
            self.report_stationary()

    def report_truth(self):
        raw_pos, raw_th, filt_pos, filt_th, filt_vel, diff_vel = [], [], [], [], [], []
        for t, x, y, th in self.raw:
            g = self.truth_at(t)
            if g is not None:
                raw_pos.append(math.hypot(x - g[1], y - g[2]))
                raw_th.append(wrap(th - g[3]))
        for (t0, x0, y0, th0), (t1, x1, y1, th1) in zip(self.raw, self.raw[1:]):
            g = self.truth_at(t1)
            if g is None or t1 - t0 < 1e-3:
                continue
            # naive: difference consecutive raw poses, rotate into body frame
            vxf, vyf = (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)
            c, s = math.cos(th1), math.sin(th1)
            diff_vel.append(math.hypot(c * vxf + s * vyf - g[4], -s * vxf + c * vyf - g[5]))
        for t, x, y, th, vx, vy, w in self.filtered:
            g = self.truth_at(t)
            if g is not None:
                filt_pos.append(math.hypot(x - g[1], y - g[2]))
                filt_th.append(wrap(th - g[3]))
                filt_vel.append(math.hypot(vx - g[4], vy - g[5]))
        print('mode: ground truth (errors measured when each message arrives)')
        print('  (RMS includes ghost detections; median shows typical smoothness)')
        print('                        raw               filtered')
        print(f'  position  median {1000 * median(raw_pos):7.1f} mm      {1000 * median(filt_pos):7.1f} mm')
        print(f'            RMS    {1000 * rms(raw_pos):7.1f} mm      {1000 * rms(filt_pos):7.1f} mm')
        print(f'  heading   median {math.degrees(median(raw_th)):7.2f} deg     {math.degrees(median(filt_th)):7.2f} deg')
        print(f'  velocity  median {median(diff_vel):7.3f} m/s     {median(filt_vel):7.3f} m/s'
              '   (raw = differencing consecutive poses)')
        print(f'            RMS    {rms(diff_vel):7.3f} m/s     {rms(filt_vel):7.3f} m/s')
        ok = (rms(filt_pos) < rms(raw_pos) and median(filt_pos) < 1.5 * median(raw_pos)
              and rms(filt_vel) < rms(diff_vel))
        print('RESULT:', 'PASS - filtered beats raw' if ok else 'FAIL - filter is not helping')

    def report_stationary(self):
        raw = np.array(self.raw)[:, 1:]
        filt = np.array(self.filtered)
        spread = lambda a: (1000 * math.hypot(np.std(a[:, 0]), np.std(a[:, 1])),
                            math.degrees(np.std(np.unwrap(a[:, 2]))))
        raw_xy, raw_th = spread(raw)
        f_xy, f_th = spread(filt[:, 1:4])
        speed = np.hypot(filt[:, 4], filt[:, 5])
        print('mode: stationary robot (no ground truth) - robot must not move')
        print(f'  position jitter  raw {raw_xy:6.2f} mm   filtered {f_xy:6.2f} mm')
        print(f'  heading jitter   raw {raw_th:6.2f} deg  filtered {f_th:6.2f} deg')
        print(f'  filtered speed   mean {np.mean(speed):5.3f} m/s   max {np.max(speed):5.3f} m/s   '
              f'|omega| mean {np.mean(np.abs(filt[:, 6])):5.3f} rad/s')
        print(f'  -> use sigma_xy:={raw_xy / 1000 / math.sqrt(2):.4f} '
              f'sigma_theta:={math.radians(raw_th):.4f} for this camera')
        ok = f_xy < raw_xy and np.mean(speed) < 0.05
        print('RESULT:', 'PASS - smoother than raw, speed ~0' if ok
              else 'CHECK - robot moved, or noise settings need tuning')


def main(args=None):
    rclpy.init(args=args)
    node = VisionCheck()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
