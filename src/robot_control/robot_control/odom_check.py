"""Measure what the odometry node is doing, with or without ground truth.

    ros2 run robot_control odom_check --ros-args -p seconds:=30.0

Records imu, wheels/state, odom and odom/gyro_bias for `seconds`, then
reports. Tests you can do on the real robot (no simulator, no vision):

  still     robot untouched for 60 s: speed ~0, heading drift per minute
            (fused vs raw gyro), and measured sensor noise to use as
            sigma_gyro / sigma_wheel
  spin      mark the robot's heading, turn it N full turns (teleop or by
            hand), put it back on the mark: total rotation should read
            N x 360 deg
  straight  drive (or push) along a tape measure: the fused distance
            should match the tape

With fake_robot running (odom/ground_truth present) it also reports errors
against the truth.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray, Float64

from .geometry import (DEFAULT_ROBOT_RADIUS, DEFAULT_WHEEL_ANGLES_DEG,
                       DEFAULT_WHEEL_RADIUS, wheel_matrix)


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def odom_row(m):
    p, tw = m.pose.pose, m.twist.twist
    return (stamp_s(m.header), p.position.x, p.position.y, yaw(p.orientation),
            tw.linear.x, tw.linear.y, tw.angular.z)


def rms(a):
    a = np.asarray(a)
    return float(np.sqrt(np.mean(np.square(a)))) if a.size else float('nan')


class OdomCheck(Node):
    def __init__(self):
        super().__init__('odom_check')
        self.duration = float(self.declare_parameter('seconds', 30.0).value)
        self.sigma_wheel = float(self.declare_parameter('sigma_wheel', 0.3).value)
        self.J = np.array(wheel_matrix(DEFAULT_WHEEL_ANGLES_DEG, DEFAULT_ROBOT_RADIUS,
                                       DEFAULT_WHEEL_RADIUS))
        self.imu, self.wheels, self.odom, self.truth = [], [], [], []
        self.bias, self.true_bias, self.residuals = [], [], []
        self.create_subscription(
            Imu, 'imu', lambda m: self.imu.append((stamp_s(m.header), m.angular_velocity.z)), 100)
        self.create_subscription(
            JointState, 'wheels/state',
            lambda m: self.wheels.append((stamp_s(m.header), list(m.velocity))), 20)
        self.create_subscription(Odometry, 'odom', lambda m: self.odom.append(odom_row(m)), 50)
        self.create_subscription(
            Odometry, 'odom/ground_truth', lambda m: self.truth.append(odom_row(m)), 50)
        self.create_subscription(Float64, 'odom/gyro_bias', lambda m: self.bias.append(m.data), 50)
        self.create_subscription(
            Float64, 'odom/ground_truth/gyro_bias', lambda m: self.true_bias.append(m.data), 50)
        self.create_subscription(
            Float32MultiArray, 'odom/wheel_residuals',
            lambda m: self.residuals.append(list(m.data)), 20)
        self.start = None
        self.created = self.get_clock().now()
        self.warned = False
        self.done = False
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'recording {self.duration:.0f} s once data arrives ...')

    def tick(self):
        now = self.get_clock().now()
        if self.start is None:
            if self.odom:
                self.start = now
            elif not self.warned and (now - self.created).nanoseconds > 5e9:
                self.warned = True
                self.get_logger().warning('no odom yet: is the odometry node running '
                                          'and receiving imu + wheels/state?')
            return
        if (now - self.start).nanoseconds * 1e-9 >= self.duration:
            self.report()
            self.done = True

    def report(self):
        odom = np.array(self.odom)
        span = odom[-1, 0] - odom[0, 0]
        t0, t1 = odom[0, 0], odom[-1, 0]
        imu = np.array([r for r in self.imu if t0 <= r[0] <= t1])
        wheels = [r for r in self.wheels if t0 <= r[0] <= t1]
        print(f'\n=== odom_check ({span:.1f} s) ===')
        print(f'rates: imu {len(imu) / span:5.1f} Hz   wheels/state {len(wheels) / span:5.1f} Hz'
              f'   odom {len(odom) / span:5.1f} Hz')

        fused_rot = np.degrees(np.sum(np.diff(np.unwrap(odom[:, 3]))))
        raw_rot = np.degrees(np.sum(imu[1:, 1] * np.diff(imu[:, 0]))) if len(imu) > 1 else math.nan
        J_pinv = np.linalg.pinv(self.J)
        wheel_rot, wheel_vel = 0.0, []
        for (ta, za), (tb, _zb) in zip(wheels, wheels[1:]):
            if np.all(np.isfinite(za)):
                v = J_pinv @ np.asarray(za)
                wheel_rot += v[2] * (tb - ta)
                wheel_vel.append((ta, *v))
        path = float(np.sum(np.hypot(np.diff(odom[:, 1]), np.diff(odom[:, 2]))))
        disp = math.hypot(odom[-1, 1] - odom[0, 1], odom[-1, 2] - odom[0, 2])
        print('total rotation:')
        print(f'  fused                  {fused_rot:9.1f} deg')
        print(f'  raw gyro (no bias fix) {raw_rot:9.1f} deg')
        print(f'  wheels only            {math.degrees(wheel_rot):9.1f} deg')
        print(f'distance: path {path:.3f} m, start->end {disp:.3f} m (fused)')
        if self.bias:
            print(f'gyro bias estimate: {self.bias[-1]:+.4f} rad/s')
        if self.residuals:
            res = np.array(self.residuals, dtype=float)
            slips = np.sum(np.abs(res) > 3 * self.sigma_wheel, axis=0)
            print('wheel residual RMS (rad/s): '
                  + ' '.join(f'{rms(res[np.isfinite(res[:, i]), i]):.2f}' for i in range(res.shape[1]))
                  + f'   readings > 3 sigma per wheel: {slips.tolist()}')

        if self.truth:
            self.report_truth(odom, np.array(wheel_vel))
        else:
            self.report_real(odom, imu, span, fused_rot, raw_rot)

    def report_real(self, odom, imu, span, fused_rot, raw_rot):
        speed = np.hypot(odom[:, 4], odom[:, 5])
        if np.mean(speed) < 0.02 and abs(fused_rot) < 5.0:
            print('\nrobot was still:')
            print(f'  fused speed mean {np.mean(speed):.4f} m/s, |omega| mean '
                  f'{np.mean(np.abs(odom[:, 6])):.4f} rad/s')
            print(f'  heading drift: fused {fused_rot * 60 / span:+.2f} deg/min, '
                  f'raw gyro {raw_rot * 60 / span:+.2f} deg/min')
            if len(imu) > 10:
                print(f'  -> measured sigma_gyro:={np.std(imu[:, 1]):.4f}', end='')
            speeds = np.array([z for _t, z in self.wheels], dtype=float)
            if speeds.size:
                print(f'  sigma_wheel:={np.nanstd(speeds):.3f}', end='')
            print()
        else:
            print('\nrobot moved: compare total rotation with the turns you made '
                  '(N x 360 deg), and distance with your tape measure')

    def report_truth(self, odom, wheel_vel):
        truth = np.array(self.truth)

        def truth_at(t):
            return np.array([np.interp(t, truth[:, 0], truth[:, i]) for i in (4, 5, 6)])

        fused_err = [np.hypot(*(r[4:6] - truth_at(r[0])[:2])) for r in odom]
        wheel_err = [np.hypot(*(np.array(r[1:3]) - truth_at(r[0])[:2])) for r in wheel_vel]
        omega_err = [r[6] - truth_at(r[0])[2] for r in odom]
        th_true = np.unwrap(truth[:, 3])
        true_rot = np.degrees(np.interp(odom[-1, 0], truth[:, 0], th_true)
                              - np.interp(odom[0, 0], truth[:, 0], th_true))
        fused_rot = np.degrees(np.sum(np.diff(np.unwrap(odom[:, 3]))))
        print('\nground truth (fake_robot):')
        print(f'  true rotation {true_rot:.1f} deg -> fused heading error {fused_rot - true_rot:+.2f} deg')
        print(f'  linear velocity RMS  wheels only {rms(wheel_err):.3f} m/s   fused {rms(fused_err):.3f} m/s')
        print(f'  omega RMS            fused {rms(omega_err):.3f} rad/s')
        if self.true_bias and self.bias:
            print(f'  gyro bias            true {self.true_bias[-1]:.4f}   '
                  f'estimated {self.bias[-1]:.4f} rad/s')
        ok = rms(fused_err) <= rms(wheel_err) and abs(fused_rot - true_rot) < 5.0
        print('RESULT:', 'PASS' if ok else 'FAIL')


def main(args=None):
    rclpy.init(args=args)
    node = OdomCheck()
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
