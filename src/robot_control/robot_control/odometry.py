"""Gyro + wheel odometry (OdomEKF) as a ROS node on the robot.

Topics:
  imu                   (Imu, in)               gyro z, 200 Hz
  wheels/state          (JointState, in)        measured wheel speeds, rad/s
  odom                  (Odometry, out)         pose in the odometry frame (starts
                                                at 0) + body velocity + covariances
  odom/gyro_bias        (Float64, out)          current gyro bias estimate, rad/s
  odom/wheel_residuals  (Float32MultiArray, out) measured - predicted wheel speed;
                                                a slipping wheel stands out here

velocity_controller uses odom (heading + omega) instead of its own gyro
integration when started with use_odom:=true.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray, Float64

from .geometry import (DEFAULT_ROBOT_RADIUS, DEFAULT_WHEEL_ANGLES_DEG,
                       DEFAULT_WHEEL_RADIUS, wheel_matrix)
from .odom_ekf import OdomEKF

EKF_PARAMS = ('sigma_gyro', 'sigma_wheel', 'accel_noise', 'ang_accel_noise',
              'bias_noise', 'initial_bias_sigma', 'max_wheel_rejects')


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry')
        self.declare_parameter('wheel_angles_deg', DEFAULT_WHEEL_ANGLES_DEG)
        self.declare_parameter('robot_radius', DEFAULT_ROBOT_RADIUS)
        self.declare_parameter('wheel_radius', DEFAULT_WHEEL_RADIUS)
        self.declare_parameter('publish_rate_hz', 100.0)
        self.declare_parameter('stats_period_s', 5.0)
        defaults = OdomEKF([[0.0, 0.0, 0.0]])
        self.declare_parameter('sigma_gyro', math.sqrt(defaults.r_gyro))
        self.declare_parameter('sigma_wheel', math.sqrt(defaults.r_wheel))
        self.declare_parameter('accel_noise', math.sqrt(defaults.accel_q))
        self.declare_parameter('ang_accel_noise', math.sqrt(defaults.ang_accel_q))
        self.declare_parameter('bias_noise', math.sqrt(defaults.bias_q))
        self.declare_parameter('initial_bias_sigma', math.sqrt(defaults.P[6, 6]))
        self.declare_parameter('max_wheel_rejects', defaults.max_wheel_rejects)

        p = lambda n: self.get_parameter(n).value
        rows = wheel_matrix(p('wheel_angles_deg'), float(p('robot_radius')),
                            float(p('wheel_radius')))
        kwargs = {n: p(n) for n in EKF_PARAMS}
        self.ekf = OdomEKF(rows, **kwargs)
        self.n_wheels = len(rows)

        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.bias_pub = self.create_publisher(Float64, 'odom/gyro_bias', 10)
        self.resid_pub = self.create_publisher(Float32MultiArray, 'odom/wheel_residuals', 10)
        self.create_subscription(Imu, 'imu', self.imu_callback, 50)
        self.create_subscription(JointState, 'wheels/state', self.wheels_callback, 10)
        self.create_timer(1.0 / float(p('publish_rate_hz')), self.publish)
        self.create_timer(float(p('stats_period_s')), self.log_stats)
        self.last_slips = [0] * self.n_wheels
        self.get_logger().info(f'Odometry up: {kwargs}')

    def stamp_s(self, header):
        t = Time.from_msg(header.stamp)
        if t.nanoseconds == 0:
            t = self.get_clock().now()
        return t.nanoseconds * 1e-9

    def imu_callback(self, msg):
        self.ekf.update_gyro(self.stamp_s(msg.header), msg.angular_velocity.z)

    def wheels_callback(self, msg):
        if len(msg.velocity) != self.n_wheels:
            self.get_logger().warning(
                f'wheels/state has {len(msg.velocity)} velocities, expected {self.n_wheels}',
                throttle_duration_sec=5.0)
            return
        self.ekf.update_wheels(self.stamp_s(msg.header), list(msg.velocity))
        resid = Float32MultiArray()
        resid.data = [float(r) for r in self.ekf.last_residuals]
        self.resid_pub.publish(resid)

    def publish(self):
        if self.ekf.t is None:
            return
        now = self.get_clock().now()
        x, P = self.ekf.predict(now.nanoseconds * 1e-9)
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = float(x[0])
        odom.pose.pose.position.y = float(x[1])
        odom.pose.pose.orientation.z = math.sin(x[2] / 2.0)
        odom.pose.pose.orientation.w = math.cos(x[2] / 2.0)
        odom.twist.twist.linear.x = float(x[3])
        odom.twist.twist.linear.y = float(x[4])
        odom.twist.twist.angular.z = float(x[5])
        # 6x6 row-major over (x, y, z, roll, pitch, yaw); fill the planar part
        pose_cov = [0.0] * 36
        twist_cov = [0.0] * 36
        planar = (0, 1, 5)
        for a in range(3):
            for b in range(3):
                pose_cov[planar[a] * 6 + planar[b]] = float(P[a, b])
                twist_cov[planar[a] * 6 + planar[b]] = float(P[3 + a, 3 + b])
        odom.pose.covariance = pose_cov
        odom.twist.covariance = twist_cov
        self.odom_pub.publish(odom)
        self.bias_pub.publish(Float64(data=float(x[6])))

    def log_stats(self):
        e = self.ekf
        if e.t is None:
            self.get_logger().warning('no imu / wheels/state data yet',
                                      throttle_duration_sec=30.0)
            return
        slips = [int(s - l) for s, l in zip(e.slips, self.last_slips)]
        self.last_slips = list(e.slips)
        self.get_logger().info(
            f'bias={e.x[6]:+.4f} rad/s  gyro={e.n_gyro} wheel_updates={e.n_wheel} '
            f'wheel_rejected={e.n_wheel_rejected} gyro_jumps={e.n_gyro_jumps} '
            f'slips(last period)={slips}')


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
