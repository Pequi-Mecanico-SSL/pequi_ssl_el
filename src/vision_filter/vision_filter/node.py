"""Filtered robot poses and velocities from SSL-Vision (one EKF per robot).

Subscribes:
    /vision/stamped/{blue,yellow}/robot<i>   geometry_msgs/PoseStamped
                                             (ssl_vision_connector, capture-stamped)
    /<team>/robot<i>/cmd_vel                 geometry_msgs/Twist
                                             (our robots only, optional motion prior)

Publishes, at publish_rate_hz, each state predicted to the current time:
    /vision/filtered/{blue,yellow}/robot<i>       geometry_msgs/Pose2D
        drop-in replacement for /vision/poses/* (m, rad, field frame)
    /vision/filtered/odom/{blue,yellow}/robot<i>  nav_msgs/Odometry
        pose (field frame) + twist (body frame, REP-103) + covariances

A robot not seen for max_coast_s stops being published, so downstream
staleness checks (rl_strategy pose_timeout) behave as with raw vision.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from nav_msgs.msg import Odometry

from .ekf import RobotEKF

COLORS = ('blue', 'yellow')
EKF_PARAMS = ('sigma_xy', 'sigma_theta', 'accel_noise', 'ang_accel_noise',
              'cmd_tau', 'cmd_timeout', 'gate', 'max_rejects')


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class VisionFilterNode(Node):

    def __init__(self):
        super().__init__('vision_filter')
        defaults = RobotEKF()
        self.declare_parameters(
            namespace='',
            parameters=[
                ('blue_robot_count', 3),
                ('yellow_robot_count', 3),
                # robots whose cmd_vel is used as a motion prior; '' = none
                ('team', 'blue'),
                ('use_cmd_vel', True),
                ('publish_rate_hz', 100.0),
                ('max_coast_s', 0.25),
                ('reinit_after_s', 1.0),
                ('stats_period_s', 5.0),
                ('sigma_xy', math.sqrt(defaults.R[0, 0])),
                ('sigma_theta', math.sqrt(defaults.R[2, 2])),
                ('accel_noise', math.sqrt(defaults.accel_q)),
                ('ang_accel_noise', math.sqrt(defaults.ang_accel_q)),
                ('cmd_tau', defaults.cmd_tau),
                ('cmd_timeout', defaults.cmd_timeout),
                ('gate', defaults.gate),
                ('max_rejects', defaults.max_rejects),
            ]
        )
        gp = lambda n: self.get_parameter(n).value
        self.ekf_kwargs = {n: gp(n) for n in EKF_PARAMS}
        self.max_coast = float(gp('max_coast_s'))
        self.reinit_after = float(gp('reinit_after_s'))
        team = gp('team')
        counts = {'blue': gp('blue_robot_count'), 'yellow': gp('yellow_robot_count')}

        self.filters = {}
        self.last_seen = {}       # key -> Time of last measurement (capture time)
        self.latency_sum = {}     # key -> (sum of receive - capture, count)
        self.pose_pubs = {}
        self.odom_pubs = {}
        for color in COLORS:
            for i in range(counts[color]):
                key = (color, i)
                self.filters[key] = RobotEKF(**self.ekf_kwargs)
                self.latency_sum[key] = (0.0, 0)
                self.create_subscription(
                    PoseStamped, f'/vision/stamped/{color}/robot{i}',
                    lambda msg, k=key: self.vision_cb(msg, k), 20)
                self.pose_pubs[key] = self.create_publisher(
                    Pose2D, f'/vision/filtered/{color}/robot{i}', 10)
                self.odom_pubs[key] = self.create_publisher(
                    Odometry, f'/vision/filtered/odom/{color}/robot{i}', 10)
                if gp('use_cmd_vel') and color == team:
                    self.create_subscription(
                        Twist, f'/{color}/robot{i}/cmd_vel',
                        lambda msg, k=key: self.cmd_cb(msg, k), 10)

        self.create_timer(1.0 / float(gp('publish_rate_hz')), self.publish)
        self.create_timer(float(gp('stats_period_s')), self.log_stats)
        self.get_logger().info(
            f'vision filter up: robots={counts} cmd_vel prior='
            f'{team if gp("use_cmd_vel") else "off"} params={self.ekf_kwargs}')

    @staticmethod
    def seconds(t):
        return t.nanoseconds * 1e-9

    def vision_cb(self, msg, key):
        stamp = Time.from_msg(msg.header.stamp)
        now = self.get_clock().now()
        ekf = self.filters[key]
        last = self.last_seen.get(key)
        if last is not None and self.seconds(stamp - last) > self.reinit_after:
            # long absence: start over rather than trust a stale velocity
            self.filters[key] = ekf = RobotEKF(**self.ekf_kwargs)
        z = (msg.pose.position.x, msg.pose.position.y,
             yaw_from_quaternion(msg.pose.orientation))
        ekf.update(self.seconds(stamp), z)
        if last is None or stamp > last:
            self.last_seen[key] = stamp
        total, n = self.latency_sum[key]
        self.latency_sum[key] = (total + self.seconds(now - stamp), n + 1)

    def cmd_cb(self, msg, key):
        t = self.seconds(self.get_clock().now())
        self.filters[key].add_command(t, msg.linear.x, msg.linear.y, msg.angular.z)

    def publish(self):
        now = self.get_clock().now()
        t_now = self.seconds(now)
        for key, ekf in self.filters.items():
            last = self.last_seen.get(key)
            if (not ekf.initialized or last is None
                    or self.seconds(now - last) > self.max_coast):
                continue
            x, P = ekf.predict(t_now)

            pose = Pose2D()
            pose.x, pose.y, pose.theta = float(x[0]), float(x[1]), float(x[2])
            self.pose_pubs[key].publish(pose)

            odom = Odometry()
            odom.header.stamp = now.to_msg()
            odom.header.frame_id = 'field'
            odom.child_frame_id = f'{key[0]}/robot{key[1]}'
            odom.pose.pose.position.x = pose.x
            odom.pose.pose.position.y = pose.y
            odom.pose.pose.orientation.z = math.sin(pose.theta / 2.0)
            odom.pose.pose.orientation.w = math.cos(pose.theta / 2.0)
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
            self.odom_pubs[key].publish(odom)

    def log_stats(self):
        lines = []
        for (color, i), ekf in self.filters.items():
            total, n = self.latency_sum[(color, i)]
            if n == 0:
                continue
            lines.append(
                f'{color}/robot{i}: updates={ekf.n_updates} rejected={ekf.n_rejected} '
                f'resets={ekf.n_resets} late_dropped={ekf.n_dropped_late} '
                f'mean_latency={1000.0 * total / n:.0f}ms')
        if lines:
            self.get_logger().info('\n  ' + '\n  '.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = VisionFilterNode()
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
