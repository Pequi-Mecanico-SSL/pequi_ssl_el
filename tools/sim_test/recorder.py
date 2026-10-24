"""Records simulator truth and stack outputs for `seconds`, then pickles them.

    python3 recorder.py <out.pkl> <seconds> [stack_ns]
"""
import json
import pickle
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray, Float64, Float64MultiArray, String

COLORS = ('blue', 'yellow')


def stamp_s(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def yaw(q):
    import math
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Recorder(Node):
    def __init__(self, ns):
        from rclpy.parameter import Parameter
        super().__init__('recorder', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.d = {}
        now = self.now
        rec = lambda name, row: self.d.setdefault(name, []).append(row)
        self.rec = rec
        for c in COLORS:
            for i in range(3):
                k = f'{c}/robot{i}'
                self.create_subscription(Pose2D, f'/vision/poses/{k}',
                                         lambda m, k=k: rec(f'raw/{k}', (now(), m.x, m.y, m.theta)), 50)
                self.create_subscription(Odometry, f'/vision/filtered/odom/{k}',
                                         lambda m, k=k: rec(f'filt/{k}', self.odom_row(m)), 50)
                self.create_subscription(Odometry, f'/vision/filtered_nohint/odom/{k}',
                                         lambda m, k=k: rec(f'filt_nohint/{k}', self.odom_row(m)), 50)
                self.create_subscription(PoseStamped, f'/vision/stamped/{k}',
                                         lambda m, k=k: rec(f'stamped/{k}', (stamp_s(m.header), now(), m.pose.position.x,
                                                                             m.pose.position.y, yaw(m.pose.orientation))), 50)
                self.create_subscription(Twist, f'/{k}/cmd_vel',
                                         lambda m, k=k: rec(f'cmd/{k}', (now(), m.linear.x, m.linear.y, m.angular.z)), 50)
        self.create_subscription(String, '/driver/phase',
                                 lambda m: rec('phase', (now(), json.loads(m.data))), 50)
        # [t (sim axis), step, rSim state...]
        self.create_subscription(Float64MultiArray, '/sim_harness/truth',
                                 lambda m: rec('truth', tuple(m.data)), 50)
        self.create_subscription(Pose2D, '/vision/poses/ball',
                                 lambda m: rec('raw_ball', (now(), m.x, m.y)), 50)
        if ns:
            self.create_subscription(Odometry, f'{ns}/odom', lambda m: rec('odom', self.odom_row(m)), 50)
            self.create_subscription(Float64, f'{ns}/odom/gyro_bias', lambda m: rec('bias', (now(), m.data)), 50)
            self.create_subscription(Float64, f'{ns}/odom/ground_truth/gyro_bias',
                                     lambda m: rec('true_bias', (now(), m.data)), 50)
            self.create_subscription(String, f'{ns}/events', lambda m: rec('events', (now(), json.loads(m.data))), 50)
            self.create_subscription(Twist, f'{ns}/cmd_vel_corrected',
                                     lambda m: rec('cmd_corr', (now(), m.linear.x, m.linear.y, m.angular.z)), 50)
            self.create_subscription(Float32MultiArray, f'{ns}/wheels/cmd',
                                     lambda m: rec('wheels_cmd', (now(), *m.data)), 50)
            self.create_subscription(Imu, f'{ns}/imu', lambda m: rec('imu', (
                stamp_s(m.header), now(), m.angular_velocity.z, m.linear_acceleration.x,
                m.linear_acceleration.y)), 100)
            self.create_subscription(JointState, f'{ns}/wheels/state', lambda m: rec('wheels', (
                stamp_s(m.header), now(), *m.velocity, *m.effort)), 100)
            self.create_subscription(Float32MultiArray, f'{ns}/odom/wheel_residuals',
                                     lambda m: rec('residuals', (now(), *m.data)), 50)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def odom_row(m):
        p, tw = m.pose.pose, m.twist.twist
        return (stamp_s(m.header), p.position.x, p.position.y, yaw(p.orientation),
                tw.linear.x, tw.linear.y, tw.angular.z)


def main():
    out, seconds = sys.argv[1], float(sys.argv[2])
    ns = sys.argv[3] if len(sys.argv) > 3 else ''
    rclpy.init()
    node = Recorder(ns)
    end = time.time() + seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    with open(out, 'wb') as f:
        pickle.dump(node.d, f)
    print('recorded', {k: len(v) for k, v in sorted(node.d.items()) if not k.startswith('cmd/')})
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
