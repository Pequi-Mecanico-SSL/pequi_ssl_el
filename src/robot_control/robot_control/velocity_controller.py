"""Closed-loop yaw control using the IMU gyro.

Sits between the velocity source and kinematics: vx/vy pass through,
angular.z gets rate feedback, and while no rotation is commanded the
integrated heading is held (drives straight during translation). The gyro
bias auto-calibrates whenever the robot is idle. If IMU data goes stale
the commanded rate passes through open-loop.

Pipeline: cmd_vel -> [this node] -> cmd_vel_corrected -> kinematics.
Output is published per input message, so the bridge deadman chain from
the teleop stream stays intact.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')
        self.declare_parameter('yaw_kp', 0.5)             # rate error -> rad/s
        self.declare_parameter('heading_hold', True)
        self.declare_parameter('heading_kp', 3.0)         # heading error -> rad/s
        self.declare_parameter('max_hold_rate', 2.0)      # rad/s
        self.declare_parameter('correction_deadband', 0.05)  # rad/s
        self.declare_parameter('bias_tau_s', 10.0)
        self.declare_parameter('bias_max_dev', 0.1)       # rad/s, don't learn beyond this
        self.declare_parameter('idle_time_s', 1.0)

        p = lambda n: self.get_parameter(n).value
        self.yaw_kp = float(p('yaw_kp'))
        self.heading_hold = bool(p('heading_hold'))
        self.heading_kp = float(p('heading_kp'))
        self.max_hold_rate = float(p('max_hold_rate'))
        self.deadband = float(p('correction_deadband'))
        self.bias_tau = float(p('bias_tau_s'))
        self.bias_max_dev = float(p('bias_max_dev'))
        self.idle_time = float(p('idle_time_s'))

        self.bias = 0.0
        self.bias_samples = 0
        self.omega = 0.0
        self.heading = 0.0
        self.hold_target = 0.0
        self.last_imu_time = None
        self.last_active_time = None

        self.pub = self.create_publisher(Twist, 'cmd_vel_corrected', 10)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_callback, 10)
        self.create_subscription(Imu, 'imu', self.imu_callback, 50)
        self.create_subscription(Empty, 'wheels/estop', self.reset_hold, 10)
        self.create_subscription(Empty, 'wheels/enable', self.reset_hold, 10)
        self.get_logger().info('Velocity controller up')

    def reset_hold(self, _=None):
        self.hold_target = self.heading

    def imu_stale(self):
        return (self.last_imu_time is None
                or (self.get_clock().now() - self.last_imu_time).nanoseconds / 1e9 > 0.2)

    def imu_callback(self, msg):
        now = self.get_clock().now()
        dt = 0.0
        if self.last_imu_time is not None:
            dt = (now - self.last_imu_time).nanoseconds / 1e9
        self.last_imu_time = now
        if not 0.0 < dt < 0.05:
            return

        raw = msg.angular_velocity.z
        idle = (self.last_active_time is None
                or (now - self.last_active_time).nanoseconds / 1e9 > self.idle_time)
        if idle and abs(raw - self.bias) < self.bias_max_dev:
            self.bias += (dt / self.bias_tau) * (raw - self.bias)
            self.bias_samples += 1
            if self.bias_samples == 2000:
                self.get_logger().info(f'gyro bias calibrated: {self.bias:.4f} rad/s')

        self.omega = raw - self.bias
        self.heading = wrap(self.heading + self.omega * dt)

    def cmd_callback(self, msg):
        vx, vy, cmd_w = msg.linear.x, msg.linear.y, msg.angular.z
        now = self.get_clock().now()
        if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(cmd_w) > 0.05:
            self.last_active_time = now

        out = Twist()
        out.linear.x, out.linear.y = vx, vy

        if self.imu_stale():
            self.get_logger().warning('IMU stale, yaw control open-loop',
                                      throttle_duration_sec=5.0)
            out.angular.z = cmd_w
            self.pub.publish(out)
            return

        translating = abs(vx) > 0.01 or abs(vy) > 0.01
        if abs(cmd_w) > 0.05:
            omega_sp = cmd_w
            self.hold_target = self.heading
        elif translating and self.heading_hold:
            err = wrap(self.hold_target - self.heading)
            omega_sp = max(-self.max_hold_rate, min(self.max_hold_rate,
                                                    self.heading_kp * err))
        else:
            # idle: follow instead of hold, so stale targets never command motion
            self.hold_target = self.heading
            omega_sp = 0.0

        omega_out = omega_sp + self.yaw_kp * (omega_sp - self.omega)
        if abs(cmd_w) < 0.05 and abs(omega_out) < self.deadband:
            omega_out = 0.0
        out.angular.z = omega_out
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityController()
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
