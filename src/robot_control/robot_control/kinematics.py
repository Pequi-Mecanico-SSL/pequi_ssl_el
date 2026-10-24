"""Robot-frame velocity to wheel velocities for the 4-omniwheel base.

Conventions (REP 103): x forward, y left, angular.z positive = CCW from top.
Positive wheel speed = clockwise seen from outside the robot.
Wheel angle = position around the robot, CCW from the front.

Topics:
  cmd_vel     (Twist, in)               robot-frame velocity, m/s and rad/s
  wheels/cmd  (Float32MultiArray, out)  4x wheel velocity, rad/s
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from .geometry import (DEFAULT_ROBOT_RADIUS, DEFAULT_WHEEL_ANGLES_DEG,
                       DEFAULT_WHEEL_RADIUS, wheel_matrix)


class Kinematics(Node):
    def __init__(self):
        super().__init__('kinematics')
        self.declare_parameter('wheel_angles_deg', DEFAULT_WHEEL_ANGLES_DEG)
        self.declare_parameter('robot_radius', DEFAULT_ROBOT_RADIUS)
        self.declare_parameter('wheel_radius', DEFAULT_WHEEL_RADIUS)
        self.declare_parameter('max_wheel_rad_s', 40.0)

        self.rows = wheel_matrix(self.get_parameter('wheel_angles_deg').value,
                                 float(self.get_parameter('robot_radius').value),
                                 float(self.get_parameter('wheel_radius').value))
        self.max_wheel = float(self.get_parameter('max_wheel_rad_s').value)

        self.pub = self.create_publisher(Float32MultiArray, 'wheels/cmd', 10)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.get_logger().info('Kinematics node up')

    def cmd_vel_callback(self, msg):
        vx, vy, w = msg.linear.x, msg.linear.y, msg.angular.z
        wheels = [jx * vx + jy * vy + jw * w for jx, jy, jw in self.rows]
        # saturate preserving the motion direction
        peak = max(abs(v) for v in wheels)
        if peak > self.max_wheel:
            wheels = [v * self.max_wheel / peak for v in wheels]
        out = Float32MultiArray()
        out.data = [float(v) for v in wheels]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = Kinematics()
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
