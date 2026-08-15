"""Robot-frame velocity to wheel velocities for the 4-omniwheel base.

Conventions (REP 103): x forward, y left, angular.z positive = CCW from top.
Positive wheel speed = clockwise seen from outside the robot.
Wheel angle = position around the robot, CCW from the front.

Topics:
  cmd_vel     (Twist, in)               robot-frame velocity, m/s and rad/s
  wheels/cmd  (Float32MultiArray, out)  4x wheel velocity, rad/s
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


class Kinematics(Node):
    def __init__(self):
        super().__init__('kinematics')
        # motor order: 0 front-right, 1 front-left, 2 back-right, 3 back-left
        self.declare_parameter('wheel_angles_deg', [-60.0, 60.0, -135.0, 135.0])
        self.declare_parameter('robot_radius', 0.0875)
        self.declare_parameter('wheel_radius', 0.0245)
        self.declare_parameter('max_wheel_rad_s', 40.0)

        angles = [math.radians(a) for a in self.get_parameter('wheel_angles_deg').value]
        self.robot_radius = float(self.get_parameter('robot_radius').value)
        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.max_wheel = float(self.get_parameter('max_wheel_rad_s').value)

        # wheel drive direction = tangent (-sin, cos) at its position angle
        self.rows = [(-math.sin(a), math.cos(a)) for a in angles]

        self.pub = self.create_publisher(Float32MultiArray, 'wheels/cmd', 10)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.get_logger().info('Kinematics node up')

    def cmd_vel_callback(self, msg):
        vx, vy, w = msg.linear.x, msg.linear.y, msg.angular.z
        wheels = [(tx * vx + ty * vy + self.robot_radius * w) / self.wheel_radius
                  for tx, ty in self.rows]
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
