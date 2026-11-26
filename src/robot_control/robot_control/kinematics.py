"""Robot-frame velocity to wheel velocities for an omniwheel base.

Conventions (REP 103): x forward, y left, angular.z positive = CCW from top.
Positive wheel speed = clockwise seen from outside the robot.
Wheel angle = position around the robot, CCW from the front; one entry per
motor id, any number of wheels (the robot's layout comes from its config file).
robot_radius (wheel to robot centre) and wheel_radius take one value, or one
per wheel for a calibrated robot.

The commanded velocity is followed with limited acceleration, so the wheels
keep their grip, and streamed at a fixed rate; without a fresh cmd_vel the
robot brakes to a stop.

Topics:
  cmd_vel     (Twist, in)               robot-frame velocity, m/s and rad/s
  wheels/cmd  (Float32MultiArray, out)  wheel velocities, rad/s
"""
import math

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


class Kinematics(Node):
    def __init__(self):
        super().__init__('kinematics')
        # default: 4 wheels, motor 0 front-right, 1 front-left, 2 back-right, 3 back-left
        self.declare_parameter('wheel_angles_deg', [-60.0, 60.0, -135.0, 135.0])
        scalar_or_list = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('robot_radius', 0.077, scalar_or_list)
        self.declare_parameter('wheel_radius', 0.0245, scalar_or_list)
        self.declare_parameter('max_wheel_rad_s', 80.0)
        self.declare_parameter('max_accel', 3.0)        # m/s^2
        self.declare_parameter('max_ang_accel', 40.0)   # rad/s^2
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('cmd_timeout_s', 0.3)

        p = lambda n: self.get_parameter(n).value
        angles = [math.radians(a) for a in p('wheel_angles_deg')]
        per_wheel = lambda v: ([float(x) for x in v] if isinstance(v, (list, tuple))
                               else [float(v)] * len(angles))
        robot_radius, wheel_radius = per_wheel(p('robot_radius')), per_wheel(p('wheel_radius'))
        self.max_wheel = float(p('max_wheel_rad_s'))
        self.max_accel = float(p('max_accel'))
        self.max_ang_accel = float(p('max_ang_accel'))
        self.dt = 1.0 / float(p('rate_hz'))
        self.cmd_timeout = float(p('cmd_timeout_s'))

        # wheel speed per unit of (vx, vy, w); drive direction = tangent
        # (-sin, cos) at the wheel's position angle
        self.rows = [(-math.sin(a) / r, math.cos(a) / r, d / r)
                     for a, d, r in zip(angles, robot_radius, wheel_radius, strict=True)]

        self.target = (0.0, 0.0, 0.0)
        self.vel = (0.0, 0.0, 0.0)      # velocity being commanded to the wheels
        self.last_cmd_time = None
        self.moving = False

        self.pub = self.create_publisher(Float32MultiArray, 'wheels/cmd', 10)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.create_timer(self.dt, self.step)
        self.get_logger().info(f'Kinematics node up, {len(angles)} wheels')

    def cmd_vel_callback(self, msg):
        self.target = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.last_cmd_time = self.get_clock().now()

    def step(self):
        if self.last_cmd_time is None:
            return
        age = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        tx, ty, tw = self.target if age < self.cmd_timeout else (0.0, 0.0, 0.0)
        vx, vy, w = self.vel

        # a constant velocity over the ground turns in the robot frame while the robot rotates
        c, s = math.cos(w * self.dt), math.sin(w * self.dt)
        vx, vy = c * vx + s * vy, -s * vx + c * vy

        dvx, dvy = tx - vx, ty - vy
        dv, dv_max = math.hypot(dvx, dvy), self.max_accel * self.dt
        if dv > dv_max:
            dvx, dvy = dvx * dv_max / dv, dvy * dv_max / dv
        dw_max = self.max_ang_accel * self.dt
        w += max(-dw_max, min(dw_max, tw - w))
        self.vel = (vx + dvx, vy + dvy, w)

        if not any(self.vel) and not any((tx, ty, tw)):
            if not self.moving:
                return      # idle: the bridge holds zero by itself
            self.moving = False
        else:
            self.moving = True
        self.publish_wheels()

    def publish_wheels(self):
        vx, vy, w = self.vel
        wheels = [kx * vx + ky * vy + kw * w for kx, ky, kw in self.rows]
        # saturate preserving the motion direction
        peak = max(abs(v) for v in wheels)
        if peak > self.max_wheel:
            k = self.max_wheel / peak
            wheels = [v * k for v in wheels]
            self.vel = (vx * k, vy * k, w * k)
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
