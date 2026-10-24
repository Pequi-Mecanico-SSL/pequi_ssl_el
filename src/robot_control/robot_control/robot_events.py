"""Collision / skid / stall / push detection (EventDetector) as a ROS node.

Topics:
  imu                    (Imu, in)                gyro z + accelerometer
  wheels/state           (JointState, in)         wheel speed (velocity) + current (effort)
  wheels/cmd             (Float32MultiArray, in)  wheel setpoints from kinematics
  odom/wheel_residuals   (Float32MultiArray, in)  from the odometry node
  events                 (String, out)            one JSON object per event, e.g.
      {"event": "impact", "kind": "linear", "t": ..., "accel": 23.1,
       "saturated": false, "direction_deg": -178.0}
      {"event": "skid", "state": "start", "wheel": 2, "candidates": [1, 2], "t": ...}
      {"event": "stall", "state": "start", "wheels": [0, 1, 2, 3], "t": ...}
      {"event": "push", "state": "end", "t": ...}
  events/status          (String, out, 10 Hz)     JSON of what is active now:
      {"skid": [2], "stalled": false, "pushed": false, "last_impact": 1790434499.2}

Watch it live:  ros2 topic echo /events
"""
import json

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray, String

from .event_detector import EventDetector
from .geometry import (DEFAULT_ROBOT_RADIUS, DEFAULT_WHEEL_ANGLES_DEG,
                       DEFAULT_WHEEL_RADIUS, wheel_matrix)

DETECTOR_PARAMS = {
    'current_limit': 1.3,
    'impact_accel': 8.0,
    'impact_ang_accel': 60.0,
    'sigma_wheel': 0.3,
    'skid_sigmas': 3.0,
    'stall_min_error': 0.3,
    'stall_current_frac': 0.8,
    'stall_max_progress': 0.5,
    'stall_window_s': 0.1,
    'stall_on_s': 0.15,
    'stall_off_s': 0.2,
    'push_speed': 0.1,
    'push_current_frac': 0.4,
    'settle_s': 0.5,
}


class RobotEvents(Node):
    def __init__(self):
        super().__init__('robot_events')
        self.declare_parameter('wheel_angles_deg', DEFAULT_WHEEL_ANGLES_DEG)
        self.declare_parameter('robot_radius', DEFAULT_ROBOT_RADIUS)
        self.declare_parameter('wheel_radius', DEFAULT_WHEEL_RADIUS)
        # imu axis giving robot forward and left, e.g. ['-y', '+x']; verify
        # on the robot (see EVENTS.md) before trusting direction_deg
        self.declare_parameter('accel_axes', ['+x', '+y'])
        for name, default in DETECTOR_PARAMS.items():
            self.declare_parameter(name, default)

        p = lambda n: self.get_parameter(n).value
        rows = wheel_matrix(p('wheel_angles_deg'), float(p('robot_radius')),
                            float(p('wheel_radius')))
        kwargs = {n: p(n) for n in DETECTOR_PARAMS}
        self.detector = EventDetector(rows, robot_radius=float(p('robot_radius')),
                                      accel_axes=tuple(p('accel_axes')), **kwargs)

        self.events_pub = self.create_publisher(String, 'events', 10)
        self.status_pub = self.create_publisher(String, 'events/status', 10)
        self.create_subscription(Imu, 'imu', self.imu_callback, 50)
        self.create_subscription(JointState, 'wheels/state', self.wheels_callback, 10)
        self.create_subscription(Float32MultiArray, 'wheels/cmd', self.cmd_callback, 10)
        self.create_subscription(Float32MultiArray, 'odom/wheel_residuals',
                                 self.residuals_callback, 10)
        self.create_timer(0.1, self.publish_status)
        self.get_logger().info(f'robot events up: {kwargs}')

    def stamp_s(self, header=None):
        t = Time.from_msg(header.stamp) if header is not None else None
        if t is None or t.nanoseconds == 0:
            t = self.get_clock().now()
        return t.nanoseconds * 1e-9

    def emit(self, events):
        for e in events:
            self.events_pub.publish(String(data=json.dumps(e)))
            self.get_logger().info(f'EVENT {e}')

    def imu_callback(self, msg):
        a = msg.linear_acceleration
        self.emit(self.detector.imu(self.stamp_s(msg.header), msg.angular_velocity.z,
                                    (a.x, a.y, a.z)))

    def cmd_callback(self, msg):
        if len(msg.data) == self.detector.n:
            self.detector.wheel_setpoints(self.stamp_s(), list(msg.data))

    def wheels_callback(self, msg):
        n = self.detector.n
        if len(msg.velocity) != n:
            return
        currents = list(msg.effort) if len(msg.effort) == n else [float('nan')] * n
        self.emit(self.detector.wheels(self.stamp_s(msg.header), list(msg.velocity), currents))

    def residuals_callback(self, msg):
        if len(msg.data) == self.detector.n:
            self.emit(self.detector.residuals(self.stamp_s(), list(msg.data)))

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self.detector.status())))


def main(args=None):
    rclpy.init(args=args)
    node = RobotEvents()
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
