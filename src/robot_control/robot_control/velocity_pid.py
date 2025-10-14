#!/usr/bin/env python3
import math
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


class MimoPidOmni(Node):
    def __init__(self):
        super().__init__("mimo_pid_omni")

        # --------- Robot selection ----------
        self.robot_id = self.declare_parameter("robot_id", 0).value
        self.color = self.declare_parameter("color", "blue").value
        # Angle where each wheel is located (0 is x axis, pointing to the front of robot)
        default_wheel_orientation = [
            math.pi / 3, # 60 degrees
            3 * math.pi / 4, # 135 degrees
            5 * math.pi / 4, # 225 degrees
            5 * math.pi / 3, # 300 degrees
        ]
        self.wheel_orientation: List[float] = self.declare_parameter(
            "wheel_orientation", default_wheel_orientation
        ).value
        self.wheel_orientation = np.array(self.wheel_orientation, dtype=float)

        # self.robot_radius = float(self.declare_parameter("robot_radius", 0.09).value)  # [m]
        self.robot_radius = float(self.declare_parameter("robot_radius", 0.175/2).value)  # [m]

        # --------- Useful extra params ----------
        self.control_rate_hz = float(self.declare_parameter("control_rate_hz", 1000.0).value)
        self.wheel_radius = float(self.declare_parameter("wheel_radius", 0.049/2.0).value)  # [m]
        output_range = float(self.declare_parameter("output_range", 25.0).value)
        self.output_mid = float(self.declare_parameter("output_mid", 50.0).value)
        self.max_output = self.output_mid + output_range
        self.min_output = self.output_mid - output_range

        # PID gains as 3x3 (MIMO). Defaults are diagonal.
        kp_default = [0.1, 0.1, 0.1]
        ki_default = [0.0, 0.0, 0.0]
        kd_default = [0.0, 0.0, 0.0]

        self.Kp =   [[kp_default[0], 0.0, 0.0],
                    [0.0, kp_default[1], 0.0],
                    [0.0, 0.0, kp_default[2]]]

        self.Ki =   [[ki_default[0], 0.0, 0.0],
                    [0.0, ki_default[1], 0.0],
                    [0.0, 0.0, ki_default[2]]]

        self.Kd =   [[kd_default[0], 0.0, 0.0],
                    [0.0, kd_default[1], 0.0],
                    [0.0, 0.0, kd_default[2]]]
        
        self.integrator_limit = float(self.declare_parameter("integrator_limit", 10.0).value)
        self.derivative_filter_alpha = float(self.declare_parameter("derivative_filter_alpha", 0.5).value)
        
        self.target_velocity = np.zeros(3)  # [vx, vy, omega]
        self.current_velocity = np.zeros(3)  # [vx, vy, omega]
        self.derivative_filtered = np.zeros(3)
        self.integrator = np.zeros(3)
        self.previous_error = 0
        self.derivative = np.zeros(3)

        # Assignment matrix
        sin = np.sin(self.wheel_orientation)
        cos = np.cos(self.wheel_orientation)
        jacobian_wheel_linear_vel = np.array([
            [-sin[0], -cos[0], -self.robot_radius],
            [-sin[1], -cos[1], -self.robot_radius],
            [-sin[2], -cos[2], -self.robot_radius],
            [-sin[3], -cos[3], -self.robot_radius],
        ], dtype=float)
        self.jacobian_wheel_ang_vel = jacobian_wheel_linear_vel / self.wheel_radius

        # --------- Topics ----------
        robot_name = f"{self.color}/robot{self.robot_id}"
        #self.sub_current_vel = self.create_subscription(
        #    Twist, f"/simulator/velocity/{robot_name}", self.current_vel_callback, 10
        #)
        self.sub_target_vel = self.create_subscription(
            Twist, f"/pid/cmd/velocity/{robot_name}", self.target_vel_callback, 10
        )
        self.pub_pwm = self.create_publisher(
            Float32MultiArray, f"/motor_commands", 10
        )
        self.sub_current_wheel_vel = self.create_subscription(
            Float32MultiArray, f"/encoder_values", self.current_wheel_vel_callback, 10
        )
        self.pub_current_vel_from_wheel = self.create_publisher(
            Twist, f"/velocity_from_wheels", 10
        )

        # --------- State ----------
        self.last_update: Optional[Time] = None

        # Control loop timer
        self.dt = 1.0 / self.control_rate_hz
        self.timer = self.create_timer(self.dt, self.pid_loop)
        self.get_logger().info(
            f"MIMO PID Omni controller up"
        )

    # -------------------- Callbacks --------------------
    def current_vel_callback(self, msg: Twist):
        #self.get_logger().info(f"Vel: [{msg.linear.x:.2f},\t{msg.linear.y:.2f},\t{msg.angular.z:.2f}]")
        #self.current_velocity = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=float)
        pass

    def target_vel_callback(self, msg: Twist):
        # self.get_logger().info(f"Got cmd: [{msg.linear.x:.2f},\t{msg.linear.y:.2f},\t{msg.angular.z:.2f}]")
        # cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=float)
        # wheel_output = self.jacobian_wheel_ang_vel @ cmd
        # self.get_logger().info(f"Cmd (rad/s): [{wheel_output[0]:.2f}, {wheel_output[1]:.2f}, {wheel_output[2]:.2f}, {wheel_output[3]:.2f}]")
        self.target_velocity = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=float)
    
    def current_wheel_vel_callback(self, msg: Float32MultiArray):
        wheel_vels = np.array(msg.data, dtype=float)
        self.current_velocity = np.linalg.pinv(self.jacobian_wheel_ang_vel) @ wheel_vels
        # Publish inferred velocity
        twist = Twist()
        twist.linear.x = float(self.current_velocity[0])
        twist.linear.y = float(self.current_velocity[1])
        twist.angular.z = float(self.current_velocity[2])
        self.pub_current_vel_from_wheel.publish(twist)

    # -------------------- Control --------------------
    def pid_loop(self):
        now = self.get_clock().now()
        if self.last_update is None:
            self.last_update = now
            return

        dt = (now - self.last_update).nanoseconds * 1e-9
        if dt <= 0.0:
            dt = self.dt
        self.last_update = now

        error = self.target_velocity - self.current_velocity

        # Integrator with clamp (anti-windup)
        self.integrator += error * dt
        integrator_norm = np.linalg.norm(self.integrator)
        if integrator_norm > self.integrator_limit:
            self.integrator *= self.integrator_limit / integrator_norm
        
        # Derivative with low-pass
        error_derivative = (error - self.previous_error) / max(dt, 1e-6)
        self.derivative_filtered = self.derivative_filter_alpha * self.derivative_filtered \
            + (1.0 - self.derivative_filter_alpha) * error_derivative
        self.previous_error = error

        pid_output = self.Kp @ error + self.Ki @ self.integrator + self.Kd @ self.derivative_filtered

        wheel_output = self.jacobian_wheel_ang_vel @ pid_output
        output = np.clip(wheel_output + self.output_mid, self.min_output, self.max_output)

        # Publish
        msg = Float32MultiArray()
        msg.data = output.astype(np.float32).tolist()
        # self.get_logger().info(f"PWM: {msg.data[0]:.1f}, {msg.data[1]:.1f}, {msg.data[2]:.1f}, {msg.data[3]:.1f}")
        # self.pub_pwm.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MimoPidOmni()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
