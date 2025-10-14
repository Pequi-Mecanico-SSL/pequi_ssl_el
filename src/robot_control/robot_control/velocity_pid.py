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
from sensor_msgs.msg import Imu


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
        output_range = float(self.declare_parameter("output_range", 40.0).value)
        self.output_mid = float(self.declare_parameter("output_mid", 50.0).value)
        self.max_output = self.output_mid + output_range
        self.min_output = self.output_mid - output_range

        # PID gains as 3x3 (MIMO). Defaults are diagonal.
        kp_default = [0.2, 0.2, 1.0]
        ki_default = [0.0, 0.0, 0.4]
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
        
        self.target_velocity = None # [vx, vy, omega]
        self.current_velocity = None # [vx, vy, omega]
        self.received_from_imu = False
        self.derivative_filtered = np.zeros(3)
        self.integrator = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.derivative = np.zeros(3)

        self.last_movement_time = self.get_clock().now()
        self.min_time_still_to_calibrate_imu = Duration(seconds=1.0)

        self.print_freq = 4.0  # Hz
        self.last_print_time = self.get_clock().now()
        self.ang_vel_offset = 0.0
        self.offset_measurements = 0

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
        self.sub_imu = self.create_subscription(
            Imu, f"/imu", self.imu_callback, 10
        )

        # --------- State ----------
        self.last_update: Optional[Time] = None

        # Control loop timer
        self.dt = 1.0 / self.control_rate_hz
        self.timer = self.create_timer(self.dt, self.pid_loop)
        self.get_logger().info(
            f"MIMO PID Omni controller up"
        )
    
    def imu_callback(self, msg: Imu):
        if self.current_velocity is None:
            return
        if (abs(msg.angular_velocity.x) > 0.1 or
            abs(msg.angular_velocity.y) > 0.1):
            self.last_movement_time = self.get_clock().now()
        if (self.get_clock().now() - self.last_movement_time) > self.min_time_still_to_calibrate_imu and self.offset_measurements < 1000:
            self.ang_vel_offset = 0.99 * self.ang_vel_offset + 0.01 * msg.angular_velocity.z
            self.offset_measurements += 1
            if self.offset_measurements == 1000:
                self.get_logger().info(f"Calibrated gyro offset: {self.ang_vel_offset:.4f}")
        self.current_velocity[2] = msg.angular_velocity.z - self.ang_vel_offset
        self.received_from_imu = True

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
        if (abs(msg.linear.x) > 0.01 or
            abs(msg.linear.y) > 0.01 or
            abs(msg.angular.z) > 0.1):
            self.last_movement_time = self.get_clock().now()
    
    def current_wheel_vel_callback(self, msg: Float32MultiArray):
        wheel_vels = np.array(msg.data, dtype=float)
        current_velocity_from_wheel = np.linalg.pinv(self.jacobian_wheel_ang_vel) @ wheel_vels
        if self.current_velocity is None:
            self.current_velocity = np.zeros(3)
        # Update only x and y from wheel encoders, angular velocity from IMU if available
        self.current_velocity[0] = current_velocity_from_wheel[0]
        self.current_velocity[1] = current_velocity_from_wheel[1]
        if not self.received_from_imu:
            self.current_velocity[2] = current_velocity_from_wheel[2]
        # Publish inferred velocity
        twist = Twist()
        twist.linear.x = float(current_velocity_from_wheel[0])
        twist.linear.y = float(current_velocity_from_wheel[1])
        twist.angular.z = float(current_velocity_from_wheel[2])
        self.pub_current_vel_from_wheel.publish(twist)

    # -------------------- Control --------------------
    def pid_loop(self):
        if self.target_velocity is None or self.current_velocity is None or self.offset_measurements < 1000:
            return
        
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

        if (now - self.last_print_time).nanoseconds * 1e-9 >= 1.0 / self.print_freq:
            # self.get_logger().info(f"PWM: {msg.data[0]:.1f}, {msg.data[1]:.1f}, {msg.data[2]:.1f}, {msg.data[3]:.1f}")
            # For orientation
            self.get_logger().info(f"P: {error[2]:.2f}, I: {self.integrator[2]:.2f}, D: {self.derivative_filtered[2]:.2f}")
            self.last_print_time = now

        self.pub_pwm.publish(msg)


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
