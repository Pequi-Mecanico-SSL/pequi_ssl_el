#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray


class WheelPid(Node):
    """Runs per-wheel PID loops to convert velocity targets into PWM commands."""

    def __init__(self) -> None:
        super().__init__("wheel_pid")

        # --------- Robot selection ----------
        self.robot_id = self.declare_parameter("robot_id", 0).value
        self.color = self.declare_parameter("color", "blue").value
        robot_name = f"{self.color}/robot{self.robot_id}"
        self.target_wheel_topic = f"/pid/cmd/wheel_velocity/{robot_name}"

        # --------- Parameters ----------
        self.control_rate_hz = float(
            self.declare_parameter("control_rate_hz", 100.0).value
        )
        output_range = float(self.declare_parameter("output_range", 20.0).value)
        self.output_mid = float(self.declare_parameter("output_mid", 50.0).value)
        self.max_output = self.output_mid + output_range
        self.min_output = self.output_mid - output_range

        self.wheel_kp = float(self.declare_parameter("wheel_kp", 0.1).value)
        self.wheel_ki = float(self.declare_parameter("wheel_ki", 0.4).value)
        self.wheel_kd = float(self.declare_parameter("wheel_kd", 0.0).value)
        self.wheel_integrator_limit = float(
            self.declare_parameter("wheel_integrator_limit", 100.0).value
        )
        self.wheel_derivative_filter_alpha = float(
            self.declare_parameter("wheel_derivative_filter_alpha", 0.5).value
        )

        # --------- State ----------
        self.target_wheel_velocities = np.zeros(4)
        self.current_wheel_velocities = np.zeros(4)
        self.wheel_integrators = np.zeros(4)
        self.wheel_previous_errors = np.zeros(4)
        self.wheel_derivative_filtered = np.zeros(4)

        self.last_update = None
        self.has_target = False
        self.has_feedback = False

        # --------- ROS interfaces ----------
        self.sub_target_wheels = self.create_subscription(
            Float32MultiArray, self.target_wheel_topic, self.target_wheel_callback, 10
        )
        self.sub_current_wheels = self.create_subscription(
            Float32MultiArray, "/encoder_values", self.encoder_callback, 10
        )
        self.pub_pwm = self.create_publisher(Float32MultiArray, "/motor_commands", 10)

        self.dt = 1.0 / self.control_rate_hz
        self.timer = self.create_timer(self.dt, self.pid_loop)
        self.print_freq = 10.0
        self.last_print_time = self.get_clock().now()

        self.get_logger().info("Wheel PID node up")

    def target_wheel_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 4:
            self.get_logger().warn("Received wheel target message with <4 elements")
            return
        incoming = np.array(msg.data[:4], dtype=float)
        previous = self.target_wheel_velocities.copy()
        if self.has_target:
            for i in range(4):
                if np.sign(previous[i]) != np.sign(incoming[i]):
                    self.wheel_integrators[i] = 0.0
        self.target_wheel_velocities = incoming
        self.has_target = True

    def encoder_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 4:
            self.get_logger().warn("Received encoder message with <4 elements")
            return
        self.current_wheel_velocities = np.array(msg.data[:4], dtype=float)
        self.has_feedback = True

    def pid_loop(self) -> None:
        if not (self.has_target and self.has_feedback):
            return

        now = self.get_clock().now()
        if self.last_update is None:
            self.last_update = now
            return

        dt = (now - self.last_update).nanoseconds * 1e-9
        if dt <= 0.0:
            dt = self.dt
        self.last_update = now

        output = np.zeros(4)
        for i in range(4):
            error = self.target_wheel_velocities[i] - self.current_wheel_velocities[i]

            # Integrator with clamp (anti-windup)
            self.wheel_integrators[i] += self.wheel_ki * error * dt
            if abs(self.wheel_integrators[i]) > self.wheel_integrator_limit:
                self.wheel_integrators[i] = np.sign(self.wheel_integrators[i]) * self.wheel_integrator_limit

            # Derivative with low-pass
            error_derivative = (error - self.wheel_previous_errors[i]) / max(dt, 1e-6)
            self.wheel_derivative_filtered[i] = (
                self.wheel_derivative_filter_alpha * self.wheel_derivative_filtered[i]
                + (1.0 - self.wheel_derivative_filter_alpha) * error_derivative
            )
            self.wheel_previous_errors[i] = error

            pid_output = self.wheel_kp * error + self.wheel_integrators[i] \
                + self.wheel_kd * self.wheel_derivative_filtered[i]

            output[i] = np.clip(pid_output + self.output_mid, self.min_output, self.max_output)

        if (now - self.last_print_time).nanoseconds * 1e-9 >= 1.0 / self.print_freq:
            self.get_logger().info(
                f"PWM: {output[0]:.1f}, {output[1]:.1f}, {output[2]:.1f}, {output[3]:.1f}"
            )
            self.last_print_time = now

        msg = Float32MultiArray()
        msg.data = output.astype(np.float32).tolist()
        self.pub_pwm.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WheelPid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
