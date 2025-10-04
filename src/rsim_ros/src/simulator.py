import robosim
from typing import List
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Twist
from std_msgs.msg import Float32MultiArray

class SimulatorNode(Node):

    def __init__(self):
        super().__init__('simulator')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('blue_robot_count', 3),
                ('yellow_robot_count', 3),
                ('frequency', 60),
            ]
        )
        self.blue_robot_count = self.get_parameter('blue_robot_count').get_parameter_value().integer_value
        self.yellow_robot_count = self.get_parameter('yellow_robot_count').get_parameter_value().integer_value
        frequency = self.get_parameter('frequency').get_parameter_value().integer_value
        time_step_ms = 1000 // frequency

        field_type = 1  # 0 for Division A, 1 for Division B, 2 Hardware Challenges
        # ball initial position [x, y, v_x, v_y] in meters and meter/s
        ball_pos = [0.0, 0.0, 0.0, 0.0]

        # robots initial positions [[x, y, angle], [x, y, angle]...], where [[id_0], [id_1]...]
        # Units are meters and degrees
        blue_robots_pos = [[-1 * i - 1, 0.0, 0.0] for i in range(self.blue_robot_count)]
        yellow_robots_pos = [[1 * i + 1, 0.0, 0.0] for i in range(self.yellow_robot_count)]

        # Init simulator
        self.sim = robosim.SSL(
            field_type,
            self.blue_robot_count,
            self.yellow_robot_count,
            time_step_ms,
            ball_pos,
            blue_robots_pos,
            yellow_robots_pos,
        )

        self.ball_publisher = self.create_publisher(Pose2D, '/simulator/poses/ball', 10)

        # /simulator/robots_poses/blue/0, /simulator/robots_poses/blue/1, ...
        self.pose_publishers = []
        self.velocity_publishers = []
        self.wheel_velocity_publishers = []
        for i in range(self.blue_robot_count):
            self.pose_publishers.append(
                self.create_publisher(Pose2D, f'/simulator/poses/blue/robot{i}', 10)
            )
            self.velocity_publishers.append(
                self.create_publisher(Twist, f'/simulator/velocity/blue/robot{i}', 10)
            )
            self.wheel_velocity_publishers.append(
                self.create_publisher(Float32MultiArray, f'/simulator/wheel_velocity/blue/robot{i}', 10)
            )
        for i in range(self.yellow_robot_count):
            self.pose_publishers.append(
                self.create_publisher(Pose2D, f'/simulator/poses/yellow/robot{i}', 10)
            )
            self.velocity_publishers.append(
                self.create_publisher(Twist, f'/simulator/velocity/yellow/robot{i}', 10)
            )
            self.wheel_velocity_publishers.append(
                self.create_publisher(Float32MultiArray, f'/simulator/wheel_velocity/yellow/robot{i}', 10)
            )

        self.latest_robot_actions = [[0.0 for _ in range(6)] for _ in range(self.blue_robot_count + self.yellow_robot_count)]
        
        for i in range(self.blue_robot_count):
            self.create_subscription(Twist, f'/simulator/cmd/blue/robot{i}',
                                        lambda msg, i=i: self.robot_cmd_callback(i, msg), 10)
            self.create_subscription(Float32MultiArray, f'/simulator/cmd/wheel/blue/robot{i}',
                                        lambda msg, i=i: self.robot_wheel_cmd_callback(i, msg), 10)
        for i in range(self.yellow_robot_count):
            self.create_subscription(Twist, f'/simulator/cmd/yellow/robot{i}',
                                        lambda msg, i=i: self.robot_cmd_callback(self.blue_robot_count + i, msg), 10)
            self.create_subscription(Float32MultiArray, f'/simulator/cmd/wheel/yellow/robot{i}',
                                        lambda msg, i=i: self.robot_wheel_cmd_callback(self.blue_robot_count + i, msg), 10)
    
        timer_period = time_step_ms / 1000.0  # seconds
        self.timer = self.create_timer(timer_period, self.update_state_and_publish)

        self.get_logger().info('Simulator Node Started')

    def robot_wheel_cmd_callback(self, i, msg):
        # self.get_logger().info(f'Robot {i} wheel command received: {msg.data}')
        self.latest_robot_actions[i] = [
            1, # has_v_wheel
            msg.data[0], # wheel_0_speed or v_x
            msg.data[1], # wheel_1_speed or v_y
            msg.data[2], # wheel_2_speed or v_angle
            msg.data[3], # wheel_3_speed (used because has_v_wheel is 1)
            0, # kick_v_x
            0, # kick_v_y
            0, # dribbler
        ]

    def robot_cmd_callback(self, i, msg):
        # self.get_logger().info(f'Robot {i} velocity command received: [{msg.linear.x}, {msg.linear.y}, {msg.angular.z}]')
        self.latest_robot_actions[i] = [
            0, # has_v_wheel
            msg.linear.x, # wheel_0_speed or v_x
            msg.linear.y, # wheel_1_speed or v_y
            msg.angular.z, # wheel_2_speed or v_angle
            0, # wheel_3_speed (not used because has_v_wheel is 0)
            0, # kick_v_x
            0, # kick_v_y
            0, # dribbler
        ]

    # convert degrees to radians, to range [-pi, pi]
    def to_rad(self, deg):
        rad = np.deg2rad(deg)
        if rad > np.pi:
            rad -= 2 * np.pi
        return rad

    def update_state_and_publish(self):
        self.sim.step(self.latest_robot_actions)

        # Units are meters, meters/s, degrees
        # state is [ball_x, ball_y, ball_z, ball_v_x, ball_v_y,
        #           blue_0_x, blue_0_y, blue_0_angle, blue_0_v_x, blue_0_v_y, blue_0_v_angle,
        #           blue_0_infrared, blue_0_desired_wheel0_speed, blue_0_desired_wheel1_speed,
        #           blue_0_desired_wheel2_speed, blue_0_desired_wheel3_speed, ...]
        state = self.sim.get_state()

        ball_msg = Pose2D()
        ball_msg.x = state[0]
        ball_msg.y = state[1]
        ball_msg.theta = 0.0
        self.ball_publisher.publish(ball_msg)

        # self.get_logger().info(f'Blue 0 desired wheel speeds: {state[12:16]}')

        for i in range(self.blue_robot_count):
            msg = Pose2D()
            msg.x = state[5 + i * 11]
            msg.y = state[6 + i * 11]
            # msg.theta = state[7 + i * 11]
            theta_deg = state[7 + i * 11]
            msg.theta = self.to_rad(theta_deg)
            self.pose_publishers[i].publish(msg)

            twist_msg = Twist()
            twist_msg.linear.x = state[8 + i * 11]
            twist_msg.linear.y = state[9 + i * 11]
            twist_msg.angular.z = state[10 + i * 11] * (np.pi / 180.0)  # convert to rad/s
            self.velocity_publishers[i].publish(twist_msg)

            wheel_msg = Float32MultiArray()
            wheel_msg.data = state[(12 + i * 11):(16 + i * 11)]
            self.wheel_velocity_publishers[i].publish(wheel_msg)

        for i in range(self.yellow_robot_count):
            msg = Pose2D()
            msg.x = state[5 + self.blue_robot_count * 11 + i * 11]
            msg.y = state[6 + self.blue_robot_count * 11 + i * 11]
            # msg.theta = state[7 + self.blue_robot_count * 11 + i * 11]
            theta_deg = state[7 + self.blue_robot_count * 11 + i * 11]
            msg.theta = self.to_rad(theta_deg)
            self.pose_publishers[self.blue_robot_count + i].publish(msg)

            twist_msg = Twist()
            twist_msg.linear.x = state[8 + self.blue_robot_count * 11 + i * 11]
            twist_msg.linear.y = state[9 + self.blue_robot_count * 11 + i * 11]
            twist_msg.angular.z = state[10 + self.blue_robot_count * 11 + i * 11] * (np.pi / 180.0)  # convert to rad/s
            self.velocity_publishers[self.blue_robot_count + i].publish(twist_msg)

            wheel_msg = Float32MultiArray()
            wheel_msg.data = state[(12 + self.blue_robot_count * 11 + i * 11):(16 + self.blue_robot_count * 11 + i * 11)]
            self.wheel_velocity_publishers[self.blue_robot_count + i].publish(wheel_msg)

def main(args=None):
    rclpy.init(args=args)

    simulator_node = SimulatorNode()

    rclpy.spin(simulator_node)

    simulator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
