from typing import List
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Twist

class ControlNode(Node):

    def __init__(self):
        super().__init__('controller')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('color', 'blue'),
                ('robot_id', 0),
                ('frequency', 10)
            ]
        )
        color = self.get_parameter('color').get_parameter_value().string_value
        robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        frequency = self.get_parameter('frequency').get_parameter_value().integer_value
        self.get_logger().info(f'Starting control node for {color} robot {robot_id}')
        
        # in radians: 45, 135, 225, 315
        self.wheel_orientation = [math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4]
        self.radius = 0.09

        self.current_pose = None
        self.target_pose = None
        self.latest_cmd = self.get_clock().now()

        # subscribe to pose2d in /simulator/poses/blue/robot0
        self.create_subscription(Pose2D, f'/simulator/poses/{color}/robot{robot_id}', self.pose_callback, 10)
        self.create_subscription(Pose2D, f'/simulator/cmd/pose/{color}/robot{robot_id}', self.target_pose_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, f'/simulator/cmd/{color}/robot{robot_id}', 10)

        self.create_timer(1 / frequency, self.control_loop)

        # D = self.velocity_coupling_matrix()
        # self.get_logger().info(f'D: {D}')
        # wheel_speeds = self.calculate_wheel_speeds(0.5, 0.0, 0.0)
        # self.get_logger().info(f'Wheel speeds: {wheel_speeds}')

    def control_loop(self):
        if self.current_pose is None or self.target_pose is None:
            return
        
        # now = self.get_clock().now()
        # if (now - self.latest_cmd).nanoseconds / 1e9 > 1:
        #     return

        # calculate the error
        error = Pose2D()
        error.x = self.target_pose.x - self.current_pose.x
        error.y = self.target_pose.y - self.current_pose.y
        error.theta = self.target_pose.theta - self.current_pose.theta

        target_distance = math.sqrt(error.x**2 + error.y**2)
        target_velocity = 1 * np.tanh(target_distance * 20)

        # calculate the control signal
        v_x = error.x * target_velocity
        v_y = error.y * target_velocity
        v_angle = error.theta * 4.0
    
        # convert to robot frame
        v_x_robot = math.cos(self.current_pose.theta) * v_x + math.sin(self.current_pose.theta) * v_y
        v_y_robot = -math.sin(self.current_pose.theta) * v_x + math.cos(self.current_pose.theta) * v_y
        v_angle_robot = v_angle

        twist = Twist()
        twist.linear.x = v_x_robot
        twist.linear.y = v_y_robot
        twist.angular.z = v_angle_robot
        self.cmd_vel_pub.publish(twist)
    
    def pose_callback(self, msg: Pose2D):
        self.current_pose = msg

    def target_pose_callback(self, msg: Pose2D):
        self.target_pose = msg
        self.latest_cmd = self.get_clock().now()
    
    def velocity_coupling_matrix(self):
        lines = []
        for i in range(len(self.wheel_orientation)):
            lines.append([
                -math.sin(self.wheel_orientation[i]), math.cos(self.wheel_orientation[i]), 1
            ])
        return np.array(lines)
    
    def calculate_wheel_speeds(self, v_x: float, v_y: float, v_angle: float):
        v = np.array([v_x, v_y, self.radius * v_angle])
        # velocity_coupling_matrix dot v
        return self.velocity_coupling_matrix().dot(v)


def main(args=None):
    rclpy.init(args=args)

    control_node = ControlNode()

    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
