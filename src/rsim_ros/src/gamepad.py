import robosim
from typing import List
import cv2
import numpy as np
import pygame
from collections import defaultdict

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Twist

class GamepadNode(Node):

    def __init__(self):
        super().__init__('gamepad')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('blue_robot_count', 3),
                ('yellow_robot_count', 3),
                ('frequency', 60)
            ]
        )
        self.blue_robot_count = self.get_parameter('blue_robot_count').get_parameter_value().integer_value
        self.yellow_robot_count = self.get_parameter('yellow_robot_count').get_parameter_value().integer_value
        frequency = self.get_parameter('frequency').get_parameter_value().integer_value

        pygame.init()
    
        self.get_logger().info('Gamepad Node Started')

        self.joystick_count = pygame.joystick.get_count()
        self.get_logger().info(f'Joystick count: {self.joystick_count}')
        
        self.pubs = defaultdict(dict)
        for i in range(self.blue_robot_count):
            # self.pubs['blue'][i] = self.create_publisher(Twist, f'/simulator/cmd/blue/robot{i}', 10)
            self.pubs['blue'][i] = self.create_publisher(Twist, f'/pid/cmd/velocity/blue/robot{i}', 10)
        for i in range(self.yellow_robot_count):
            # self.pubs['yellow'][i] = self.create_publisher(Twist, f'/simulator/cmd/yellow/robot{i}', 10)
            self.pubs['yellow'][i] = self.create_publisher(Twist, f'/pid/cmd/velocity/yellow/robot{i}', 10)
        self.team = 'blue'
        self.robot = 0
        self.prev_b4 = 0
        self.prev_b5 = 0
        self.deadzone = 0.1

        self.create_timer(1 / frequency, self.process_joystick_inputs)

    # print every axis and button
    def print_all_keys(self):
        for id in range(self.joystick_count):
            joystick = pygame.joystick.Joystick(id)
            joystick.init()
            for i in range(joystick.get_numaxes()):
                self.get_logger().info(f'J{joystick.get_id()} Axis {i}: {joystick.get_axis(i)}')
            for i in range(joystick.get_numbuttons()):
                self.get_logger().info(f'J{joystick.get_id()} Button {i}: {joystick.get_button(i)}')
            
    #L1: b4, R1: b5
    def process_joystick_inputs(self):
        # self.print_all_keys()
        for id in range(self.joystick_count):
            if id == 0:
                joystick = pygame.joystick.Joystick(id)
                joystick.init()

                w = joystick.get_axis(3)
                if abs(w) < self.deadzone:
                    w = 0.0
                right = joystick.get_axis(0)
                if abs(right) < self.deadzone:
                    right = 0.0
                forward = joystick.get_axis(1)
                if abs(forward) < self.deadzone:
                    forward = 0.0
                
                b4 = joystick.get_button(4)
                if b4 and not self.prev_b4:
                    self.team = 'blue' if self.team == 'yellow' else 'yellow'
                    count = self.blue_robot_count if self.team == 'blue' else self.yellow_robot_count
                    self.robot = min(self.robot, count - 1)
                    self.get_logger().info(f"Team: {self.team}, Robot: {self.robot}")
                self.prev_b4 = b4
                b5 = joystick.get_button(5)
                if b5 and not self.prev_b5:
                    count = self.blue_robot_count if self.team == 'blue' else self.yellow_robot_count
                    self.robot = (self.robot + 1) % count
                    self.get_logger().info(f"Robot: {self.robot}")
                self.prev_b5 = b5

                msg = Twist()
                msg.linear.x = forward * 3
                msg.linear.y = right * 3
                msg.angular.z = w * 5
                self.pubs[self.team][self.robot].publish(msg)
                

def main(args=None):
    rclpy.init(args=args)

    gamepad_node = GamepadNode()

    rclpy.spin(gamepad_node)

    gamepad_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
