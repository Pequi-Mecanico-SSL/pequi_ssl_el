from typing import List
import math
import numpy as np
# import os
# os.environ['SDL_AUDIODRIVER'] = 'dsp'
import pygame

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Twist

from .render_lib import SSLRenderField, SSLRobot, Ball, COLORS

class RenderNode(Node):

    def __init__(self):
        super().__init__('render')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('blue_robot_count', 2),
                ('yellow_robot_count', 2),
                ('framerate', 60)
            ]
        )
        self.blue_robot_count = self.get_parameter('blue_robot_count').get_parameter_value().integer_value
        self.yellow_robot_count = self.get_parameter('yellow_robot_count').get_parameter_value().integer_value
        self.framerate = self.get_parameter('framerate').get_parameter_value().integer_value

        pygame.init()
        pygame.display.init()
        pygame.display.set_caption('SSL')
        self.field_renderer = SSLRenderField()
        self.window = pygame.display.set_mode(self.field_renderer.window_size)

        self.selected_robot = None
        self.target = None
        self.yellow_robot_poses = {}
        self.blue_robot_poses = {}
        self.ball_pose = None
        for i in range(self.blue_robot_count):
            callback_blue = lambda msg, robot=i: self.pose_callback(msg, robot, "blue")
            self.create_subscription(Pose2D, f'/simulator/poses/blue/robot{i}', callback_blue, 10)
        for i in range(self.yellow_robot_count):
            callback_yellow = lambda msg, robot=i: self.pose_callback(msg, robot, "yellow")
            self.create_subscription(Pose2D, f'/simulator/poses/yellow/robot{i}', callback_yellow, 10)
        callback_ball = lambda msg: self.pose_callback(msg, 0, "ball")
        self.create_subscription(Pose2D, f'/simulator/poses/ball', callback_ball, 10)

        self.cmd_pubs = {"blue": [], "yellow": []}
        for i in ["blue", "yellow"]:
            count = self.blue_robot_count if i == "blue" else self.yellow_robot_count
            for j in range(count):
                self.cmd_pubs[i].append(self.create_publisher(Pose2D, f'/simulator/cmd/pose/{i}/robot{j}', 10))
                
            
        
        self.clock = pygame.time.Clock()
        self.create_timer(1/self.framerate, self.render_timer)
        
        self.get_logger().info('Started render node')
    
    def pose_callback(self, msg: Pose2D, robot: int, object_type: str):
        # self.get_logger().info(f'Pose received: {msg.x}, {msg.y}, {msg.theta}')
        if object_type == "yellow":
            self.yellow_robot_poses[robot] = msg
        elif object_type == "blue":
            self.blue_robot_poses[robot] = msg
        elif object_type == "ball":
            self.ball_pose = msg
    
    def meters_to_pixels(self, x, y):
        x_scaled = x * self.field_renderer.scale + self.field_renderer.center_x
        y_scaled = y * self.field_renderer.scale + self.field_renderer.center_y
        return x_scaled, y_scaled
    
    def pixels_to_meters(self, x, y):
        x_scaled = (x - self.field_renderer.center_x) / self.field_renderer.scale
        y_scaled = (y - self.field_renderer.center_y) / self.field_renderer.scale
        return x_scaled, y_scaled
    
    def radians_to_degrees(self, radians):
        return math.degrees(radians)
    
    def render_robot(self, i, color):
        if color == COLORS["YELLOW"]:
            pose = self.yellow_robot_poses.get(i)
        else:
            pose = self.blue_robot_poses.get(i)
        if pose is None:
            return
        x, y = self.meters_to_pixels(pose.x, pose.y)
        render_robot = SSLRobot(x, y, self.radians_to_degrees(pose.theta),
                                self.field_renderer.scale, i, color)
        render_robot.draw(self.window)
        if self.selected_robot == (color, i):
            pygame.draw.circle(self.window, color, (int(x), int(y)), 15, 2)
    
    def render_ball(self):
        if self.ball_pose is None:
            return
        x, y = self.meters_to_pixels(self.ball_pose.x, self.ball_pose.y)
        ball = Ball(
            x, y,
            self.field_renderer.scale
        )
        ball.draw(self.window)
    
    def clicked_on_robot(self, click_pos):
        click_x, click_y = click_pos
        robot_size = 0.09 * self.field_renderer.scale
        for color in [COLORS["BLUE"], COLORS["YELLOW"]]:
            count = self.blue_robot_count if color == COLORS["BLUE"] else self.yellow_robot_count
            for i in range(count):
                pose = self.blue_robot_poses.get(i) if color == COLORS["BLUE"]\
                    else self.yellow_robot_poses.get(i)
                if pose is None:
                    continue
                x, y = self.meters_to_pixels(pose.x, pose.y)
                if math.sqrt((x - click_x)**2 + (y - click_y)**2) < robot_size:
                    return i, color
        return None, None
    
    def handle_clicks(self):
        event_list = pygame.event.get()
        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN:
                robot, color = self.clicked_on_robot(event.pos)
                if robot is not None:
                    self.target = None
                    selected = (color, robot)
                    if selected == self.selected_robot:
                        self.selected_robot = None
                    else:
                        self.selected_robot = selected
                else:
                    if self.selected_robot is not None:
                        self.target = event.pos
    
    def publish_target(self):
        if self.target is not None:
            x, y = self.pixels_to_meters(*self.target)
            color, robot = self.selected_robot
            color_str = "blue" if color == COLORS["BLUE"] else "yellow"
            msg = Pose2D()
            msg.x = x
            msg.y = y
            msg.theta = 0.0
            self.cmd_pubs[color_str][robot].publish(msg)

    def render_timer(self):
        self.field_renderer.draw(self.window)
        for i in range(self.blue_robot_count):
            self.render_robot(i, COLORS["BLUE"])
        for i in range(self.yellow_robot_count):
            self.render_robot(i, COLORS["YELLOW"])
        self.render_ball()
        if self.target is not None:
            pygame.draw.circle(self.window, COLORS["RED"], self.target, 5, 2)

        pygame.event.pump()
        pygame.display.update()

        self.handle_clicks()
        self.publish_target()


def main(args=None):
    rclpy.init(args=args)

    render_node = RenderNode()

    rclpy.spin(render_node)

    render_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
