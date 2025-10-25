# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from collections import defaultdict
from geometry_msgs.msg import Pose2D, Twist
import numpy as np
from .rl import Sim2Real


class Strategy(Node):

    def __init__(self):
        super().__init__('strategy')
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

        self.get_logger().info('Strategy Node Started')

        self.state = {
            **{f'blue_{i}': [0.0, 0.0, 0.0] for i in range(self.blue_robot_count)},
            **{f'yellow_{i}': [0.0, 0.0, 0.0] for i in range(self.yellow_robot_count)},
            'ball': [0.0, 0.0]
        }

        self.pubs = defaultdict(dict)
        for i in range(self.blue_robot_count):
            self.pubs['blue'][i] = self.create_publisher(Twist, f'/simulator/cmd/blue/robot{i}', 10)
        for i in range(self.yellow_robot_count):
            self.pubs['yellow'][i] = self.create_publisher(Twist, f'/simulator/cmd/yellow/robot{i}', 10)
        self.team = 'blue'

        # Subscribe to robot and ball poses to keep self.state updated
        for i in range(self.blue_robot_count):
            self.create_subscription(
                Pose2D,
                f'/simulator/poses/blue/robot{i}',
                lambda msg, i=i: self.pose_callback(msg, 'blue', i),
                10,
            )
        for i in range(self.yellow_robot_count):
            self.create_subscription(
                Pose2D,
                f'/simulator/poses/yellow/robot{i}',
                lambda msg, i=i: self.pose_callback(msg, 'yellow', i),
                10,
            )
        self.create_subscription(Pose2D, '/simulator/poses/ball', self.ball_callback, 10)


        #/simulator/poses/blue/robot0
        #/simulator/poses/blue/robot1
        #/simulator/poses/blue/robot2
        #/simulator/poses/yellow/robot0
        #/simulator/poses/yellow/robot1
        #/simulator/poses/yellow/robot2

        self.rl_model = Sim2Real(
            field_length=9.0,
            field_width=6.0,
            max_ep_length=30*40
        )

        self.timer = self.create_timer(1 / frequency, self.timer_callback)

    def timer_callback(self):
        actions = self.rl_model.state_to_action(self.state, convert=True)
        for i in range(self.blue_robot_count):
            pass
            self.publish_action('blue', i, actions[f'blue_{i}'])
        for i in range(self.yellow_robot_count):
            self.publish_action('yellow', i, actions[f'yellow_{i}'])
        
        #msg = Twist()
        ## Set Twist fields directly; Twist has no 'data' field
        #msg.linear.x = 1.0
        #msg.linear.y = 0.0
        #msg.linear.z = 0.0
        #msg.angular.x = 0.0
        #msg.angular.y = 0.0
        #msg.angular.z = 0.5

        #self.pubs['blue'][1].publish(msg)
        ## Log a single formatted string
        #self.get_logger().info(
        #    f'Publishing Twist: linear=({msg.linear.x}, {msg.linear.y}, {msg.linear.z}), '
        #    f'angular=({msg.angular.x}, {msg.angular.y}, {msg.angular.z})'
        #)

    def publish_action(self, color: str, index: int, action: list):
        self.pubs[color][index]
        msg = Twist()

        msg.linear.x = action[0]
        msg.linear.y = action[1]
        #msg.angular.z = action[2]
        msg.angular.z = np.deg2rad(action[2])

        self.pubs[color][index].publish(msg)
        self.get_logger().info(
            f'Publishing Twist {color}_{index}: linear=({msg.linear.x}, {msg.linear.y}, {msg.linear.z}), '
            f'angular=({msg.angular.x}, {msg.angular.y}, {msg.angular.z})'
        )

    def pose_callback(self, msg: Pose2D, color: str, index: int):
        key = f'{color}_{index}'
        #self.state[key] = [msg.x, msg.y, msg.theta]
        self.state[key] = [msg.x, msg.y, np.rad2deg(msg.theta)]

    def ball_callback(self, msg: Pose2D):
        self.state['ball'] = [msg.x, msg.y]


def main(args=None):
    rclpy.init(args=args)

    strategy = Strategy()

    rclpy.spin(strategy)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    strategy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
