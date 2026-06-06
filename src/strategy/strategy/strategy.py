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
from std_msgs.msg import String, Bool
from collections import defaultdict
from geometry_msgs.msg import Pose2D, Twist
import numpy as np
from .rl import Sim2Real
from .behaviors import (
    HaltBehavior,
    StopBehavior,
    RLBehavior,
    PrepareKickoffBehavior,
    DirectFreeBehavior,
    BallPlacementBehavior,
)


class Strategy(Node):

    def __init__(self):
        super().__init__('strategy')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('blue_robot_count', 3),
                ('yellow_robot_count', 3),
                ('frequency', 60),
                ('team', 'yellow'),  # which team this AI controls: 'blue' or 'yellow'
            ]
        )
        self.blue_robot_count = self.get_parameter('blue_robot_count').get_parameter_value().integer_value
        self.yellow_robot_count = self.get_parameter('yellow_robot_count').get_parameter_value().integer_value
        frequency = self.get_parameter('frequency').get_parameter_value().integer_value
        self.team = self.get_parameter('team').get_parameter_value().string_value

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
        # referee state
        self.ref_command = 'STOP'
        self.next_command = None
        self.blue_on_positive_half = None
        self.designated_position = None  # tuple (x,y) meters

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

        # Subscribe to referee connector topics
        self.create_subscription(String, '/referee/command', self.ref_command_cb, 10)
        self.create_subscription(String, '/referee/next_command', self.ref_next_command_cb, 10)
        self.create_subscription(String, '/referee/status_message', lambda msg: None, 10)  # optional
        self.create_subscription(Bool, '/referee/blue_on_positive_half', self.ref_half_cb, 10)
        self.create_subscription(Pose2D, '/referee/designated_position', self.ref_designated_cb, 10)


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

    def _active_behavior(self):
        cmd = self.ref_command or 'STOP'
        # Determine attack direction for formations: if our opponent goal is +x
        # Default: attack towards +x
        attack_dir = 1.0
        if self.blue_on_positive_half is not None:
            # blue goal on +x; so blue attacks -x, yellow attacks +x
            attack_dir = -1.0 if (self.team == 'blue' and self.blue_on_positive_half) else 1.0

        aux = {
            'attack_dir': attack_dir,
            'designated_position': self.designated_position,
        }

        # Map command -> behavior
        if cmd == 'HALT':
            return HaltBehavior(self.blue_robot_count, self.yellow_robot_count, self.team), aux
        if cmd == 'STOP':
            return StopBehavior(self.blue_robot_count, self.yellow_robot_count, self.team), aux
        if cmd in ('NORMAL_START', 'FORCE_START'):
            return RLBehavior(self.blue_robot_count, self.yellow_robot_count, self.team, self.rl_model), aux
        if cmd == 'PREPARE_KICKOFF_YELLOW':
            return PrepareKickoffBehavior(self.blue_robot_count, self.yellow_robot_count, self.team, 'yellow'), aux
        if cmd == 'PREPARE_KICKOFF_BLUE':
            return PrepareKickoffBehavior(self.blue_robot_count, self.yellow_robot_count, self.team, 'blue'), aux
        if cmd == 'DIRECT_FREE_YELLOW':
            return DirectFreeBehavior(self.blue_robot_count, self.yellow_robot_count, self.team, 'yellow'), aux
        if cmd == 'DIRECT_FREE_BLUE':
            return DirectFreeBehavior(self.blue_robot_count, self.yellow_robot_count, self.team, 'blue'), aux
        if cmd == 'BALL_PLACEMENT_YELLOW':
            return BallPlacementBehavior(self.blue_robot_count, self.yellow_robot_count, self.team, 'yellow'), aux
        if cmd == 'BALL_PLACEMENT_BLUE':
            return BallPlacementBehavior(self.blue_robot_count, self.yellow_robot_count, self.team, 'blue'), aux
        # Penalties/timeouts/others: default to STOP to be safe
        return StopBehavior(self.blue_robot_count, self.yellow_robot_count, self.team), aux

    def timer_callback(self):
        behavior, aux = self._active_behavior()
        actions = behavior.step(self.state, aux)
        # Publish actions
        for i in range(self.blue_robot_count):
            self.publish_action('blue', i, actions.get(f'blue_{i}', (0.0, 0.0, 0.0)))
        for i in range(self.yellow_robot_count):
            self.publish_action('yellow', i, actions.get(f'yellow_{i}', (0.0, 0.0, 0.0)))
        
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

    def publish_action(self, color: str, index: int, action):
        self.pubs[color][index]
        msg = Twist()

        msg.linear.x = float(action[0])
        msg.linear.y = float(action[1])
        msg.angular.z = np.deg2rad(float(action[2]))

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

    # Referee callbacks
    def ref_command_cb(self, msg: String):
        self.ref_command = msg.data
        self.get_logger().info(f'Ref command: {self.ref_command}')

    def ref_next_command_cb(self, msg: String):
        self.next_command = msg.data

    def ref_half_cb(self, msg: Bool):
        self.blue_on_positive_half = msg.data

    def ref_designated_cb(self, msg: Pose2D):
        self.designated_position = (msg.x, msg.y)


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
