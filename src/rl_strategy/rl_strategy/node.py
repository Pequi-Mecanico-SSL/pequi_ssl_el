"""ROS node for the RL strategy: vision poses in, robot velocity commands out.

Thin by design: topics in, topics out, staleness safety. All policy logic
lives behind ``adapter.PolicyAdapter``; this file must never import
``rl_strategy.vendored``.

Subscribes (real SI, right-handed field frame, the ssl_vision_connector
contract):
    /vision/poses/{blue,yellow}/robot<i>   geometry_msgs/Pose2D  (m, rad)
    /vision/poses/ball                     geometry_msgs/Pose2D  (m)

Publishes (body frame, REP-103: x forward, y left, z CCW), one topic per
robot under the per-robot namespace convention:
    /<team>/robot<i>/cmd_vel               geometry_msgs/Twist   (m/s, rad/s)

When robot_control is launched under the namespace /<color>/robot<i>
(e.g. ROS_NAMESPACE=/yellow/robot0), its cmd_vel subscription lands on this
topic without remapping; otherwise remap cmd_vel accordingly.

Safety follows the reference deploy script: while the team has no fresh
vision, publish zero Twists and do not advance the observation stack.
"""
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Twist
from std_msgs.msg import Empty
from ament_index_python.packages import get_package_share_directory

from .adapter import PolicyAdapter


class RLStrategyNode(Node):

    def __init__(self):
        super().__init__('rl_strategy')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('team', 'blue'),
                ('blue_robot_count', 3),
                ('yellow_robot_count', 3),
                ('field_length_m', 3.3),
                ('field_width_m', 2.0),
                ('fps', 30),
                ('action_mode', 'mean'),
                ('action_seed', 0),
                ('pose_timeout', 0.25),
                ('weights_path', ''),
                ('episode_reset', True),
                # 1.0 = feed the policy real meters (default);
                # 0.0 = derive 9.0/field_length (expected to degrade
                # ball-contact behaviors; experiments only).
                ('scale', 1.0),
                # "parked" | "kickoff": poses assumed for robots vision does
                # not see. kickoff = training spawn marks (solo mode).
                ('placeholder_mode', 'parked'),
                # Solo mode (one real robot plays team slot 0, the other five
                # are placeholders): set solo_cmd_topic to activate. Slot-0
                # pose comes from solo_pose_topic regardless of vision color:
                # the physical marker color is a vision detail, independent
                # of the policy team.
                ('solo_cmd_topic', ''),
                ('solo_pose_topic', '/vision/poses/yellow/robot0'),
                # Real-hardware caps applied at the very edge (0.0 = uncapped).
                ('cmd_speed_cap', 0.0),      # m/s, linear norm
                ('cmd_angular_cap', 0.0),    # rad/s
            ]
        )
        gp = lambda n: self.get_parameter(n).get_parameter_value()
        self.team = gp('team').string_value
        n_blue = gp('blue_robot_count').integer_value
        n_yellow = gp('yellow_robot_count').integer_value
        fps = gp('fps').integer_value
        self.pose_timeout = gp('pose_timeout').double_value

        weights_path = gp('weights_path').string_value
        if not weights_path:
            weights_path = (
                get_package_share_directory('rl_strategy')
                + '/weights/policy_state.portable.npz'
            )

        self.adapter = PolicyAdapter(
            weights_path=weights_path,
            team=self.team,
            field_length_m=gp('field_length_m').double_value,
            field_width_m=gp('field_width_m').double_value,
            n_robots_blue=n_blue,
            n_robots_yellow=n_yellow,
            action_mode=gp('action_mode').string_value,
            action_seed=gp('action_seed').integer_value,
            episode_reset=gp('episode_reset').bool_value,
            scale=(gp('scale').double_value or None),
            placeholder_mode=gp('placeholder_mode').string_value,
        )
        self.solo_cmd_topic = gp('solo_cmd_topic').string_value
        self.solo = bool(self.solo_cmd_topic)
        self.speed_cap = gp('cmd_speed_cap').double_value
        self.angular_cap = gp('cmd_angular_cap').double_value

        self.n_robots = {'blue': n_blue, 'yellow': n_yellow}
        # latest_pose[(color, i)] = (x, y, theta); stamp[(color, i)] = monotonic
        self.latest_pose = {}
        self.stamp = {}
        self.ball = None
        self.ball_stamp = 0.0

        if self.solo:
            # One real robot -> team slot 0; every other slot is a placeholder.
            self.create_subscription(
                Pose2D, gp('solo_pose_topic').string_value,
                lambda msg: self._pose_cb(msg, self.team, 0), 10)
            self.cmd_pubs = {
                0: self.create_publisher(Twist, self.solo_cmd_topic, 10)}
        else:
            for color in ('blue', 'yellow'):
                for i in range(self.n_robots[color]):
                    self.create_subscription(
                        Pose2D, f'/vision/poses/{color}/robot{i}',
                        lambda msg, c=color, i=i: self._pose_cb(msg, c, i), 10)
            self.cmd_pubs = {
                i: self.create_publisher(
                    Twist, f'/{self.team}/robot{i}/cmd_vel', 10)
                for i in range(self.n_robots[self.team])
            }
        self.create_subscription(Pose2D, '/vision/poses/ball', self._ball_cb, 10)

        # Manual episode reset (the kickoff-formation reset assumes a simulator
        # teleport; on a real field trigger it by hand):
        #   ros2 topic pub --once /rl_strategy/reset std_msgs/msg/Empty
        self.create_subscription(
            Empty, '/rl_strategy/reset', self._manual_reset, 10)

        self._was_stale = True
        self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(
            f'RL strategy up: team={self.team} '
            f'scale={self.adapter.scale:.3f} weights={weights_path}'
        )

    def _manual_reset(self, _msg):
        self.adapter.reset()
        self.get_logger().info('manual episode reset (/rl_strategy/reset)')

    def _pose_cb(self, msg, color, i):
        self.latest_pose[(color, i)] = (msg.x, msg.y, msg.theta)
        self.stamp[(color, i)] = time.monotonic()

    def _ball_cb(self, msg):
        self.ball = (msg.x, msg.y)
        self.ball_stamp = time.monotonic()

    def _fresh_world(self):
        """Real-SI world dict of only fresh entities."""
        now = time.monotonic()
        world = {'blue': {}, 'yellow': {}, 'ball': None}
        for (color, i), pose in self.latest_pose.items():
            if now - self.stamp[(color, i)] <= self.pose_timeout:
                world[color][i] = pose
        if self.ball is not None and now - self.ball_stamp <= self.pose_timeout:
            world['ball'] = self.ball
        return world

    def _publish_zeros(self):
        for pub in self.cmd_pubs.values():
            pub.publish(Twist())

    def _tick(self):
        world = self._fresh_world()

        # Safety, as in the reference deploy script: no fresh team robot ->
        # zeros, stack frozen.
        if not world[self.team]:
            if not self._was_stale:
                self.get_logger().warn('team vision stale; sending zero commands')
            self._was_stale = True
            self._publish_zeros()
            return
        if self._was_stale:
            self.get_logger().info('team vision fresh; strategy active')
            self._was_stale = False

        try:
            commands = self.adapter.step(world)
            if self.adapter.did_kickoff_reset:
                self.get_logger().info('kickoff formation detected; episode reset')
        except Exception as e:
            self.get_logger().error(f'strategy step failed: {e}; sending zeros')
            self._publish_zeros()
            return

        for i, pub in self.cmd_pubs.items():
            twist = Twist()
            if i in world[self.team] and i in commands:
                v_fwd, v_left, v_ang, _kick = commands[i]  # kick: no hardware yet
                if self.speed_cap > 0.0:
                    norm = (v_fwd * v_fwd + v_left * v_left) ** 0.5
                    if norm > self.speed_cap:
                        k = self.speed_cap / norm
                        v_fwd, v_left = v_fwd * k, v_left * k
                if self.angular_cap > 0.0:
                    v_ang = max(-self.angular_cap, min(self.angular_cap, v_ang))
                twist.linear.x = float(v_fwd)
                twist.linear.y = float(v_left)
                twist.angular.z = float(v_ang)
            # unseen team robots get an explicit zero Twist (safety)
            pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = RLStrategyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
