"""Behavior router and command mux, one instance per team.

Decides which behavior drives the robots from the game state (table in sslrules.routing) and relays
only that behavior's commands:

    /referee/game_state                              in   pequi_msgs/GameState
    /<team>/behaviors/<name>/robot<i>/cmd_vel        in   geometry_msgs/Twist   (every behavior publishes here)
    /<team>/router/active                            out  pequi_msgs/ActiveBehavior (latched; on change + 5 Hz)
    /<team>/robot<i>/cmd_vel                         out  geometry_msgs/Twist   (the active behavior's, relayed)

Safety: on every switch one zero command goes out to every robot before the new behavior is relayed; a
robot whose active behavior has not published for `behavior_timeout` seconds gets zeros; `halt` (and no
referee at all, by default) means zeros. Adding a behavior is a row in sslrules.routing plus a node that
publishes under its name.

With only the packages in this repository, the router relays rl_strategy (behavior `rl`) in RUNNING and sends
zero commands in the other referee states: the positioning behaviors are not part of this repository yet.

Per-robot override: when the referee allows fewer robots than we field (yellow/red card, /referee/state
max_allowed_bots), the highest-numbered robots are relayed from the `bench` behavior instead, which parks
them in the substitution zone behind our goal [EL 5.5.1, 9].
"""
import time

import rclpy
from geometry_msgs.msg import Twist
from pequi_msgs.msg import ActiveBehavior, GameState as GameStateMsg, RefereeState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from sslrules import GameState, Team
from sslrules.routing import BENCH, DEFAULT_TABLE, HALT, behavior_for

TEAM_OF = {'blue': Team.BLUE, 'yellow': Team.YELLOW}


class Router(Node):

    def __init__(self):
        super().__init__('router')
        self.declare_parameters(namespace='', parameters=[
            ('team', 'blue'),
            ('robot_count', 3),
            ('frequency', 30),            # safety / keepalive tick
            ('behavior_timeout', 0.5),    # s without a command from the active behavior -> zeros
            ('referee_timeout', 1.0),     # s without /referee/game_state -> `no_referee_behavior`
            ('no_referee_behavior', 'halt'),   # 'halt' at a match; 'rl' to play without a game controller
        ])
        gp = lambda n: self.get_parameter(n).get_parameter_value()
        self.team = gp('team').string_value
        self.our_team = TEAM_OF[self.team]
        self.n = gp('robot_count').integer_value
        self.behavior_timeout = gp('behavior_timeout').double_value
        self.referee_timeout = gp('referee_timeout').double_value
        self.no_referee_behavior = gp('no_referee_behavior').string_value
        frequency = gp('frequency').integer_value

        self.behaviors = sorted((set(DEFAULT_TABLE.values()) | {BENCH}) - {HALT})
        self.benched = set()      # robot indices relayed from `bench` (card in effect)
        self.latest = {}          # (behavior, i) -> (Twist, monotonic stamp)
        self.active = HALT
        self.activation_id = 0
        self.game_state = None
        self.game_state_stamp = 0.0
        self._last_announce = 0.0

        self.cmd_pubs = [self.create_publisher(Twist, f'/{self.team}/robot{i}/cmd_vel', 10) for i in range(self.n)]
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.active_pub = self.create_publisher(ActiveBehavior, f'/{self.team}/router/active', latched)
        self.create_subscription(GameStateMsg, '/referee/game_state', self._game_state_cb, 10)
        self.create_subscription(RefereeState, '/referee/state', self._referee_cb, 10)
        for b in self.behaviors:
            for i in range(self.n):
                self.create_subscription(
                    Twist, f'/{self.team}/behaviors/{b}/robot{i}/cmd_vel',
                    lambda msg, b=b, i=i: self._cmd_cb(msg, b, i), 10)
        self.create_timer(1.0 / frequency, self._tick)
        self._announce(force=True)
        self.get_logger().info(f'router up: team={self.team} behaviors={self.behaviors} default={self.active}')

    # ---- inputs --------------------------------------------------------------

    def _game_state_cb(self, msg):
        self.game_state = msg
        self.game_state_stamp = time.monotonic()
        name, params = behavior_for(GameState(msg.state), Team(msg.for_team), self.our_team)
        self._switch(name, params, msg)

    def _referee_cb(self, msg):
        info = msg.blue if self.team == 'blue' else msg.yellow
        allowed = int(info.max_allowed_bots)
        benched = set(range(allowed, self.n)) if 0 <= allowed < self.n else set()   # 0 = no robot allowed (cards)
        if benched != self.benched:
            self.get_logger().info(f'{allowed} robots allowed -> benching {sorted(benched) or "none"}')
            for i in benched ^ self.benched:
                self.cmd_pubs[i].publish(Twist())
            self.benched = benched
            self._announce(force=True)

    def _source(self, i):
        """Behavior whose commands robot i receives."""
        return BENCH if i in self.benched else self.active

    def _cmd_cb(self, msg, behavior, i):
        self.latest[(behavior, i)] = (msg, time.monotonic())
        if behavior == self._source(i) and self.active != HALT:
            self.cmd_pubs[i].publish(msg)     # relay immediately: no added latency

    # ---- switching -------------------------------------------------------------

    def _switch(self, name, params, msg):
        if name == self.active:
            self._params = params
            return
        self.activation_id += 1
        for pub in self.cmd_pubs:
            pub.publish(Twist())              # one clean zero between behaviors
        prev, self.active, self._params = self.active, name, params
        self._announce(force=True, msg=msg)
        self.get_logger().info(f'{prev} -> {name} (activation {self.activation_id}, ours={params.get("ours")})')

    def _announce(self, force=False, msg=None):
        now = time.monotonic()
        if not force and now - self._last_announce < 0.2:
            return
        self._last_announce = now
        a = ActiveBehavior()
        a.header.stamp = self.get_clock().now().to_msg()
        a.behavior = self.active
        a.activation_id = self.activation_id
        params = getattr(self, '_params', {})
        a.ours = bool(params.get('ours', False))
        a.for_team = int(params.get('for_team', 0))
        a.game_state = int(msg.state) if msg is not None else (int(self.game_state.state) if self.game_state else 0)
        a.benched = sorted(self.benched)
        self.active_pub.publish(a)

    # ---- safety tick -------------------------------------------------------------

    def _tick(self):
        now = time.monotonic()
        if self.game_state is not None and now - self.game_state_stamp > self.referee_timeout:
            if self.active != self.no_referee_behavior:
                self.get_logger().warn(f'no /referee/game_state for {self.referee_timeout:.1f} s -> {self.no_referee_behavior}')
                self._switch(self.no_referee_behavior, {'ours': False, 'for_team': 0}, None)
        self._announce()
        for i, pub in enumerate(self.cmd_pubs):
            if self.active == HALT:
                pub.publish(Twist())
                continue
            latest = self.latest.get((self._source(i), i))
            if latest is None or now - latest[1] > self.behavior_timeout:
                pub.publish(Twist())          # source behavior silent for this robot: hold still
                self.get_logger().warn(f'robot{i}: no command from {self._source(i)} for {"ever" if latest is None else f"{now - latest[1]:.2f} s"} -> zeros '
                                       f'(active={self.active}, benched={sorted(self.benched)}, known={sorted(set(k[0] for k in self.latest))})', throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = Router()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
