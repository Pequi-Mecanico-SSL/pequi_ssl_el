"""SSL game-controller multicast -> ROS 2.

    /referee/state        pequi_msgs/RefereeState   the wire message, decoded, SI units
    /referee/game_state   pequi_msgs/GameState      derived state incl. ball-in-play (needs /vision/poses/ball)

The wire (224.5.23.1:10003, ~40 Hz) carries the command and its counter, the next command, the designated
ball position and the game events. It does not say when the ball is in play after a kick-off or free kick;
that is derived here with sslrules.InPlayTracker (ball moved 0.05 m, or the kick timer ran out), the same
way the reference autoref does it.

Units: the wire mixes mm (designated_position), m (game-event locations) and us (times); everything published
here is metres and seconds.
"""
import json
import socket

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from google.protobuf.json_format import MessageToDict
from pequi_msgs.msg import GameEvent, GameState, RefereeState, TeamInfo
from rclpy.node import Node

import sslrules
from sslrules import Command, InPlayTracker, Stage

from .proto import ssl_gc_game_event_pb2 as gepb
from .proto import ssl_gc_referee_message_pb2 as refpb

RULESETS = {'EL': sslrules.EL, 'EL_REFEREED': sslrules.EL_REFEREED, 'DIVB': sslrules.DIVB}


class RefereeConnector(Node):

    def __init__(self):
        super().__init__('referee_connector')
        self.declare_parameters(namespace='', parameters=[
            ('referee_ip', '224.5.23.1'),
            ('referee_port', 10003),
            ('referee_iface', '0.0.0.0'),   # interface address to join on; 0.0.0.0 = default route
            ('frequency', 60),
            ('ruleset', 'EL'),
            ('ball_topic', '/vision/poses/ball'),
            ('ball_timeout', 0.5),       # s without a ball pose -> in-play tracking pauses
            ('stale_warn', 2.0),         # s without referee packets -> warn
        ])
        gp = lambda n: self.get_parameter(n).get_parameter_value()
        ip = gp('referee_ip').string_value
        port = gp('referee_port').integer_value
        iface = gp('referee_iface').string_value
        frequency = gp('frequency').integer_value
        self.ball_timeout = gp('ball_timeout').double_value
        self.stale_warn = gp('stale_warn').double_value
        self.rules = RULESETS[gp('ruleset').string_value]

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        mreq = socket.inet_aton(ip) + socket.inet_aton(iface)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.sock.setblocking(False)

        self.state_pub = self.create_publisher(RefereeState, '/referee/state', 10)
        self.game_state_pub = self.create_publisher(GameState, '/referee/game_state', 10)
        self.create_subscription(Pose2D, gp('ball_topic').string_value, self._ball_cb, 10)

        self.tracker = InPlayTracker(self.rules)
        self.ball = None
        self.ball_stamp = 0.0
        self.last_packet_time = None
        self.last_packet = None
        self._warned_stale = False
        self._seen_counter = None

        self.create_timer(1.0 / frequency, self._tick)
        self.get_logger().info(f'listening for referee on {ip}:{port}'
                               + (f' on interface {iface}' if iface != '0.0.0.0' else '') + f', ruleset {self.rules.name}')

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _ball_cb(self, msg):
        self.ball = np.array([msg.x, msg.y])
        self.ball_stamp = self._now()

    def _drain(self):
        """Newest packet in the socket queue (the GC sends one copy per interface; all identical)."""
        newest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            newest = data
        return newest

    def _tick(self):
        now = self._now()
        data = self._drain()
        if data is not None:
            msg = refpb.Referee()
            msg.ParseFromString(data)
            self.last_packet = msg
            self.last_packet_time = now
            if self._warned_stale:
                self.get_logger().info('referee packets back')
                self._warned_stale = False
        if self.last_packet is None:
            return
        if now - self.last_packet_time > self.stale_warn and not self._warned_stale:
            self.get_logger().warn(f'no referee packets for {self.stale_warn:.0f} s')
            self._warned_stale = True
        if now - self.last_packet_time > 1.0:
            return      # a silent referee: publish nothing, so that everyone downstream halts (a HALT may have been missed)

        m = self.last_packet
        ref = self._to_referee_state(m, now)
        self.state_pub.publish(ref)

        ball = self.ball if (self.ball is not None and now - self.ball_stamp <= self.ball_timeout) else None
        placement = None
        if m.HasField('designated_position'):
            placement = np.array([m.designated_position.x, m.designated_position.y]) / 1000.0
        sit = self.tracker.update(
            command=Command(m.command), command_counter=m.command_counter, stage=Stage(m.stage),
            next_command=Command(m.next_command) if m.HasField('next_command') else None,
            action_time_remaining=m.current_action_time_remaining / 1e6 if m.HasField('current_action_time_remaining') else None,
            placement_position=placement, ball_xy=ball, t=now)
        if m.command_counter != self._seen_counter:
            self._seen_counter = m.command_counter
            self.get_logger().info(
                f'command #{m.command_counter} {refpb.Referee.Command.Name(m.command)} -> '
                f'{sit.state.name}/{sit.for_team.name}')

        gs = GameState()
        gs.header.stamp = self.get_clock().now().to_msg()
        gs.state = int(sit.state)
        gs.state_name = sit.state.name
        gs.for_team = int(sit.for_team)
        gs.for_team_name = sit.for_team.name
        gs.has_next_state = sit.next_state is not None
        gs.next_state = int(sit.next_state) if sit.next_state is not None else 0
        gs.next_for_team = int(sit.next_for_team)
        gs.ball_in_play = sit.ball_in_play_since is not None
        gs.ball_in_play_since = float(sit.ball_in_play_since or 0.0)
        gs.command = int(sit.command)
        gs.command_counter = sit.command_counter
        gs.stage = int(sit.stage)
        gs.has_placement_position = sit.placement_position is not None
        if sit.placement_position is not None:
            gs.placement_x, gs.placement_y = (float(v) for v in sit.placement_position)
        gs.blue_team_on_positive_half = bool(m.blue_team_on_positive_half)
        self.game_state_pub.publish(gs)

    # ---- wire -> message -------------------------------------------------

    @staticmethod
    def _team_info(t):
        ti = TeamInfo()
        ti.name = t.name
        ti.score = t.score
        ti.red_cards = t.red_cards
        ti.yellow_cards = t.yellow_cards
        ti.yellow_card_times = [x / 1e6 for x in t.yellow_card_times]
        ti.timeouts = t.timeouts
        ti.timeout_time = t.timeout_time / 1e6
        ti.goalkeeper = t.goalkeeper
        ti.foul_counter = t.foul_counter
        ti.ball_placement_failures = t.ball_placement_failures
        ti.can_place_ball = t.can_place_ball
        ti.max_allowed_bots = t.max_allowed_bots
        ti.bot_substitution_intent = t.bot_substitution_intent
        ti.ball_placement_failures_reached = t.ball_placement_failures_reached
        return ti

    @staticmethod
    def _game_event(ev):
        ge = GameEvent()
        ge.type = gepb.GameEvent.Type.Name(ev.type)
        ge.origin = list(ev.origin)
        d = MessageToDict(ev, preserving_proto_field_name=True)
        which = ev.WhichOneof('event')
        payload = d.get(which, {}) if which else {}
        by_team = payload.get('by_team') or payload.get('by_bot', {}).get('team') or 'UNKNOWN'
        ge.by_team = by_team
        ge.details = json.dumps(payload, separators=(',', ':'))
        return ge

    def _to_referee_state(self, m, now):
        r = RefereeState()
        r.header.stamp = self.get_clock().now().to_msg()
        r.packet_timestamp = m.packet_timestamp
        r.source_identifier = m.source_identifier
        r.match_type = refpb.MatchType.Name(m.match_type)
        r.stage = m.stage
        r.stage_name = refpb.Referee.Stage.Name(m.stage)
        r.stage_time_left = m.stage_time_left / 1e6 if m.HasField('stage_time_left') else 0.0
        r.command = m.command
        r.command_name = refpb.Referee.Command.Name(m.command)
        r.command_counter = m.command_counter
        r.command_timestamp = m.command_timestamp
        r.has_next_command = m.HasField('next_command')
        if r.has_next_command:
            r.next_command = m.next_command
            r.next_command_name = refpb.Referee.Command.Name(m.next_command)
        r.has_designated_position = m.HasField('designated_position')
        if r.has_designated_position:
            r.designated_x = m.designated_position.x / 1000.0
            r.designated_y = m.designated_position.y / 1000.0
        r.blue_team_on_positive_half = m.blue_team_on_positive_half
        r.current_action_time_remaining = m.current_action_time_remaining / 1e6 if m.HasField('current_action_time_remaining') else 0.0
        r.yellow = self._team_info(m.yellow)
        r.blue = self._team_info(m.blue)
        r.game_events = [self._game_event(ev) for ev in m.game_events]
        return r


def main(args=None):
    rclpy.init(args=args)
    node = RefereeConnector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
