import socket
import json
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose2D


class SSLRefereeProtobufToROS(Node):
    def __init__(self):
        super().__init__('referee_protobuf_to_ros')

        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('group_ip', '224.5.23.1'),
                ('port', 10003),
                ('interface_ip', '0.0.0.0'),
                ('frequency', 60),
            ]
        )

        self.group_ip = self.get_parameter('group_ip').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().integer_value
        self.interface_ip = self.get_parameter('interface_ip').get_parameter_value().string_value

        frequency = self.get_parameter('frequency').get_parameter_value().integer_value
        time_step_ms = max(1, 1000 // max(1, frequency))

        # Socket setup (UDP multicast)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.sock.bind(('', self.port))
        mreq = socket.inet_aton(self.group_ip) + socket.inet_aton(self.interface_ip)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        # Non-blocking to keep timer responsive
        self.sock.settimeout(0.0)

        # Publishers
        self.stage_pub = self.create_publisher(String, '/referee/stage', 10)
        self.command_pub = self.create_publisher(String, '/referee/command', 10)
        self.next_command_pub = self.create_publisher(String, '/referee/next_command', 10)
        self.status_pub = self.create_publisher(String, '/referee/status_message', 10)
        self.designated_pos_pub = self.create_publisher(Pose2D, '/referee/designated_position', 10)
        self.blue_on_pos_half_pub = self.create_publisher(Bool, '/referee/blue_on_positive_half', 10)
        self.team_blue_pub = self.create_publisher(String, '/referee/blue/team_info', 10)
        self.team_yellow_pub = self.create_publisher(String, '/referee/yellow/team_info', 10)

        # Import generated protobufs lazily to fail early if missing
        # The generated modules use top-level packages 'state' and 'geom'.
        # Ensure the local 'messages' folder is on sys.path so those imports resolve.
        try:
            import sys
            from pathlib import Path
            messages_root = Path(__file__).parent / 'messages'
            sys.path.insert(0, str(messages_root))
            import state.ssl_gc_referee_message_pb2 as ref_pb2
            self.ref_pb2 = ref_pb2
        except Exception as e:
            self.get_logger().error(f'Failed to import referee protobufs: {e}')
            raise

        # Timer
        timer_period = time_step_ms / 1000.0
        self.timer = self.create_timer(timer_period, self._tick)
        self.get_logger().info('SSL Referee Protobuf Connector Node Started')

    def _recv_packet(self) -> Optional[bytes]:
        try:
            data, addr = self.sock.recvfrom(65536)
            self.get_logger().debug(f"Received {len(data)} bytes from {addr}")
            return data
        except (BlockingIOError, InterruptedError):
            return None
        except socket.timeout:
            return None

    def _publish_team_info(self, info, pub):
        payload = {
            'name': info.name if hasattr(info, 'name') else '',
            'score': int(getattr(info, 'score', 0)),
            'red_cards': int(getattr(info, 'red_cards', 0)),
            'yellow_cards': int(getattr(info, 'yellow_cards', 0)),
            'yellow_card_times': list(getattr(info, 'yellow_card_times', [])),
            'timeouts': int(getattr(info, 'timeouts', 0)),
            'timeout_time': int(getattr(info, 'timeout_time', 0)),
            'goalkeeper': int(getattr(info, 'goalkeeper', 0)),
            'foul_counter': int(getattr(info, 'foul_counter', 0)) if info.HasField('foul_counter') else 0,
            'ball_placement_failures': int(getattr(info, 'ball_placement_failures', 0)) if info.HasField('ball_placement_failures') else 0,
            'can_place_ball': bool(getattr(info, 'can_place_ball', False)) if info.HasField('can_place_ball') else False,
            'max_allowed_bots': int(getattr(info, 'max_allowed_bots', 0)) if info.HasField('max_allowed_bots') else 0,
        }
        msg = String()
        msg.data = json.dumps(payload)
        pub.publish(msg)

    def _tick(self):
        packet = self._recv_packet()
        if not packet:
            return

        # Parse protobuf
        try:
            ref_msg = self.ref_pb2.Referee.FromString(packet)
        except Exception as e:
            self.get_logger().warn(f'Failed to parse referee message: {e}')
            return

        # Stage
        try:
            stage_name = self.ref_pb2.Referee.Stage.Name(ref_msg.stage)
            self.stage_pub.publish(String(data=stage_name))
        except Exception:
            pass

        # Command
        try:
            command_name = self.ref_pb2.Referee.Command.Name(ref_msg.command)
            self.command_pub.publish(String(data=command_name))
        except Exception:
            pass

        # Next command (optional)
        if ref_msg.HasField('next_command'):
            try:
                next_cmd_name = self.ref_pb2.Referee.Command.Name(ref_msg.next_command)
                self.next_command_pub.publish(String(data=next_cmd_name))
            except Exception:
                pass

        # Status message (optional)
        if ref_msg.HasField('status_message'):
            self.status_pub.publish(String(data=ref_msg.status_message))

        # Team on positive half (optional)
        if ref_msg.HasField('blue_team_on_positive_half'):
            self.blue_on_pos_half_pub.publish(Bool(data=bool(ref_msg.blue_team_on_positive_half)))

        # Designated position (optional)
        if ref_msg.HasField('designated_position'):
            dp = ref_msg.designated_position
            pose = Pose2D()
            # Values are in mm in referee message spec (SSL-Vision coordinates)
            # Keep units consistent with other nodes: map mm to meters here
            pose.x = dp.x / 1000.0
            pose.y = dp.y / 1000.0
            pose.theta = 0.0
            self.designated_pos_pub.publish(pose)

        # Team infos
        if ref_msg.HasField('blue'):
            self._publish_team_info(ref_msg.blue, self.team_blue_pub)
        if ref_msg.HasField('yellow'):
            self._publish_team_info(ref_msg.yellow, self.team_yellow_pub)


def main(args=None):
    rclpy.init(args=args)
    node = SSLRefereeProtobufToROS()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
