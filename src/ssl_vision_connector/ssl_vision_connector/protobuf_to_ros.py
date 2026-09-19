"""SSL-Vision multicast -> real-world SI pose topics.

Publishes what the camera sees, in real meters and radians, in a
right-handed field frame (x toward the +x goal, y left, theta CCW):

    /vision/poses/{blue,yellow}/robot<i>   geometry_msgs/Pose2D  (m, rad)
    /vision/poses/ball                     geometry_msgs/Pose2D  (m)
    /vision/field                          std_msgs/Float32MultiArray
                                           [field_length_m, field_width_m]

No coordinate scaling happens here; the mapping to the policy's 9x6
training frame is done in the rl_strategy package. Downstream consumers
receive unscaled SI coordinates.

`mirror` parameter: our real ssl-vision calibration emits a mirrored frame
(y and theta run backwards vs. right-handed; determined empirically with the real
camera setup). mirror:=true (default) flips y and theta to
correct it. grSim's vision output is already right-handed: use mirror:=false
on port 10020.

Robot identity: SSL-Vision pattern ids flicker on a partial field, so the
top-k most-persistently-seen vision ids (presence counter with decay) map to
robot0..robot<k-1> per color, ordered by vision id.
"""
import socket

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32MultiArray

from .messages.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket

DECAY_PARAMETER = 0.9
MAX_VISION_IDS = 16


class SSLVisionProtobufToROS(Node):

    def __init__(self):
        super().__init__('protobuf_to_ros')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('blue_robot_count', 3),
                ('yellow_robot_count', 3),
                ('frequency', 60),
                # 224.5.23.2:10006 = real ssl-vision (our config)
                # 224.5.23.2:10020 = grSim default
                ('vision_ip', '224.5.23.2'),
                ('vision_port', 10006),
                # true: correct our mirrored ssl-vision calibration (real camera)
                # false: source is already right-handed (grSim)
                ('mirror', True),
            ]
        )
        gp = lambda n: self.get_parameter(n).get_parameter_value()
        self.robot_count = {
            'blue': gp('blue_robot_count').integer_value,
            'yellow': gp('yellow_robot_count').integer_value,
        }
        frequency = gp('frequency').integer_value
        self.mirror = gp('mirror').bool_value

        vision_ip = gp('vision_ip').string_value
        vision_port = gp('vision_port').integer_value
        self.get_logger().info(
            f'listening for vision on {vision_ip}:{vision_port} '
            f'(mirror={self.mirror})'
        )

        self.sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.sock.bind(('', vision_port))
        mreq = socket.inet_aton(vision_ip) + socket.inet_aton('0.0.0.0')
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        # Non-blocking: each tick drains the queue and publishes from the
        # freshest packets. A blocking one-packet-per-tick read falls behind
        # whenever the source outpaces the timer (grSim: ~235 pkt/s vs 60 Hz)
        # and the queue backlog turns into seconds of vision lag, so the
        # strategy acts on a seconds-old world. Same drain pattern as the RL
        # repository's deploy script.
        self.sock.setblocking(False)

        self.ball_publisher = self.create_publisher(
            Pose2D, '/vision/poses/ball', 10)
        self.field_publisher = self.create_publisher(
            Float32MultiArray, '/vision/field', 10)
        self.pose_publishers = {
            color: [
                self.create_publisher(
                    Pose2D, f'/vision/poses/{color}/robot{i}', 10)
                for i in range(self.robot_count[color])
            ]
            for color in ('blue', 'yellow')
        }

        self.presence_counter = {
            'blue': [0.0] * MAX_VISION_IDS,
            'yellow': [0.0] * MAX_VISION_IDS,
        }
        # cached field geometry (meters); None until first geometry packet
        self.field_length_m = None
        self.field_width_m = None

        self.create_timer(1.0 / frequency, self.get_protobuf_and_publish)
        self.get_logger().info('SSL Vision Protobuf Connector Node Started')

    # ---- robot identity ---------------------------------------------------

    def get_robot_id_converter(self, color):
        """Top-k most-seen vision ids, ordered by vision id -> ros index."""
        robot_list = [(count, vid) for vid, count
                      in enumerate(self.presence_counter[color])]
        top_k = sorted(robot_list, key=lambda x: x[0],
                       reverse=True)[:self.robot_count[color]]
        return [pair[1] for pair in sorted(top_k, key=lambda x: x[1])]

    def decay_presence_counter(self):
        for color in ('blue', 'yellow'):
            counters = self.presence_counter[color]
            for i in range(len(counters)):
                counters[i] *= DECAY_PARAMETER

    def increase_presence_counter(self, packet):
        for robot in packet.detection.robots_yellow:
            self.presence_counter['yellow'][robot.robot_id] += 1
        for robot in packet.detection.robots_blue:
            self.presence_counter['blue'][robot.robot_id] += 1

    # ---- publishing -------------------------------------------------------

    def publish_robot(self, robot, converter, publishers):
        if robot.robot_id not in converter:
            # likely a detection flicker; not one of our persistent robots
            return
        index = converter.index(robot.robot_id)

        msg = Pose2D()
        msg.x = robot.x / 1000.0
        msg.y = robot.y / 1000.0
        msg.theta = float(robot.orientation)
        if self.mirror:
            msg.y = -msg.y
            msg.theta = -msg.theta
        publishers[index].publish(msg)

    def get_protobuf_and_publish(self):
        # Drain everything queued since the last tick; process each packet so
        # the presence counters see all detections, publishing each packet as it is processed (the
        # last packet published per entity is the freshest).
        packets = []
        while True:
            try:
                data, _addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                packets.append(SSL_WrapperPacket.FromString(data))
            except Exception as e:
                self.get_logger().warn(
                    f'Failed to parse vision packet: {e}',
                    throttle_duration_sec=5.0)
        if not packets:
            return

        for packet in packets:
            # field geometry (mm -> m)
            try:
                if packet.HasField('geometry') and packet.geometry.HasField('field'):
                    self.field_length_m = packet.geometry.field.field_length / 1000.0
                    self.field_width_m = packet.geometry.field.field_width / 1000.0
            except Exception as e:
                self.get_logger().warn(f'Failed to parse geometry: {e}')

            # ball (first detection)
            if len(packet.detection.balls) > 0:
                ball = packet.detection.balls[0]
                ball_msg = Pose2D()
                ball_msg.x = ball.x / 1000.0
                ball_msg.y = ball.y / 1000.0
                if self.mirror:
                    ball_msg.y = -ball_msg.y
                ball_msg.theta = 0.0
                self.ball_publisher.publish(ball_msg)

            self.increase_presence_counter(packet)

            for color, robots in (('blue', packet.detection.robots_blue),
                                  ('yellow', packet.detection.robots_yellow)):
                converter = self.get_robot_id_converter(color)
                for robot in robots:
                    self.publish_robot(robot, converter, self.pose_publishers[color])

        # decay once per tick (not per packet) so the decay timescale follows
        # the timer frequency rather than the packet rate
        self.decay_presence_counter()

        if self.field_length_m is not None:
            field_msg = Float32MultiArray()
            field_msg.data = [self.field_length_m, self.field_width_m]
            self.field_publisher.publish(field_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SSLVisionProtobufToROS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
