#!/usr/bin/env python3

import argparse
import socket

from ssl_vision_connector.messages.messages_robocup_ssl_wrapper_pb2 import (
    SSL_WrapperPacket,
)


def build_packet() -> SSL_WrapperPacket:
    packet = SSL_WrapperPacket()
    frame = packet.detection

    frame.frame_number = 1
    frame.t_capture = 0.0
    frame.t_sent = 0.0
    frame.camera_id = 0

    ball = frame.balls.add()
    ball.confidence = 0.9
    ball.area = 0
    ball.x = 500.0
    ball.y = -300.0
    ball.z = 0.0
    ball.pixel_x = 320.0
    ball.pixel_y = 240.0

    blue = frame.robots_blue.add()
    blue.confidence = 0.8
    blue.robot_id = 0
    blue.x = 1000.0
    blue.y = 2000.0
    blue.orientation = 0.5
    blue.pixel_x = 640.0
    blue.pixel_y = 360.0
    blue.height = 0.0

    yellow = frame.robots_yellow.add()
    yellow.confidence = 0.8
    yellow.robot_id = 2
    yellow.x = -1500.0
    yellow.y = 500.0
    yellow.orientation = -1.2
    yellow.pixel_x = 400.0
    yellow.pixel_y = 260.0
    yellow.height = 0.0

    return packet


def send_packet(group: str, port: int, ttl: int) -> None:
    packet = build_packet()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
    try:
        sock.sendto(packet.SerializeToString(), (group, port))
    finally:
        sock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a mock RoboCup SSL wrapper packet over UDP."
    )
    parser.add_argument(
        "--group",
        default="224.5.23.2",
        help="Multicast group to send to (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10006,
        help="UDP port to send to (default: %(default)s)",
    )
    parser.add_argument(
        "--ttl",
        type=int,
        default=1,
        help="Multicast TTL value (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    send_packet(args.group, args.port, args.ttl)


if __name__ == "__main__":
    main()
