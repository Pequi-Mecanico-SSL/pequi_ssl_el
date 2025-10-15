#!/usr/bin/env python3

import argparse
import socket
import sys

from ssl_vision_connector.messages.messages_robocup_ssl_wrapper_pb2 import (
    SSL_WrapperPacket,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen for RoboCup SSL wrapper packets and dump their contents."
    )
    parser.add_argument(
        "--group",
        default="224.5.23.2",
        help="Multicast group to join (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10006,
        help="UDP port to bind to (default: %(default)s)",
    )
    parser.add_argument(
        "--iface",
        default="0.0.0.0",
        help="Interface address to use when joining the multicast group (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind(("", args.port))
    except OSError as exc:
        print(f"Failed to bind to port {args.port}: {exc}", file=sys.stderr)
        sys.exit(1)

    mreq = socket.inet_aton(args.group) + socket.inet_aton(args.iface)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    print(f"Listening for SSL wrapper packets on {args.group}:{args.port} ...")

    try:
        while True:
            data, addr = sock.recvfrom(4096)
            packet = SSL_WrapperPacket()
            packet.ParseFromString(data)

            detection = packet.detection
            print(
                f"From {addr[0]}:{addr[1]} frame={detection.frame_number} "
                f"balls={len(detection.balls)} blue={len(detection.robots_blue)} "
                f"yellow={len(detection.robots_yellow)}"
            )

            if detection.balls:
                ball = detection.balls[0]
                print(
                    f"  Ball: x={ball.x:.2f}mm y={ball.y:.2f}mm conf={ball.confidence:.2f}"
                )

            for robot in detection.robots_blue:
                print(
                    f"  Blue[{robot.robot_id}] pos=({robot.x:.2f},{robot.y:.2f})mm "
                    f"theta={robot.orientation:.2f} conf={robot.confidence:.2f}"
                )
            for robot in detection.robots_yellow:
                print(
                    f"  Yellow[{robot.robot_id}] pos=({robot.x:.2f},{robot.y:.2f})mm "
                    f"theta={robot.orientation:.2f} conf={robot.confidence:.2f}"
                )
            print("---")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
