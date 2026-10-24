"""Fake SSL-Vision source with known ground truth, for testing without a
simulator or camera.

Sends SSL wrapper packets over UDP multicast like ssl-vision does, with
realistic defects: position/orientation noise, processing latency
(t_sent - t_capture), random packet loss, a 200 ms blackout every 5 s and
occasional ghost detections 1 m away. The noise-free state is published as

    /vision/ground_truth/{blue,yellow}/robot0   nav_msgs/Odometry (200 Hz)

  blue robot0     drives a figure-eight while turning
  yellow robot0   stands still (for the stationary jitter check)

Defaults to the grSim port (10020, right-handed) so it never collides with
the real camera; run the connector with vision_port:=10020 mirror:=false.
"""
import math
import random
import socket

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from ssl_vision_connector.messages.messages_robocup_ssl_wrapper_pb2 import (
    SSL_WrapperPacket,
)


def blue_truth(t):
    """(x, y, theta, vx_body, vy_body, omega) of the moving robot."""
    x, y = 1.2 * math.sin(0.5 * t), 0.7 * math.sin(1.0 * t)
    vxf, vyf = 0.6 * math.cos(0.5 * t), 0.7 * math.cos(1.0 * t)
    th = 2.0 * math.sin(0.4 * t)
    w = 0.8 * math.cos(0.4 * t)
    c, s = math.cos(th), math.sin(th)
    return (x, y, th, c * vxf + s * vyf, -s * vxf + c * vyf, w)


def yellow_truth(_t):
    return (-0.8, 0.4, 1.0, 0.0, 0.0, 0.0)


TRUTHS = {'blue': blue_truth, 'yellow': yellow_truth}


class FakeVision(Node):

    def __init__(self):
        super().__init__('fake_vision')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('vision_ip', '224.5.23.2'),
                ('vision_port', 10020),
                ('rate_hz', 60.0),
                ('sigma_xy', 0.005),
                ('sigma_theta', 0.03),
                ('latency_s', 0.02),
                ('loss', 0.05),
                ('ghost', 0.01),
                ('blackout_every_s', 5.0),
                ('blackout_s', 0.2),
            ]
        )
        gp = lambda n: self.get_parameter(n).value
        self.target = (gp('vision_ip'), int(gp('vision_port')))
        self.sigma_xy = gp('sigma_xy')
        self.sigma_theta = gp('sigma_theta')
        self.latency = gp('latency_s')
        self.loss = gp('loss')
        self.ghost = gp('ghost')
        self.blackout_every = gp('blackout_every_s')
        self.blackout = gp('blackout_s')

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

        self.truth_pubs = {
            color: self.create_publisher(
                Odometry, f'/vision/ground_truth/{color}/robot0', 10)
            for color in TRUTHS
        }
        self.t0 = self.now_s()
        self.frame = 0
        self.create_timer(1.0 / gp('rate_hz'), self.send_frame)
        self.create_timer(0.005, self.publish_truth)
        self.get_logger().info(
            f'fake vision -> {self.target[0]}:{self.target[1]} at {gp("rate_hz")} Hz')

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def send_frame(self):
        now = self.now_s()
        t_capture = now - self.latency
        t = t_capture - self.t0
        self.frame += 1
        if (t % self.blackout_every) < self.blackout or random.random() < self.loss:
            return

        packet = SSL_WrapperPacket()
        det = packet.detection
        det.frame_number = self.frame
        det.camera_id = 0
        # the vision PC's clock differs from ours; only the difference matters
        det.t_capture = t_capture + 1000.0
        det.t_sent = now + 1000.0
        for color, truth in TRUTHS.items():
            x, y, th, *_ = truth(t)
            robot = (det.robots_blue if color == 'blue' else det.robots_yellow).add()
            robot.confidence = 0.9
            robot.robot_id = 0
            robot.x = 1000.0 * (x + random.gauss(0.0, self.sigma_xy))
            robot.y = 1000.0 * (y + random.gauss(0.0, self.sigma_xy))
            robot.orientation = th + random.gauss(0.0, self.sigma_theta)
            if random.random() < self.ghost:
                robot.x += 1000.0
            robot.pixel_x = robot.pixel_y = 0.0
        self.sock.sendto(packet.SerializeToString(), self.target)

    def publish_truth(self):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9 - self.t0
        for color, truth in TRUTHS.items():
            x, y, th, vx, vy, w = truth(t)
            msg = Odometry()
            msg.header.stamp = now.to_msg()
            msg.header.frame_id = 'field'
            msg.child_frame_id = f'{color}/robot0'
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.orientation.z = math.sin(th / 2.0)
            msg.pose.pose.orientation.w = math.cos(th / 2.0)
            msg.twist.twist.linear.x = vx
            msg.twist.twist.linear.y = vy
            msg.twist.twist.angular.z = w
            self.truth_pubs[color].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
