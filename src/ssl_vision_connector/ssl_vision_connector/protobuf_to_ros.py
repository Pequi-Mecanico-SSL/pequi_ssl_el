from typing import List
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Twist

import socket

from .messages.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket


# Simulator uses a 9 by 6 field and flips y-axis
# Use geometry from wrapper packets to scale to simulator field
# TODO: remove unecessary exception handling code


class SSLVisionProtobufToROS(Node):

    def __init__(self):
        super().__init__('protobuf_to_ros')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('blue_robot_count', 3),
                ('yellow_robot_count', 3),
                ('frequency', 60),
            ]
        )
        self.robot_count = {}
        self.robot_count['blue'] = self.get_parameter('blue_robot_count').get_parameter_value().integer_value
        self.robot_count['yellow'] = self.get_parameter('yellow_robot_count').get_parameter_value().integer_value

        frequency = self.get_parameter('frequency').get_parameter_value().integer_value
        time_step_ms = 1000 // frequency


        # Protobuf connection setup
        self.ssl_vision_ip = '224.5.23.2'
        self.ssl_vision_port = 10006

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.sock.bind(('', self.ssl_vision_port))
        mreq = socket.inet_aton(self.ssl_vision_ip) + socket.inet_aton('0.0.0.0')
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        self.ball_publisher = self.create_publisher(Pose2D, '/simulator/poses/ball', 10)



        self.pose_publishers = {}
        self.pose_publishers['blue'] = []
        self.pose_publishers['yellow'] = []

        for i in range(self.robot_count['blue']):
            self.pose_publishers['blue'].append(
                self.create_publisher(Pose2D, f'/simulator/poses/blue/robot{i}', 10)
            )
        for i in range(self.robot_count['yellow']):
            self.pose_publishers['yellow'].append(
                self.create_publisher(Pose2D, f'/simulator/poses/yellow/robot{i}', 10)
            )

        self.presence_counter = {}
        self.presence_counter['blue'] = [0 for i in range(16)]
        self.presence_counter['yellow'] = [0 for i in range(16)]
        
        timer_period = time_step_ms / 1000.0  # seconds
        self.timer = self.create_timer(timer_period, self.get_protobuf_and_publish)

        self.get_logger().info('SSL Vision Protobuf Connector Node Started')

        # Cached field geometry from SSL-Vision (meters). None until first geometry packet.
        self.field_length_m = None
        self.field_width_m = None
        # Simulator field dimensions in meters (Division B: 9 x 6)
        self.sim_length_m = 9.0
        self.sim_width_m = 6.0

    
    def get_robot_id_converter(self, color: str ):
        print(color, "presence counter: ", self.presence_counter[color])
        robot_list = [(count, id) for id, count in enumerate(self.presence_counter[color])]
        print(color, "robot list: ", robot_list)
        sorted_by_count = sorted(robot_list, key=lambda x: x[0], reverse=True)
        print("sorted by count: ", sorted_by_count)
        top_k = sorted_by_count[:self.robot_count[color]]
        print("top k: ", top_k)
        sorted_by_id = sorted(top_k, key=lambda x: x[1])
        print("sorted_by_id: ", sorted_by_id)
        converter = [pair[1] for pair in sorted_by_id]
        print("converter: ", converter)
        return converter

    def decay_presence_counter(self):
        DECAY_PARAMETER = 0.9

        for i in range(len(self.presence_counter['yellow'])):
            self.presence_counter['yellow'][i] *= DECAY_PARAMETER

        for i in range(len(self.presence_counter['blue'])):
            self.presence_counter['blue'][i] *= DECAY_PARAMETER
        
        #print("Blue presence counter: ", self.blue_robot_id_presence_counter)
        #print("Yellow presence counter: ", self.yellow_robot_id_presence_counter)
        
    def increase_presence_counter(self, ssl_protobuf_packet):
        for yellow_robot in ssl_protobuf_packet.detection.robots_yellow:
            self.presence_counter['yellow'][yellow_robot.robot_id] += 1
        for blue_robot in ssl_protobuf_packet.detection.robots_blue:
            self.presence_counter['blue'][blue_robot.robot_id] += 1
        


    def check_and_publish_robot(self, robot, robot_id_converter, publishers):
            print(robot)

            if robot.robot_id not in robot_id_converter:
                # This means the robot that was detected might be a flicker of the vision system
                print("Robot id", robot.robot_id, "is not in the top k robots, classifying as flicker")
                return
            else:
                index = robot_id_converter.index(robot.robot_id)
                print("Robot ", robot.robot_id, " has a ros id of", index)

            msg = Pose2D()

            # Scale using geometry if available: convert mm -> m and map real field to simulator field
            print("PROTOBUF MESSAGE: (x:",robot.x,", y:",robot.y,", orientation:",robot.orientation,")\n")
            if self.field_length_m and self.field_width_m:
                scale_x = self.sim_length_m / self.field_length_m
                scale_y = self.sim_width_m / self.field_width_m
            else:
                # Fallback: assume SSL-EL (4.5 x 3 m) to simulator (9 x 6 m)
                scale_x = self.sim_length_m / 4.5
                scale_y = self.sim_width_m / 3.0

            msg.x = (robot.x / 1000.0) * scale_x
            msg.y = -(robot.y / 1000.0) * scale_y
            msg.theta = -robot.orientation
            publishers[index].publish(msg)


    def get_protobuf_and_publish(self):
        # Get converter from robot_id to our 0 to robot_count IDs
        robot_id_converter = {}
        robot_id_converter['blue'] = self.get_robot_id_converter('blue')
        robot_id_converter['yellow'] = self.get_robot_id_converter('yellow')

        # Units are meters, meters/s, degrees
        # state is [ball_x, ball_y, ball_z, ball_v_x, ball_v_y,
        #           blue_0_x, blue_0_y, blue_0_angle, blue_0_v_x, blue_0_v_y, blue_0_v_angle,
        #           blue_0_infrared, blue_0_desired_wheel0_speed, blue_0_desired_wheel1_speed,
        #           blue_0_desired_wheel2_speed, blue_0_desired_wheel3_speed, ...]

        # Get protobuf datagram
        data = self.sock.recvfrom(4096)
        self.get_logger().info(f"Received {len(data[0])} bytes from {data[1]}")
        ssl_protobuf_packet = SSL_WrapperPacket.FromString(data[0])

        # Update cached field geometry if provided
        try:
            if ssl_protobuf_packet.HasField('geometry') and ssl_protobuf_packet.geometry.HasField('field'):
                # Convert mm to meters
                self.field_length_m = ssl_protobuf_packet.geometry.field.field_length / 1000.0
                self.field_width_m = ssl_protobuf_packet.geometry.field.field_width / 1000.0
        except Exception as e:
            # Keep previous geometry on any parsing error
            self.get_logger().warn(f"Failed to parse geometry: {e}")

        #print("Detection = ", ssl_protobuf_packet)

        # Publish ball position
        if len(ssl_protobuf_packet.detection.balls) > 0:
            ball_msg = Pose2D()
            protobuf_ball = ssl_protobuf_packet.detection.balls[0]
            #print("Ball: ", protobuf_ball)
            if self.field_length_m and self.field_width_m:
                scale_x = self.sim_length_m / self.field_length_m
                scale_y = self.sim_width_m / self.field_width_m
            else:
                scale_x = self.sim_length_m / 4.5
                scale_y = self.sim_width_m / 3.0
            ball_msg.x = (protobuf_ball.x / 1000.0) * scale_x
            ball_msg.y = -(protobuf_ball.y / 1000.0) * scale_y
            ball_msg.theta = 0.0
            self.ball_publisher.publish(ball_msg)
        
        # Increase presence counter
        self.increase_presence_counter(ssl_protobuf_packet)
        
        # Decay presence counter
        self.decay_presence_counter()
            
        
        for blue_robot in ssl_protobuf_packet.detection.robots_blue:
            self.check_and_publish_robot(blue_robot, robot_id_converter['blue'], self.pose_publishers['blue'])
        
        for yellow_robot in ssl_protobuf_packet.detection.robots_yellow:
            self.check_and_publish_robot(yellow_robot, robot_id_converter['yellow'], self.pose_publishers['yellow'])

def main(args=None):
    rclpy.init(args=args)

    protobuf_to_ros = SSLVisionProtobufToROS()

    rclpy.spin(protobuf_to_ros)

    protobuf_to_ros.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
