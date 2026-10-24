"""Adapter between rSim and the pequi stack, for testing on the simulator.
Test harness only, not part of the repo.

Time       run with use_sim_time:=true; everything is stamped with the
           simulated time of its step, t = 0.016 * step (= /clock)
Truth      /sim_harness/truth  Float64MultiArray [t, step, rSim state...] (recorder)
           /vision/ground_truth/<color>/robot<i>  Odometry, body twist (vision_check)
           <stack>/odom/ground_truth (+ /gyro_bias)                   (odom_check)
Camera     SSL-Vision UDP packets (unicast to 127.0.0.1) from the true poses with noise, latency,
           loss, ghosts, blackouts -> ssl_vision_connector, like ssl-vision
Relay      /<color>/robot<i>/cmd_vel -> /simulator/cmd/<color>/robot<i>; the
           stack robot is driven by its velocity_controller output
           (<stack>/cmd_vel_corrected), as on the real robot
Robot      for the stack robot, emulates the Pi's sensors from the true motion:
           <stack>/imu           gyro (bias ramp + noise), accelerometer
                                 (body-frame accel + noise + tilt, +-4 g clip)
           <stack>/wheels/state  encoders of OUR wheel geometry (no slip) +
                                 noise; effort = modeled ESC current from
                                 <stack>/wheels/cmd setpoints
           /sim_harness/skid     [wheel, extra rad/s, seconds] injects a skid
"""
import math
import random
import socket
from collections import deque

import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray, Float64, Float64MultiArray

from ssl_vision_connector.messages.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket
from robot_control.geometry import (DEFAULT_ROBOT_RADIUS, DEFAULT_WHEEL_ANGLES_DEG,
                                    DEFAULT_WHEEL_RADIUS, wheel_matrix)

COLORS = ('blue', 'yellow')
SIM_DT = 0.016          # rsim_ros: time_step_ms = 1000 // 60
G4 = 4.0 * 9.80665


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def parse_state(data, counts):
    """rSim state -> ball (x, y), {key: (x, y, th, vxg, vyg, w)} in m, rad."""
    s = data
    ball = (s[0], s[1])
    robots = {}
    j = 5
    for c in COLORS:
        for i in range(counts[c]):
            x, y, th_deg, vx, vy, w_deg = s[j:j + 6]
            robots[(c, i)] = (x, y, wrap(math.radians(th_deg)), vx, vy, math.radians(w_deg))
            j += 11
    return ball, robots


class Harness(Node):
    def __init__(self):
        super().__init__('sim_harness')
        P = dict(blue_robot_count=3, yellow_robot_count=3, stack_robot='blue/robot0',
                 camera_hz=60.0, sigma_xy=0.005, sigma_theta=0.03, latency_s=0.02,
                 loss=0.03, ghost=0.005, blackout_every_s=0.0, blackout_s=0.2,
                 vision_port=10020, vision_ip='127.0.0.1', gyro_bias=0.02, gyro_bias_end=0.05,
                 bias_ramp_s=60.0, sigma_gyro=0.01, sigma_accel=0.3, sigma_wheel=0.3,
                 seed=1)
        for k, v in P.items():
            self.declare_parameter(k, v)
        p = {k: self.get_parameter(k).value for k in P}
        self.p = p
        random.seed(p['seed'])
        self.rng = np.random.default_rng(p['seed'])
        self.counts = {'blue': p['blue_robot_count'], 'yellow': p['yellow_robot_count']}
        self.stack = p['stack_robot']
        self.stack_key = tuple(self.stack.replace('robot', '').split('/')) if self.stack else None
        if self.stack_key:
            self.stack_key = (self.stack_key[0], int(self.stack_key[1]))
        self.J = np.array(wheel_matrix(DEFAULT_WHEEL_ANGLES_DEG, DEFAULT_ROBOT_RADIUS,
                                       DEFAULT_WHEEL_RADIUS))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

        self.offsets = deque()    # (arrival, arrival - SIM_DT * step)
        self.history = deque()    # (t, robots, ball) per simulator step
        self.t_first = None
        self.frame = 0
        self.prev = None          # (step, vxg, vyg) of the stack robot

        self.truth_pub = self.create_publisher(Float64MultiArray, '/sim_harness/truth', 50)
        self.gt_pubs, self.sim_cmd = {}, {}
        for c in COLORS:
            for i in range(self.counts[c]):
                key = (c, i)
                self.gt_pubs[key] = self.create_publisher(
                    Odometry, f'/vision/ground_truth/{c}/robot{i}', 10)
                self.sim_cmd[key] = self.create_publisher(Twist, f'/simulator/cmd/{c}/robot{i}', 10)
                src = 'cmd_vel_corrected' if key == self.stack_key else 'cmd_vel'
                self.create_subscription(Twist, f'/{c}/robot{i}/{src}',
                                         lambda m, k=key: self.sim_cmd[k].publish(m), 10)
        self.create_subscription(Float64MultiArray, '/simulator/state', self.on_state, 50)

        if self.stack_key:
            ns = '/' + self.stack
            self.imu_pub = self.create_publisher(Imu, f'{ns}/imu', 50)
            self.wheel_pub = self.create_publisher(JointState, f'{ns}/wheels/state', 10)
            self.odom_truth_pub = self.create_publisher(Odometry, f'{ns}/odom/ground_truth', 10)
            self.bias_truth_pub = self.create_publisher(Float64, f'{ns}/odom/ground_truth/gyro_bias', 10)
            self.create_subscription(Float32MultiArray, f'{ns}/wheels/cmd', self.on_setpoints, 10)
            self.create_subscription(Float32MultiArray, '/sim_harness/skid', self.on_skid, 10)
        self.setpoints = np.zeros(4)
        self.skid = None          # (wheel, extra, t_end)

        self.create_timer(1.0 / p['camera_hz'], self.camera)
        self.get_logger().info(f'sim harness up: stack={self.stack} params={p}')

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def stamp(t):
        return Time(nanoseconds=int(t * 1e9)).to_msg()

    # ---- truth --------------------------------------------------------------

    def sim_time(self, step):
        return SIM_DT * step      # matches /clock published by sim_node.py

    def on_state(self, m):
        step = int(m.data[0])
        t = self.sim_time(step)
        if self.t_first is None:
            self.t_first = t
        ball, robots = parse_state(m.data[1:], self.counts)
        self.truth_pub.publish(Float64MultiArray(data=[t, float(step)] + list(m.data[1:])))
        self.history.append((t, robots, ball))
        while len(self.history) > 200:
            self.history.popleft()
        for key, (x, y, th, vxg, vyg, w) in robots.items():
            c, s = math.cos(th), math.sin(th)
            vxb, vyb = c * vxg + s * vyg, -s * vxg + c * vyg
            odom = Odometry()
            odom.header.stamp = self.stamp(t)
            odom.header.frame_id = 'field'
            odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
            odom.twist.twist.linear.x, odom.twist.twist.linear.y = vxb, vyb
            odom.twist.twist.angular.z = w
            self.gt_pubs[key].publish(odom)
            if key == self.stack_key:
                self.odom_truth_pub.publish(odom)
                self.emulate_robot(t, step, th, vxg, vyg, np.array([vxb, vyb, w]))

    # ---- camera -------------------------------------------------------------

    def camera(self):
        if not self.history:
            return
        now = self.now_s()
        target = now - self.p['latency_s']
        snap = None
        for item in self.history:
            if item[0] <= target:
                snap = item
            else:
                break
        if snap is None:
            return
        t_cap, robots, ball = snap
        self.frame += 1
        every, length = self.p['blackout_every_s'], self.p['blackout_s']
        if every > 0 and ((now - self.t_first) % every) < length:
            return
        if random.random() < self.p['loss']:
            return
        packet = SSL_WrapperPacket()
        det = packet.detection
        det.frame_number = self.frame
        det.camera_id = 0
        det.t_capture = t_cap + 1000.0     # vision PC clock differs; only the
        det.t_sent = now + 1000.0          # difference is used
        sxy, sth = self.p['sigma_xy'], self.p['sigma_theta']
        b = det.balls.add()
        b.confidence = 0.9
        b.x = 1000.0 * (ball[0] + random.gauss(0, sxy))
        b.y = 1000.0 * (ball[1] + random.gauss(0, sxy))
        b.pixel_x = b.pixel_y = 0.0
        for (c, i), (x, y, th, *_v) in robots.items():
            r = (det.robots_blue if c == 'blue' else det.robots_yellow).add()
            r.confidence = 0.9
            r.robot_id = i
            r.x = 1000.0 * (x + random.gauss(0, sxy))
            r.y = 1000.0 * (y + random.gauss(0, sxy))
            r.orientation = th + random.gauss(0, sth)
            if random.random() < self.p['ghost']:
                r.x += 1000.0
            r.pixel_x = r.pixel_y = 0.0
        # unicast to this machine by default (the connector also accepts it), so
        # camera packets never reach another machine or test run; set
        # vision_ip:=224.5.23.2 to send multicast like ssl-vision
        self.sock.sendto(packet.SerializeToString(), (self.p['vision_ip'], int(self.p['vision_port'])))

    # ---- stack robot sensors -----------------------------------------------

    def bias(self, t):
        f = min(1.0, (t - self.t_first) / self.p['bias_ramp_s'])
        return self.p['gyro_bias'] + f * (self.p['gyro_bias_end'] - self.p['gyro_bias'])

    def on_setpoints(self, m):
        if len(m.data) == 4:
            self.setpoints = np.array(m.data, dtype=float)

    def on_skid(self, m):
        wheel, extra, dur = m.data
        self.skid = (int(wheel), float(extra), self.now_s() + float(dur))
        self.get_logger().info(f'injecting skid: wheel {int(wheel)} +{extra} rad/s for {dur} s')

    def emulate_robot(self, t, step, th, vxg, vyg, vb):
        stamp = self.stamp(t)
        if self.prev is None or step <= self.prev[0]:
            ag = np.zeros(2)
        else:
            ag = (np.array([vxg, vyg]) - self.prev[1:]) / (SIM_DT * (step - self.prev[0]))
        self.prev = (step, vxg, vyg)
        c, s = math.cos(th), math.sin(th)
        ab = np.array([c * ag[0] + s * ag[1], -s * ag[0] + c * ag[1]])
        ab += np.array([0.15, -0.10]) + self.rng.normal(0, self.p['sigma_accel'], 2)
        ab = np.clip(ab, -G4, G4)
        bias = self.bias(t)
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = 'imu'
        imu.angular_velocity.z = float(vb[2] + bias + self.rng.normal(0, self.p['sigma_gyro']))
        imu.linear_acceleration.x, imu.linear_acceleration.y = float(ab[0]), float(ab[1])
        imu.linear_acceleration.z = 9.81
        self.imu_pub.publish(imu)
        self.bias_truth_pub.publish(Float64(data=bias))

        true_wheels = self.J @ vb
        speeds = true_wheels + self.rng.normal(0, self.p['sigma_wheel'], 4)
        if self.skid and self.now_s() < self.skid[2]:
            speeds[self.skid[0]] += self.skid[1]
        current = np.clip(0.1 * (self.setpoints - true_wheels), -1.3, 1.3)
        current += 0.05 * np.sign(true_wheels) + self.rng.normal(0, 0.03, 4)
        js = JointState()
        js.header.stamp = stamp
        js.name = [f'wheel_{i}' for i in range(4)]
        js.velocity = [float(v) for v in speeds]
        js.effort = [float(v) for v in current]
        self.wheel_pub.publish(js)


def main():
    rclpy.init()
    node = Harness()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
