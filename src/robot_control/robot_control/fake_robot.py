"""Plays a synthetic robot on the real robot topics, in real time, so the
odometry and robot_events nodes can be tested without hardware or a
simulator. Do not run it on the robot itself: it publishes imu,
wheels/state and wheels/cmd.

scenario:=odometry (default, sim_sensors.py, 47 s)
  imu                           (Imu)        gyro z with drifting bias, 200 Hz
  wheels/state                  (JointState) wheel speeds with slip/stale events, 50 Hz
  odom/ground_truth             (Odometry)   true pose and body velocity, 100 Hz
  odom/ground_truth/gyro_bias   (Float64)    true gyro bias

scenario:=events (sim_events.py, 30 s): impacts, stall, skid, push
  imu                           (Imu)        gyro z + accelerometer, 200 Hz
  wheels/state                  (JointState) wheel speeds + motor current (effort), 50 Hz
  wheels/cmd                    (Float32MultiArray) wheel setpoints, 50 Hz
  and logs "TRUE EVENT ..." when each scripted event happens, to compare
  with the events topic
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray, Float64

from . import sim_events
from .sim_sensors import DURATION, SyntheticRun, gyro_bias


class FakeRobot(Node):
    def __init__(self):
        super().__init__('fake_robot')
        seed = int(self.declare_parameter('seed', 0).value)
        self.scenario = self.declare_parameter('scenario', 'odometry').value
        if self.scenario == 'events':
            self.sim = sim_events.EventRun(seed=seed)
            self.events = self.sim.stream()
            self.duration = sim_events.DURATION
            self.true_events = sorted(
                [(start, f'{name} starts') for name, start, _end in sim_events.EVENTS]
                + [(end, f'{name} ends') for name, start, end in sim_events.EVENTS
                   if end != start])
            wheel = sim_events.SKID[0]
            self.true_events = [(t, txt + (f' (wheel {wheel})' if txt == 'skid starts' else ''))
                                for t, txt in self.true_events]
        else:
            self.sim = SyntheticRun(seed=seed)
            self.events = self.sim.events()
            self.duration = DURATION
            self.true_events = []
        self.next_event = 0
        self.next_truth = 0.0

        self.imu_pub = self.create_publisher(Imu, 'imu', 50)
        self.wheel_pub = self.create_publisher(JointState, 'wheels/state', 10)
        self.truth_pub = self.create_publisher(Odometry, 'odom/ground_truth', 10)
        self.bias_pub = self.create_publisher(Float64, 'odom/ground_truth/gyro_bias', 10)
        self.cmd_pub = self.create_publisher(Float32MultiArray, 'wheels/cmd', 10)
        self.t0 = self.get_clock().now().nanoseconds
        self.create_timer(0.002, self.tick)
        self.get_logger().info(f'fake robot: playing {self.duration:.0f} s of '
                               f'synthetic sensors ({self.scenario} scenario)')

    def stamp(self, t):
        return Time(nanoseconds=self.t0 + int(t * 1e9)).to_msg()

    def tick(self):
        elapsed = (self.get_clock().now().nanoseconds - self.t0) * 1e-9
        while self.true_events and self.true_events[0][0] <= elapsed:
            t, txt = self.true_events.pop(0)
            self.get_logger().info(f'TRUE EVENT at {t:.2f}s: {txt}')
        if elapsed > self.duration:
            self.get_logger().info('fake robot: done', once=True)
            return
        while self.next_event < len(self.events) and self.events[self.next_event][1] <= elapsed:
            kind, t, z = self.events[self.next_event]
            self.next_event += 1
            if kind == 'imu':
                msg = Imu()
                msg.header.stamp = self.stamp(t)
                msg.header.frame_id = 'imu'
                if self.scenario == 'events':
                    gyro, ax, ay = z
                    msg.linear_acceleration.x = ax
                    msg.linear_acceleration.y = ay
                    msg.linear_acceleration.z = 9.81
                else:
                    gyro = z
                msg.angular_velocity.z = float(gyro)
                self.imu_pub.publish(msg)
            else:
                msg = JointState()
                msg.header.stamp = self.stamp(t)
                if self.scenario == 'events':
                    speeds, currents, setpoints = z
                    msg.effort = [float(c) for c in currents]
                    self.cmd_pub.publish(Float32MultiArray(data=[float(v) for v in setpoints]))
                else:
                    speeds = z
                msg.name = [f'wheel_{i}' for i in range(len(speeds))]
                msg.velocity = [float(v) for v in speeds]
                self.wheel_pub.publish(msg)
        while self.scenario != 'events' and self.next_truth <= elapsed:
            t = self.next_truth
            self.next_truth += 0.01
            k = self.sim.index(t)
            x, y, th = self.sim.pose[k]
            vx, vy, w = self.sim.vel[k]
            msg = Odometry()
            msg.header.stamp = self.stamp(t)
            msg.header.frame_id = 'odom'
            msg.pose.pose.position.x = float(x)
            msg.pose.pose.position.y = float(y)
            msg.pose.pose.orientation.z = math.sin(th / 2.0)
            msg.pose.pose.orientation.w = math.cos(th / 2.0)
            msg.twist.twist.linear.x = float(vx)
            msg.twist.twist.linear.y = float(vy)
            msg.twist.twist.angular.z = float(w)
            self.truth_pub.publish(msg)
            self.bias_pub.publish(Float64(data=gyro_bias(t)))


def main(args=None):
    rclpy.init(args=args)
    node = FakeRobot()
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
