"""Scripted drive for blue robot0 (sim time): idle, then laps of a rectangle
driven with translation-only commands (heading hold active), a full spin
after each lap and a pause after lap 3. Publishes /blue/robot0/cmd_vel
(body frame, like rl_strategy) and the current phase on /driver/phase.

The rectangle is steered with the TRUE pose (field-frame velocity toward the
next corner, rotated into the body frame with the true heading), so the path
stays in an empty region whatever the heading hold does; what is measured is
how much the robot's real heading changes while it translates.
"""
import json
import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String

A, B, C, D = (0.5, 1.5), (0.5, 2.5), (-1.0, 2.5), (-1.0, 1.5)


def build_plan(laps=6):
    plan = [('idle', 12.0, None), ('goto', 2.5, (D, 0.75))]
    for lap in range(laps):
        plan += [('goto', 2.5, (A, 0.75)), ('goto', 2.5, (B, 0.5)),
                 ('goto', 2.5, (C, 0.75)), ('goto', 2.5, (D, 0.5)),
                 ('spin', 2 * math.pi / 4.0, 4.0), ('idle', 0.5, None)]
        if lap == 2:
            plan.append(('idle', 3.0, None))
    return plan


class Driver(Node):
    def __init__(self):
        super().__init__('driver', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.pub = self.create_publisher(Twist, '/blue/robot0/cmd_vel', 10)
        self.phase_pub = self.create_publisher(String, '/driver/phase', 10)
        self.create_subscription(Float64MultiArray, '/sim_harness/truth', self.on_truth, 10)
        self.plan = build_plan()
        self.total = sum(p[1] for p in self.plan)
        self.pose = None
        self.t0 = None
        self.phase = -1
        self.done = False
        self.create_timer(1.0 / 30.0, self.tick)

    def on_truth(self, m):
        x, y, th_deg = m.data[7], m.data[8], m.data[9]
        self.pose = (x, y, math.radians(th_deg))

    def tick(self):
        if self.pose is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        el = now - self.t0
        acc, k = 0.0, None
        for i, p in enumerate(self.plan):
            if el < acc + p[1]:
                k = i
                break
            acc += p[1]
        cmd = Twist()
        if k is None:
            self.pub.publish(cmd)
            self.done = True
            return
        kind, _dur, arg = self.plan[k]
        if k != self.phase:
            self.phase = k
            self.phase_pub.publish(String(data=json.dumps({'k': k, 'kind': kind, 't': now})))
        if kind == 'goto':
            (tx, ty), vmax = arg
            x, y, th = self.pose
            vx, vy = 3.0 * (tx - x), 3.0 * (ty - y)
            n = math.hypot(vx, vy)
            if n > vmax:
                vx, vy = vx * vmax / n, vy * vmax / n
            c, s = math.cos(th), math.sin(th)
            cmd.linear.x, cmd.linear.y = c * vx + s * vy, -s * vx + c * vy
        elif kind == 'spin':
            cmd.angular.z = arg
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = Driver()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
