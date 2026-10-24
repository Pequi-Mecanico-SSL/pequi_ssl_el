"""Scripted event course (sim time). Blue robot0 runs the robot stack
(/blue/robot0/cmd_vel -> velocity_controller -> simulator); blue robot1 is
the "opponent", commanded directly (/blue/robot1/cmd_vel). Both are steered
with true poses from /sim_harness/truth. Phases are announced on
/driver/phase so the analysis can match detections to them.

  wall    robot0 drives into the +y wall at 1 m/s and keeps pushing 2 s
  back    robot0 backs off to y = 1.5 and stops
  push    robot0 stands still; robot1 drives into it from -x and shoves it
  side    robot0 drives +x; robot1 rams it from -y
  skid    robot0 drives -x; wheel 1 is made to skid for 1 s (injected)
"""
import json
import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String


def toward(pose, target, vmax, k=3.0):
    x, y, th = pose
    vx, vy = k * (target[0] - x), k * (target[1] - y)
    n = math.hypot(vx, vy)
    if n > vmax:
        vx, vy = vx * vmax / n, vy * vmax / n
    c, s = math.cos(th), math.sin(th)
    return c * vx + s * vy, -s * vx + c * vy


class Driver(Node):
    def __init__(self):
        super().__init__('driver', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.pub0 = self.create_publisher(Twist, '/blue/robot0/cmd_vel', 10)
        self.pub1 = self.create_publisher(Twist, '/blue/robot1/cmd_vel', 10)
        self.skid_pub = self.create_publisher(Float32MultiArray, '/sim_harness/skid', 10)
        self.phase_pub = self.create_publisher(String, '/driver/phase', 10)
        self.create_subscription(Float64MultiArray, '/sim_harness/truth', self.on_truth, 10)
        self.p0 = self.p1 = None
        # (name, seconds)
        self.plan = [('idle', 3.0), ('wall', 5.5), ('back', 3.0), ('idle', 1.5),
                     ('approach', 3.0), ('push', 3.0), ('retreat', 2.5), ('idle', 1.0),
                     ('setup_side', 4.0), ('side', 3.0), ('idle', 1.5),
                     ('setup_skid', 3.0), ('skid', 3.0), ('idle', 2.0)]
        self.t0 = None
        self.phase = -1
        self.done = False
        self.skid_sent = False
        self.create_timer(1.0 / 30.0, self.tick)

    def on_truth(self, m):
        a = m.data
        self.p0 = (a[7], a[8], math.radians(a[9]))
        self.p1 = (a[18], a[19], math.radians(a[20]))

    def tick(self):
        if self.p0 is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        el = now - self.t0
        acc, k = 0.0, None
        for i, (_n, dur) in enumerate(self.plan):
            if el < acc + dur:
                k = i
                break
            acc += dur
        c0, c1 = Twist(), Twist()
        if k is None:
            self.pub0.publish(c0)
            self.pub1.publish(c1)
            self.done = True
            return
        name = self.plan[k][0]
        if k != self.phase:
            self.phase = k
            self.phase_pub.publish(String(data=json.dumps({'k': k, 'kind': name, 't': now})))
        if name == 'wall':
            c0.linear.x, c0.linear.y = toward(self.p0, (self.p0[0], 5.0), 1.0)
        elif name == 'back':
            c0.linear.x, c0.linear.y = toward(self.p0, (-1.0, 1.5), 0.75)
        elif name == 'approach':           # robot1 lines up 0.6 m to the -x side of robot0
            c1.linear.x, c1.linear.y = toward(self.p1, (self.p0[0] - 0.6, self.p0[1]), 1.0)
        elif name == 'push':               # robot1 drives through robot0's position
            c1.linear.x, c1.linear.y = toward(self.p1, (self.p0[0] + 1.0, self.p0[1]), 1.0)
        elif name == 'retreat':
            c1.linear.x, c1.linear.y = toward(self.p1, (-2.5, 0.0), 1.0)
        elif name == 'setup_side':         # robot0 to (-2, 1.2), robot1 below the path
            c0.linear.x, c0.linear.y = toward(self.p0, (-2.0, 1.2), 0.75)
            c1.linear.x, c1.linear.y = toward(self.p1, (-0.6, 0.0), 1.0)
        elif name == 'side':               # robot0 crosses, robot1 aims at it
            c0.linear.x, c0.linear.y = toward(self.p0, (1.5, 1.2), 0.8)
            c1.linear.x, c1.linear.y = toward(self.p1, (self.p0[0], self.p0[1]), 1.5, k=10.0)
        elif name == 'setup_skid':
            c0.linear.x, c0.linear.y = toward(self.p0, (1.5, 2.2), 0.75)
            c1.linear.x, c1.linear.y = toward(self.p1, (-2.5, 0.0), 1.0)
        elif name == 'skid':
            c0.linear.x, c0.linear.y = toward(self.p0, (-1.5, 2.2), 0.8)
            if el - acc > 1.0 and not self.skid_sent:
                self.skid_pub.publish(Float32MultiArray(data=[1.0, 6.0, 1.0]))
                self.skid_sent = True
                self.phase_pub.publish(String(data=json.dumps({'k': k, 'kind': 'skid_injected', 't': now})))
        self.pub0.publish(c0)
        self.pub1.publish(c1)


def main():
    rclpy.init()
    node = Driver()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
