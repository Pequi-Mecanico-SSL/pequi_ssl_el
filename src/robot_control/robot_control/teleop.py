"""Joystick teleop: /dev/input/js* -> cmd_vel. Stdlib only (kernel joystick API).

Default mapping is the RadioMaster Boxer over BLE ("ExpressLRS Joystick"):
right stick = translation (axis1 up = forward, axis0 right = -y), left stick
X (axis4) = rotation (right = clockwise). axis2 is the arm switch: positive = enable motors,
negative = estop.

Scales are signed: wheel-frame value = axis * scale, so a negative scale
inverts the control. Publishes zeros while sticks are centered so the bridge
deadman stays fed; on device loss it publishes zeros and keeps retrying.
"""
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

JS_EVENT = struct.Struct('<IhBB')
EV_BUTTON, EV_AXIS = 0x01, 0x02


class Teleop(Node):
    def __init__(self):
        super().__init__('teleop')
        self.declare_parameter('device', '/dev/input/js0')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('axis_x', 1)
        self.declare_parameter('axis_y', 0)
        self.declare_parameter('axis_w', 4)
        self.declare_parameter('scale_x', 1.0)    # m/s at full deflection
        self.declare_parameter('scale_y', -1.0)   # m/s
        self.declare_parameter('scale_w', -3.0)   # rad/s
        self.declare_parameter('deadzone', 0.1)
        self.declare_parameter('arm_axis', 2)     # up = enable, down = estop; -1 to disable
        self.declare_parameter('estop_button', -1)

        p = lambda n: self.get_parameter(n).value
        self.device = p('device')
        self.axis_map = {'x': p('axis_x'), 'y': p('axis_y'), 'w': p('axis_w')}
        self.scales = {'x': float(p('scale_x')), 'y': float(p('scale_y')),
                       'w': float(p('scale_w'))}
        self.deadzone = float(p('deadzone'))
        self.arm_axis = p('arm_axis')
        self.estop_button = p('estop_button')

        self.axes = {}
        self.arm_state = None
        self.connected = False
        self.lock = threading.Lock()

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.estop_pub = self.create_publisher(Empty, 'wheels/estop', 10)
        self.enable_pub = self.create_publisher(Empty, 'wheels/enable', 10)

        threading.Thread(target=self.read_loop, daemon=True).start()
        self.create_timer(1.0 / float(p('rate_hz')), self.publish_cmd)

    def read_loop(self):
        while True:
            try:
                with open(self.device, 'rb') as js:
                    self.get_logger().info(f'joystick connected on {self.device}')
                    self.connected = True
                    while True:
                        _, value, ev_type, number = JS_EVENT.unpack(js.read(JS_EVENT.size))
                        if ev_type & EV_AXIS:
                            if number == self.arm_axis:
                                self.handle_arm(value / 32767.0)
                            else:
                                with self.lock:
                                    self.axes[number] = value / 32767.0
                        elif ev_type & EV_BUTTON and not ev_type & 0x80:
                            if number == self.estop_button and value:
                                self.get_logger().warning('estop button pressed')
                                self.estop_pub.publish(Empty())
            except (OSError, struct.error):
                self.get_logger().warning(
                    f'no joystick on {self.device}, retrying', throttle_duration_sec=10.0)
                self.connected = False
                with self.lock:
                    self.axes = {}
                time.sleep(2.0)

    def handle_arm(self, value):
        # also applied from the device's init events, so the bridge state
        # syncs to the physical switch on startup and on reconnect
        state = 'up' if value > 0.5 else 'down' if value < -0.5 else self.arm_state
        prev, self.arm_state = self.arm_state, state
        if state == prev:
            return
        if state == 'up':
            self.get_logger().info('arm switch up: enabling motors')
            self.enable_pub.publish(Empty())
        elif state == 'down':
            self.get_logger().warning('arm switch down: estop')
            self.estop_pub.publish(Empty())

    def axis(self, name):
        with self.lock:
            v = self.axes.get(self.axis_map[name], 0.0)
        return 0.0 if abs(v) < self.deadzone else v

    def publish_cmd(self):
        msg = Twist()
        msg.linear.x = self.axis('x') * self.scales['x']
        msg.linear.y = self.axis('y') * self.scales['y']
        msg.angular.z = self.axis('w') * self.scales['w']
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Teleop()
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
