"""Bridge between ROS 2 wheel-velocity topics and the motor CAN bus.

Talks to the four B-G431B-ESC1 boards using the protocol defined in
stm32-motor-controller/PROTOCOL.md. The ESCs run their own velocity PID;
this node only streams setpoints and republishes telemetry.

Topics:
  wheels/cmd    (Float32MultiArray, in)  4x wheel velocity target, rad/s
  wheels/state  (JointState, out)        measured velocity (rad/s) + Iq (A)
  wheels/estop  (Empty, in)             broadcast ESTOP, latches until enable
  wheels/enable (Empty, in)             re-enable all motors, clears estop
"""
import math
import socket
import struct

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32MultiArray

N_MOTORS = 4
ID_ESTOP, ID_SETPOINT, ID_CMD, ID_TLM, ID_HELLO = 0x000, 0x080, 0x100, 0x200, 0x300
(OP_STOP, OP_VEL, OP_ENABLE, OP_DISABLE, OP_CURLIM,
 OP_TELEM, OP_TERM, OP_PING, OP_WATCHDOG) = range(9)
FLAG_ENABLED, FLAG_FOC_OK, FLAG_WD_TRIPPED = 1, 2, 4


class CanBridge(Node):
    def __init__(self):
        super().__init__('can_bridge')
        self.declare_parameter('can_interface', 'can0')
        self.declare_parameter('telemetry_period_ms', 20)
        self.declare_parameter('watchdog_ms', 500)
        self.declare_parameter('setpoint_rate_hz', 50.0)
        self.declare_parameter('cmd_timeout_s', 0.5)
        self.declare_parameter('current_limit_a', 0.0)  # 0 = firmware default

        iface = self.get_parameter('can_interface').value
        self.telemetry_period_ms = int(self.get_parameter('telemetry_period_ms').value)
        self.watchdog_ms = int(self.get_parameter('watchdog_ms').value)
        setpoint_rate = float(self.get_parameter('setpoint_rate_hz').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout_s').value)
        self.current_limit = float(self.get_parameter('current_limit_a').value)

        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        # only telemetry and hello frames reach us
        self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                             struct.pack('<II', ID_TLM, 0x7FC)
                             + struct.pack('<II', ID_HELLO, 0x7FC))
        self.sock.bind((iface,))
        self.sock.setblocking(False)

        self.targets = [0.0] * N_MOTORS
        self.last_cmd_time = None
        self.estopped = False
        self.telem = {}       # id -> (vel, iq_a, flags, stamp)
        self.last_flags = {}

        self.state_pub = self.create_publisher(JointState, 'wheels/state', 10)
        self.create_subscription(Float32MultiArray, 'wheels/cmd', self.cmd_callback, 10)
        self.create_subscription(Empty, 'wheels/estop', self.estop_callback, 10)
        self.create_subscription(Empty, 'wheels/enable', self.enable_callback, 10)

        for i in range(N_MOTORS):
            self.configure_motor(i)

        self.create_timer(0.005, self.poll_can)
        self.create_timer(1.0 / setpoint_rate, self.stream_setpoints)
        self.create_timer(self.telemetry_period_ms / 1000.0, self.publish_state)

        self.get_logger().info(f'CAN bridge up on {iface}')

    # ---- CAN tx ----

    def send(self, can_id, data=b''):
        try:
            self.sock.send(struct.pack('<IB3x8s', can_id, len(data), data.ljust(8, b'\0')))
        except OSError as e:
            self.get_logger().error(f'CAN send failed: {e}',
                                    throttle_duration_sec=1.0)

    def send_cmd(self, motor_id, op, args=b''):
        self.send(ID_CMD + motor_id, bytes([op]) + args)

    def configure_motor(self, i):
        self.send_cmd(i, OP_TELEM, struct.pack('<H', self.telemetry_period_ms))
        self.send_cmd(i, OP_WATCHDOG, struct.pack('<H', self.watchdog_ms))
        if self.current_limit > 0:
            self.send_cmd(i, OP_CURLIM, struct.pack('<f', self.current_limit))
        if not self.estopped:
            self.send_cmd(i, OP_ENABLE)

    def send_setpoints(self, targets):
        ints = [max(-32767, min(32767, round(v * 100))) for v in targets]
        self.send(ID_SETPOINT, struct.pack('<4h', *ints))

    # ---- subscriptions ----

    def cmd_callback(self, msg):
        vals = list(msg.data)
        if len(vals) != N_MOTORS:
            self.get_logger().warning(
                f'wheels/cmd has {len(vals)} values, expected {N_MOTORS}',
                throttle_duration_sec=5.0)
            vals = (vals + [0.0] * N_MOTORS)[:N_MOTORS]
        self.targets = [float(v) for v in vals]
        self.last_cmd_time = self.get_clock().now()
        if not self.estopped:
            self.send_setpoints(self.targets)

    def estop_callback(self, _):
        self.estopped = True
        self.targets = [0.0] * N_MOTORS
        self.send(ID_ESTOP)
        self.get_logger().warning('ESTOP: all motors disabled (wheels/enable to clear)')

    def enable_callback(self, _):
        self.estopped = False
        for i in range(N_MOTORS):
            self.send_cmd(i, OP_ENABLE)
        self.get_logger().info('all motors enabled')

    # ---- timers ----

    def stream_setpoints(self):
        if self.estopped:
            return
        if self.last_cmd_time is not None:
            age = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
            if age > self.cmd_timeout and any(self.targets):
                self.get_logger().warning(
                    f'no wheels/cmd for {age:.1f}s, zeroing targets')
                self.targets = [0.0] * N_MOTORS
        self.send_setpoints(self.targets)

    def poll_can(self):
        while True:
            try:
                raw = self.sock.recv(16)
            except BlockingIOError:
                return
            except OSError as e:
                self.get_logger().error(f'CAN recv failed: {e}',
                                        throttle_duration_sec=1.0)
                return
            self.handle_frame(raw)

    def publish_state(self):
        if not self.telem:
            return
        now = self.get_clock().now()
        msg = JointState()
        msg.header.stamp = now.to_msg()
        for i in range(N_MOTORS):
            msg.name.append(f'wheel_{i}')
            entry = self.telem.get(i)
            fresh = entry and (now - entry[3]).nanoseconds / 1e9 < 1.0
            msg.velocity.append(entry[0] if fresh else math.nan)
            msg.effort.append(entry[1] if fresh else math.nan)
            if not fresh:
                self.get_logger().warning(f'motor {i}: telemetry stale',
                                          throttle_duration_sec=5.0)
        self.state_pub.publish(msg)

    # ---- CAN rx ----

    def handle_frame(self, raw):
        can_id, dlc, data = struct.unpack('<IB3x8s', raw)
        can_id &= socket.CAN_EFF_MASK
        data = data[:dlc]
        if ID_TLM <= can_id < ID_TLM + N_MOTORS and dlc >= 8:
            i = can_id - ID_TLM
            vel, _tgt, iq, flags, _uq = struct.unpack('<hhhBb', data)
            self.telem[i] = (vel / 100.0, iq / 1000.0, flags, self.get_clock().now())
            self.check_flags(i, flags)
        elif ID_HELLO <= can_id < ID_HELLO + N_MOTORS and dlc >= 8:
            uid, fw, fl, mid, _ = struct.unpack('<IBBBB', data)
            self.get_logger().warning(
                f'motor {mid} rebooted (uid=0x{uid:08X} fw={fw} '
                f"cs={'ok' if fl & 1 else 'FAIL'} foc={'ok' if fl & 2 else 'FAIL'}), "
                'reconfiguring')
            self.configure_motor(mid)

    def check_flags(self, i, flags):
        prev = self.last_flags.get(i)
        self.last_flags[i] = flags
        if flags == prev:
            return
        if not flags & FLAG_FOC_OK:
            self.get_logger().error(f'motor {i}: initFOC failed')
        if flags & FLAG_WD_TRIPPED and not (prev or 0) & FLAG_WD_TRIPPED:
            if self.estopped:
                return
            self.get_logger().warning(f'motor {i}: watchdog tripped, re-enabling')
            self.send_cmd(i, OP_ENABLE)
        elif prev is not None and not flags & FLAG_ENABLED and (prev & FLAG_ENABLED):
            self.get_logger().warning(f'motor {i}: disabled')

    def shutdown(self):
        self.send(ID_ESTOP)
        self.sock.close()


def main(args=None):
    rclpy.init(args=args)
    bridge = CanBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.shutdown()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
