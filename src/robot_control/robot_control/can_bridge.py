"""Bridge between ROS 2 wheel-velocity topics and the motor CAN bus.

Talks to the B-G431B-ESC1 boards (motor ids 0..n_motors-1) using the protocol defined in
stm32-motor-controller/PROTOCOL.md. The ESCs run their own velocity loop;
this node streams setpoints, republishes telemetry and watches the motors.

Topics:
  wheels/cmd    (Float32MultiArray, in)  wheel velocity targets, rad/s
  wheels/state  (JointState, out)        measured velocity (rad/s), Iq (A) and wheel
                                         angle (rad, continuous since the bridge started)
  wheels/estop  (Empty, in)             broadcast ESTOP, latches until enable
  wheels/enable (Empty, in)             re-enable all motors, clears estop
  /diagnostics  (DiagnosticArray, out)  once a second: one status per motor (state, fault,
                                         bus voltage, board temperature, encoder errors,
                                         error of the electrical zero) and one for the bus
                                         (battery, estop, CAN error counters)

A wheel that turns against its command, or runs beyond it, for tracking_timeout_s
has its torque reversed (stale encoder calibration, swapped phase wires): the bridge
disables all motors and says why. A wheel that only lags its command (blocked,
accelerating) never trips it.
"""
import math
import socket
import struct

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32MultiArray

MAX_MOTORS = 4   # slots in the setpoint frame
ID_ESTOP, ID_SETPOINT, ID_CMD, ID_TLM, ID_DIAG, ID_HELLO, ID_STAT = (
    0x000, 0x080, 0x100, 0x200, 0x280, 0x300, 0x480)
(OP_STOP, OP_VEL, OP_ENABLE, OP_DISABLE, OP_CURLIM,
 OP_TELEM, OP_TERM, OP_PING, OP_WATCHDOG, OP_GAIN, OP_DIAG) = range(11)
GAIN_VEL_KP, GAIN_VEL_KI = 0, 1
FLAG_ENABLED, FLAG_READY, FLAG_WD_TRIPPED, FLAG_FAULT = 1, 2, 4, 8
FLAG_LIMITED = 0x40    # firmware 10: Iq limit lowered while the windings are warm
GUARD_FW = 10          # first firmware that guards the windings; older ones get at most CONTINUOUS_A
CONTINUOUS_A = 3.0
FAULTS = ('none', 'encoder', 'overcurrent', 'undervoltage', 'overvoltage',
          'not calibrated', 'current sense', 'angle (reversed torque: calibrate)')
POS_TO_RAD = 2 * math.pi / 4096
OK, WARN, ERROR, STALE = range(4)
# a wheel whose speed error shrank by this much is on its way to the command [rad/s]
TRACKING_PROGRESS = 10.0
ANGLE_WARN_DEG, ANGLE_ERROR_DEG = 25, 45
ENCODER_ERRORS_WARN = 10   # new bad encoder frames between two status frames


def ntc_celsius(raw):
    """Diag frame NTC byte -> board temperature (10k B3425 NTC to 3.3 V over 4.7k)."""
    r = 4700.0 * (255.5 / (min(max(raw, 1), 254) + 0.5) - 1.0)
    return 1.0 / (math.log(r / 10000.0) / 3425.0 + 1.0 / 298.15) - 273.15


class CanBridge(Node):
    def __init__(self):
        super().__init__('can_bridge')
        self.declare_parameter('can_interface', 'can0')
        self.declare_parameter('n_motors', 4)
        self.declare_parameter('telemetry_period_ms', 20)
        self.declare_parameter('watchdog_ms', 500)
        self.declare_parameter('setpoint_rate_hz', 50.0)
        self.declare_parameter('cmd_timeout_s', 0.5)
        # ESC velocity-loop tuning, pushed on startup and on ESC reboot
        # (values are volatile on the ESC side); 0 = keep firmware default
        self.declare_parameter('current_limit_a', 1.3)
        self.declare_parameter('vel_pid_kp', 0.0)
        self.declare_parameter('vel_pid_ki', 0.0)
        # reversed-torque watchdog; timeout 0 = off
        self.declare_parameter('tracking_error_rad_s', 20.0)
        self.declare_parameter('tracking_timeout_s', 2.0)
        # diagnostics; a period or a limit of 0 switches that part off
        self.declare_parameter('diag_period_ms', 1000)
        self.declare_parameter('battery_warn_v', 21.0)    # 6S LiPo at 3.5 V per cell
        self.declare_parameter('battery_error_v', 19.8)   # 3.3 V per cell
        self.declare_parameter('temp_warn_c', 80.0)
        self.declare_parameter('temp_error_c', 100.0)

        self.iface = self.get_parameter('can_interface').value
        self.n_motors = min(MAX_MOTORS, int(self.get_parameter('n_motors').value))
        self.telemetry_period_ms = int(self.get_parameter('telemetry_period_ms').value)
        self.watchdog_ms = int(self.get_parameter('watchdog_ms').value)
        setpoint_rate = float(self.get_parameter('setpoint_rate_hz').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout_s').value)
        self.current_limit = float(self.get_parameter('current_limit_a').value)
        self.vel_pid_kp = float(self.get_parameter('vel_pid_kp').value)
        self.vel_pid_ki = float(self.get_parameter('vel_pid_ki').value)
        self.tracking_error = float(self.get_parameter('tracking_error_rad_s').value)
        self.tracking_timeout = float(self.get_parameter('tracking_timeout_s').value)
        self.diag_period_ms = int(self.get_parameter('diag_period_ms').value)
        self.battery_warn = float(self.get_parameter('battery_warn_v').value)
        self.battery_error = float(self.get_parameter('battery_error_v').value)
        self.temp_warn = float(self.get_parameter('temp_warn_c').value)
        self.temp_error = float(self.get_parameter('temp_error_c').value)

        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.setsockopt(
            socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
            b''.join(struct.pack('<II', base, 0x7FC)
                     for base in (ID_TLM, ID_DIAG, ID_HELLO, ID_STAT)))
        self.sock.bind((self.iface,))
        self.sock.setblocking(False)

        self.targets = [0.0] * self.n_motors
        self.sent = [0.0] * self.n_motors   # setpoints the ESCs have
        self.last_cmd_time = None
        self.estopped = False
        self.trip = ''        # why the tracking watchdog stopped the robot
        self.tracking = {}    # id -> (start, speed error then) while it runs against its command
        self.telem = {}       # id -> (vel, iq_a, flags, stamp)
        self.last_flags = {}
        self.hello = {}       # id -> (uid, firmware version)
        self.pos_raw = {}     # id -> last wheel angle word
        self.pos = {}         # id -> unwrapped wheel angle, rad
        self.diag = {}        # id -> (iq_ref, id, uq, vbus, temp, stamp)
        self.stat = {}        # id -> (parity errors, rejected frames, isr us, angle deg, ud)
        self.enc_rise = {}    # id -> new bad encoder frames in the last status period

        self.state_pub = self.create_publisher(JointState, 'wheels/state', 10)
        self.create_subscription(Float32MultiArray, 'wheels/cmd', self.cmd_callback, 10)
        self.create_subscription(Empty, 'wheels/estop', self.estop_callback, 10)
        self.create_subscription(Empty, 'wheels/enable', self.enable_callback, 10)

        self.start_time = self.get_clock().now()
        for i in range(self.n_motors):
            self.send_cmd(i, OP_PING)   # its hello tells the firmware version
            self.configure_motor(i)

        self.create_timer(0.005, self.poll_can)
        self.create_timer(1.0 / setpoint_rate, self.stream_setpoints)
        self.create_timer(self.telemetry_period_ms / 1000.0, self.publish_state)
        if self.diag_period_ms > 0:
            self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
            self.create_timer(self.diag_period_ms / 1000.0, self.publish_diagnostics)

        self.get_logger().info(f'CAN bridge up on {self.iface}')

    def seconds_since(self, stamp):
        return (self.get_clock().now() - stamp).nanoseconds / 1e9

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
        self.send_cmd(i, OP_DIAG, struct.pack('<H', self.diag_period_ms))
        self.send_current_limit(i)
        for idx, val in ((GAIN_VEL_KP, self.vel_pid_kp),
                         (GAIN_VEL_KI, self.vel_pid_ki)):
            if val > 0:
                self.send_cmd(i, OP_GAIN, struct.pack('<Bf', idx, val))
        if not self.estopped:
            self.send_cmd(i, OP_ENABLE)

    def send_current_limit(self, i):
        if self.current_limit <= 0:
            return
        limit = self.current_limit
        fw = self.hello.get(i, (None, None))[1]
        if limit > CONTINUOUS_A and (fw is None or fw < GUARD_FW):
            limit = CONTINUOUS_A
            if fw is not None:
                self.get_logger().warning(f'motor {i}: firmware {fw} does not guard the windings, '
                                          f'current limit {limit:g} A instead of {self.current_limit:g} A')
        self.send_cmd(i, OP_CURLIM, struct.pack('<f', limit))

    def send_setpoints(self, targets):
        ints = [max(-32767, min(32767, round(v * 100))) for v in targets]
        ints += [0] * (MAX_MOTORS - len(ints))
        self.send(ID_SETPOINT, struct.pack('<4h', *ints))
        self.sent = list(targets)

    # ---- subscriptions ----

    def cmd_callback(self, msg):
        vals = list(msg.data)
        if len(vals) != self.n_motors:
            self.get_logger().warning(
                f'wheels/cmd has {len(vals)} values, expected {self.n_motors}',
                throttle_duration_sec=5.0)
            vals = (vals + [0.0] * self.n_motors)[:self.n_motors]
        self.targets = [float(v) for v in vals]
        self.last_cmd_time = self.get_clock().now()
        if not self.estopped:
            self.send_setpoints(self.targets)

    def stop_all(self):
        self.estopped = True
        self.targets = [0.0] * self.n_motors
        self.tracking.clear()
        self.send(ID_ESTOP)

    def estop_callback(self, _):
        self.stop_all()
        self.get_logger().warning('ESTOP: all motors disabled (wheels/enable to clear)')

    def enable_callback(self, _):
        self.estopped = False
        self.trip = ''
        for i in range(self.n_motors):
            self.send_cmd(i, OP_ENABLE)
        self.get_logger().info('all motors enabled')

    # ---- timers ----

    def stream_setpoints(self):
        if self.estopped:
            return
        if self.last_cmd_time is not None:
            age = self.seconds_since(self.last_cmd_time)
            if age > self.cmd_timeout and any(self.targets):
                self.get_logger().warning(
                    f'no wheels/cmd for {age:.1f}s, zeroing targets')
                self.targets = [0.0] * self.n_motors
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

    def fresh(self, i):
        entry = self.telem.get(i)
        return bool(entry) and self.seconds_since(entry[3]) < 1.0

    def publish_state(self):
        if not self.telem:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for i in range(self.n_motors):
            msg.name.append(f'wheel_{i}')
            fresh = self.fresh(i)
            msg.velocity.append(self.telem[i][0] if fresh else math.nan)
            msg.effort.append(self.telem[i][1] if fresh else math.nan)
            msg.position.append(self.pos[i] if fresh and i in self.pos else math.nan)
            if not fresh:
                self.get_logger().warning(f'motor {i}: telemetry stale',
                                          throttle_duration_sec=5.0)
        self.state_pub.publish(msg)

    # ---- CAN rx ----

    def handle_frame(self, raw):
        can_id, dlc, data = struct.unpack('<IB3x8s', raw)
        can_id &= socket.CAN_EFF_MASK
        base, i = can_id & 0x7FC, can_id & 3
        if dlc < 8 or i >= self.n_motors:
            return
        if base == ID_TLM:
            vel, pos, iq, flags, fault = struct.unpack('<hHhBB', data)
            vel, iq = vel / 100.0, iq / 1000.0
            self.telem[i] = (vel, iq, flags, self.get_clock().now())
            if i in self.pos_raw:
                step = (pos - self.pos_raw[i] + 0x8000) % 0x10000 - 0x8000
                self.pos[i] += step * POS_TO_RAD
            else:
                self.pos.setdefault(i, 0.0)
            self.pos_raw[i] = pos
            self.check_flags(i, flags, fault)
            self.check_tracking(i, vel, iq, flags)
        elif base == ID_DIAG:
            iq_ref, i_d, uq, vbus, ntc = struct.unpack('<hhhBB', data)
            self.diag[i] = (iq_ref / 1000.0, i_d / 1000.0, uq / 100.0, vbus / 5.0,
                            ntc_celsius(ntc), self.get_clock().now())
        elif base == ID_STAT:
            if self.hello.get(i, (0, 0))[1] >= 9:
                parity, jumps, isr, angle, ud = struct.unpack('<HHBbh', data)
                isr, angle = isr / 4.0, None if angle == -128 else angle * 2
            else:
                parity, jumps, isr, ud = struct.unpack('<HHHh', data)
                isr, angle = isr / 100.0, None
            if i in self.stat:
                self.enc_rise[i] = (parity - self.stat[i][0] + jumps - self.stat[i][1]) % 0x10000
            self.stat[i] = (parity, jumps, isr, angle, ud / 100.0)
        elif base == ID_HELLO:
            uid, fw, fl, mid, _ = struct.unpack('<IBBBB', data)
            known = mid in self.hello
            self.hello[mid] = (uid, fw)
            self.pos_raw.pop(mid, None)   # the angle word restarts with the board
            self.stat.pop(mid, None)
            self.enc_rise.pop(mid, None)
            status = (f"current sense {'ok' if fl & 1 else 'FAIL'}, "
                      f"{'calibrated' if fl & 2 else 'NOT CALIBRATED'}, "
                      f"encoder {'ok' if fl & 4 else 'MISSING'}")
            if not known and self.seconds_since(self.start_time) < 2.0:
                # answer to our ping at startup: configured, but for the limit that depends on the firmware
                self.get_logger().info(f'motor {mid}: uid=0x{uid:08X} fw={fw}, {status}')
                self.send_current_limit(mid)
                return
            self.get_logger().warning(
                f'motor {mid} booted (uid=0x{uid:08X} fw={fw}, {status}), configuring')
            self.configure_motor(mid)

    def check_flags(self, i, flags, fault):
        prev = self.last_flags.get(i)
        self.last_flags[i] = (flags, fault)
        if (flags, fault) == prev:
            return
        prev_flags = prev[0] if prev else 0
        if fault and (prev is None or fault != prev[1]):
            name = FAULTS[fault] if fault < len(FAULTS) else fault
            self.get_logger().error(f'motor {i}: fault: {name} (wheels/enable to retry)')
        elif not flags & FLAG_READY and (prev is None or prev_flags & FLAG_READY):
            self.get_logger().error(
                f'motor {i}: not ready (calibration / encoder / current sense)')
        if flags & FLAG_WD_TRIPPED and not prev_flags & FLAG_WD_TRIPPED:
            if self.estopped:
                return
            self.get_logger().warning(f'motor {i}: watchdog tripped, re-enabling')
            self.send_cmd(i, OP_ENABLE)
        elif prev is not None and not flags & FLAG_ENABLED and (prev_flags & FLAG_ENABLED):
            self.get_logger().warning(f'motor {i}: disabled')

    def check_tracking(self, i, vel, iq, flags):
        if self.tracking_timeout <= 0 or self.estopped or not flags & FLAG_ENABLED:
            self.tracking.pop(i, None)
            return
        target = self.sent[i]
        err = abs(vel - target)
        # behind the command is a blocked or accelerating wheel; against it or
        # beyond it, the torque pushes the wrong way
        if err <= self.tracking_error or (vel * target >= 0 and abs(vel) <= abs(target)):
            self.tracking.pop(i, None)
            return
        now = self.get_clock().now()
        start = self.tracking.get(i)
        if start is None or err < start[1] - TRACKING_PROGRESS:
            self.tracking[i] = (now, err)
            return
        if (now - start[0]).nanoseconds / 1e9 < self.tracking_timeout:
            return
        self.trip = (f'motor {i} ran against its command for {self.tracking_timeout:.1f} s '
                     f'(command {target:+.1f}, measured {vel:+.1f} rad/s, Iq {iq:+.2f} A)')
        self.stop_all()
        self.get_logger().error(
            f'{self.trip}: all motors disabled. Its torque is reversed: recalibrate its '
            f'encoder, check its phase wires (wheels/enable to clear)')

    # ---- diagnostics ----

    def publish_diagnostics(self):
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        volts = []
        for i in range(self.n_motors):
            status, vbus = self.motor_status(i)
            msg.status.append(status)
            if vbus is not None:
                volts.append(vbus)
        msg.status.append(self.bus_status(volts))
        self.diag_pub.publish(msg)

    @staticmethod
    def make_status(name, hardware_id, problems, values):
        status = DiagnosticStatus()
        status.name = name
        status.hardware_id = hardware_id
        status.level = bytes([max((level for level, _ in problems), default=OK)])
        status.message = '; '.join(text for _, text in problems) or 'ok'
        status.values = [KeyValue(key=k, value=str(v)) for k, v in values]
        return status

    def motor_status(self, i):
        """-> (DiagnosticStatus, bus voltage or None)"""
        problems, values = [], []
        uid, fw = self.hello.get(i, (None, None))
        if fw is not None:
            values.append(('firmware', fw))
        vbus = None
        if not self.fresh(i):
            problems.append((STALE, 'no telemetry'))
        else:
            vel, iq, flags, _ = self.telem[i]
            fault = self.last_flags[i][1]
            if fault:
                name = FAULTS[fault] if fault < len(FAULTS) else str(fault)
                problems.append((ERROR, f'fault: {name}'))
            elif not flags & FLAG_READY:
                problems.append((ERROR, 'not ready (calibration / encoder / current sense)'))
            elif flags & FLAG_WD_TRIPPED:
                problems.append((WARN, 'watchdog tripped'))
            elif not flags & FLAG_ENABLED:
                problems.append((WARN, 'disabled'))
            elif flags & FLAG_LIMITED:
                problems.append((WARN, 'current limit lowered: windings warm'))
            values += [('enabled', bool(flags & FLAG_ENABLED)),
                       ('fault', FAULTS[fault] if fault < len(FAULTS) else fault),
                       ('velocity (rad/s)', f'{vel:.2f}'),
                       ('command (rad/s)', f'{self.sent[i]:.2f}'),
                       ('Iq (A)', f'{iq:.3f}')]
            if i in self.pos:
                values.append(('wheel angle (rad)', f'{self.pos[i]:.3f}'))
            diag = self.diag.get(i)
            if diag and self.seconds_since(diag[5]) < 3.5 * self.diag_period_ms / 1000.0:
                iq_ref, i_d, uq, vbus, temp, _ = diag
                if self.temp_error > 0 and temp >= self.temp_error:
                    problems.append((ERROR, f'board at {temp:.0f} C'))
                elif self.temp_warn > 0 and temp >= self.temp_warn:
                    problems.append((WARN, f'board at {temp:.0f} C'))
                values += [('bus voltage (V)', f'{vbus:.1f}'),
                           ('board temperature (C)', f'{temp:.0f}'),
                           ('Iq reference (A)', f'{iq_ref:.3f}'),
                           ('Id (A)', f'{i_d:.3f}'),
                           ('Uq (V)', f'{uq:.2f}')]
                stat = self.stat.get(i)
                if stat:
                    parity, jumps, isr, angle, ud = stat
                    if self.enc_rise.get(i, 0) >= ENCODER_ERRORS_WARN:
                        problems.append((WARN, f'{self.enc_rise[i]} bad encoder frames'))
                    if angle is not None and abs(angle) >= ANGLE_WARN_DEG:
                        level = ERROR if abs(angle) >= ANGLE_ERROR_DEG else WARN
                        problems.append(
                            (level, f'electrical zero off by {angle:+d} deg: calibrate'))
                    values += [('Ud (V)', f'{ud:.2f}'),
                               ('encoder parity errors', parity),
                               ('encoder rejected frames', jumps),
                               ('control interrupt max (us)', f'{isr:.2f}'),
                               ('electrical zero error (deg)',
                                'not measured' if angle is None else angle)]
            else:
                # switched off by a bench tool, or the request was lost
                self.send_cmd(i, OP_DIAG, struct.pack('<H', self.diag_period_ms))
        hardware_id = f'esc {uid:08X}' if uid is not None else ''
        return self.make_status(f'can_bridge: motor {i}', hardware_id, problems, values), vbus

    def bus_status(self, volts):
        problems, values = [], [('interface', self.iface)]
        if self.trip:
            problems.append((ERROR, f'stopped: {self.trip}'))
        elif self.estopped:
            problems.append((WARN, 'estop'))
        if volts:
            battery = sum(volts) / len(volts)
            if battery < self.battery_error:
                problems.append((ERROR, f'battery at {battery:.1f} V'))
            elif battery < self.battery_warn:
                problems.append((WARN, f'battery at {battery:.1f} V'))
            values.append(('battery (V)', f'{battery:.1f}'))
        values.append(('estop', self.estopped))
        if self.last_cmd_time is not None:
            values.append(('wheels/cmd age (s)', f'{self.seconds_since(self.last_cmd_time):.2f}'))
        for name in ('rx_errors', 'tx_errors', 'rx_over_errors'):
            try:
                with open(f'/sys/class/net/{self.iface}/statistics/{name}') as f:
                    values.append((f'can {name}', f.read().strip()))
            except OSError:
                pass
        return self.make_status('can_bridge: bus', self.iface, problems, values)

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
