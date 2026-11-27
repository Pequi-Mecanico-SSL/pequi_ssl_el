"""Robot motion control: body velocity commands to the motors, in one process.

Reads the motor CAN bus and the IMU directly and runs the whole loop at a fixed
rate: state estimate (wheels + gyro), a velocity reference that respects the
motors' current limit, and wheel speed setpoints with current feedforward from
the robot's model (drive_model.py). The ESCs keep their own velocity loops and
correct what the model misses.

Topics:
  cmd_vel        (Twist, in)              body velocity, m/s and rad/s, held constant over the ground until
                                          the next one; without a fresh one the robot brakes
  wheels/cmd     (Float32MultiArray, in)  raw wheel speeds, rad/s, for bench tests; overrides cmd_vel while it streams
  wheels/estop   (Empty, in)              all motors off, latched until wheels/enable
  wheels/enable  (Empty, in)
  wheels/state   (JointState, out)        wheel angle (rad), speed (rad/s), Iq (A)
  imu            (Imu, out)               rates (rad/s) and accelerations (m/s^2), raw
  odom           (Odometry, out)          pose since start (wheels + gyro) and body velocity
  cmd_vel_ref    (Twist, out)             the velocity being commanded to the wheels
  drive/status   (String, out)            one line per second: motors, bus voltage, temperatures, loop timing
"""
import collections
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Empty, Float32MultiArray, String

from . import esc_bus
from .drive_model import Model, Reference
from .esc_bus import EscBus
from .lsm6ds33 import Lsm6ds33


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Drive(Node):
    def __init__(self):
        super().__init__('drive')
        any_type = ParameterDescriptor(dynamic_typing=True)
        d = self.declare_parameter
        d('can_interface', 'can0')
        d('i2c_bus', 1)
        d('rate_hz', 100.0)
        d('cmd_timeout_s', 0.3)
        d('watchdog_ms', 500)
        d('diag_period_ms', 500)
        # robot model (drive_model.py); default: 4 wheels, motor 0 front-right, 1 front-left, 2 back-right, 3 back-left
        d('wheel_angles_deg', [-60.0, 60.0, -135.0, 135.0])
        d('robot_radius', 0.077, any_type)
        d('wheel_radius', 0.0245, any_type)
        d('oblique_loss', 0.0, any_type)
        d('creep', 0.0, any_type)
        d('mass', 2.0)
        d('inertia', 0.008)
        d('motor_kt', 0.0347)
        d('roller_friction', 0.0, any_type)
        d('rolling_current', 0.0, any_type)
        d('feedforward', True)
        # limits
        d('current_limit_a', 3.0)       # motor current the reference plans with
        d('current_headroom', 1.25)     # ESC Iq limit over it: room for the wheel loops' corrections
        d('max_accel', 5.0)             # m/s^2
        d('max_ang_accel', 60.0)        # rad/s^2
        d('max_wheel_rad_s', 80.0)
        # how far the reference may run ahead of the measured velocity
        d('max_lead', 0.15)             # m/s
        d('max_lead_yaw', 1.5)          # rad/s
        # heading: gain (1/s) on the difference between the expected and the gyro's turn angle;
        # expected = the commanded yaw rate as the robot answers it (setpoint delay, then this lag)
        d('heading_gain', 3.0)
        d('heading_max_error', 0.2)     # rad
        d('yaw_response_s', 0.04)
        # the wheels realise a translation this long after its setpoint: while turning, setpoints are aimed ahead by it
        d('response_s', 0.04)
        # the wheels answer a current this long after it is commanded: speed setpoints are held back by it
        d('setpoint_delay_s', 0.03)
        # ESC parameters pushed at start and when an ESC reboots, "name=value" (names of esc_bus.PARAMS)
        d('esc_params', [''])
        d('log_dir', '')

        p = lambda n: self.get_parameter(n).value
        self.dt = 1.0 / float(p('rate_hz'))
        self.cmd_timeout = float(p('cmd_timeout_s'))
        self.model = Model(p('wheel_angles_deg'), p('robot_radius'), p('wheel_radius'), p('oblique_loss'), p('creep'),
                           p('mass'), p('inertia'), p('motor_kt'), p('roller_friction'), p('rolling_current'))
        self.n = self.model.n
        self.current_limit = float(p('current_limit_a'))
        self.use_ff = bool(p('feedforward'))
        self.esc_limit = self.current_limit * float(p('current_headroom'))
        self.ref = Reference(self.model, self.current_limit,
                             p('max_accel'), p('max_ang_accel'), p('max_wheel_rad_s'),
                             p('max_lead'), p('max_lead_yaw'))
        self.heading_gain = float(p('heading_gain'))
        self.heading_max = float(p('heading_max_error'))
        self.yaw_blend = min(1.0, self.dt / max(float(p('yaw_response_s')), 1e-6))
        self.yaw_expected = 0.0
        self.ref_history = collections.deque(maxlen=1 + round(float(p('setpoint_delay_s')) / self.dt))
        self.aim_ahead = float(p('setpoint_delay_s')) + float(p('response_s'))
        self.esc_params = [(k.strip(), float(v)) for k, v in
                           (s.split('=') for s in p('esc_params') if '=' in s)]

        self.bus = EscBus(p('can_interface'))
        try:
            self.imu = Lsm6ds33(int(p('i2c_bus')))
        except OSError as e:
            self.imu = None
            self.get_logger().error(f'no IMU ({e}): yaw from the wheels only')

        self.target = np.zeros(3)
        self.cmd_time = None
        self.wheel_cmd = None
        self.wheel_cmd_time = None
        self.estopped = False
        self.start = time.monotonic()

        z = lambda: np.zeros(self.n)
        self.speed, self.iq, self.angle = z(), z(), z()        # telemetry
        self.iq_smooth = z()
        self.angle_raw = [None] * self.n
        self.angle_used = z()                                  # wheel angles at the last odometry step
        self.tlm_time = [None] * self.n
        self.flags = {}
        self.fw = {}
        self.vbus, self.temp = z(), z()

        self.gyro_bias = 0.0
        self.gyro_time = None
        self.yaw_rate = None        # bias-free gyro rate of this step, None while the IMU is silent
        self.still_time = 0.0
        self.heading_error = 0.0
        self.vel = np.zeros(3)      # estimated body velocity
        self.pose = np.zeros(3)     # x, y, heading since start
        self.setpoint, self.ff = z(), z()
        self.ticks = 0
        self.late = 0.0

        self.log = None
        if p('log_dir'):
            os.makedirs(p('log_dir'), exist_ok=True)
            self.log = open(os.path.join(p('log_dir'), 'drive.csv'), 'w')
            w = lambda name: ','.join(f'{name}{i}' for i in range(self.n))
            self.log.write('t,tx,ty,tw,rx,ry,rw,ax,ay,aw,ex,ey,ew,gz,bias,herr,px,py,pth,accx,accy,'
                           f'{w("sp")},{w("ff")},{w("v")},{w("i")},{w("p")}\n')

        # telemetry best effort: a subscriber that misses a message is not waited for (control and telemetry share the loop)
        self.state_pub = self.create_publisher(JointState, 'wheels/state', qos_profile_sensor_data)
        self.imu_pub = self.create_publisher(Imu, 'imu', qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, 'odom', qos_profile_sensor_data)
        self.ref_pub = self.create_publisher(Twist, 'cmd_vel_ref', qos_profile_sensor_data)
        self.status_pub = self.create_publisher(String, 'drive/status', qos_profile_sensor_data)
        # best effort: a command lost on the Wi-Fi is dropped, never delivered late
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, 'wheels/cmd', self.on_wheel_cmd, 10)
        self.create_subscription(Empty, 'wheels/estop', self.on_estop, 10)
        self.create_subscription(Empty, 'wheels/enable', self.on_enable, 10)

        for i in range(self.n):
            self.bus.command(i, esc_bus.OP_PING)     # its hello tells the firmware version
            self.configure_motor(i)
        self.get_logger().info(f'drive up: {self.n} wheels, {1 / self.dt:.0f} Hz, '
                               f'feedforward {"on" if self.use_ff else "off"}')

    # ---- motors ----

    def configure_motor(self, i):
        b = self.bus
        b.command(i, esc_bus.OP_TELEM, int(round(self.dt * 1000)).to_bytes(2, 'little'))
        b.command(i, esc_bus.OP_WATCHDOG, int(self.get_parameter('watchdog_ms').value).to_bytes(2, 'little'))
        b.command(i, esc_bus.OP_DIAG, int(self.get_parameter('diag_period_ms').value).to_bytes(2, 'little'))
        b.set_param(i, 'iq_limit', self.esc_limit)
        for name, value in self.esc_params:
            b.set_param(i, name, value)
        if not self.estopped:
            b.command(i, esc_bus.OP_ENABLE)

    def read_bus(self, now):
        for f in self.bus.poll():
            kind, i = f[0], f[1]
            if i >= self.n:
                continue
            if kind == 'tlm':
                _, _, speed, raw, iq, flags, fault = f
                self.speed[i], self.iq[i] = speed, iq
                if self.angle_raw[i] is not None:
                    self.angle[i] += ((raw - self.angle_raw[i] + 0x8000) % 0x10000 - 0x8000) * esc_bus.POS_TO_RAD
                self.angle_raw[i] = raw
                self.tlm_time[i] = now
                self.check_flags(i, flags, fault)
            elif kind == 'diag':
                self.vbus[i], self.temp[i] = f[2], f[3]
            elif kind == 'hello':
                _, _, uid, fw, fl = f
                known = i in self.fw
                self.fw[i] = fw
                self.angle_raw[i] = None
                status = (f"current sense {'ok' if fl & 1 else 'FAIL'}, "
                          f"{'calibrated' if fl & 2 else 'NOT CALIBRATED'}, "
                          f"encoder {'ok' if fl & 4 else 'MISSING'}")
                if not known and now - self.start < 2.0:
                    # answer to the ping at start: already configured
                    self.get_logger().info(f'motor {i}: uid=0x{uid:08X} fw={fw}, {status}')
                else:
                    self.get_logger().warning(f'motor {i} booted (uid=0x{uid:08X} fw={fw}, {status}), configuring')
                    self.configure_motor(i)

    def check_flags(self, i, flags, fault):
        prev = self.flags.get(i)
        self.flags[i] = (flags, fault)
        if (flags, fault) == prev:
            return
        prev_flags = prev[0] if prev else 0
        if fault and (prev is None or fault != prev[1]):
            name = esc_bus.FAULTS[fault] if fault < len(esc_bus.FAULTS) else fault
            self.get_logger().error(f'motor {i}: fault: {name} (wheels/enable to retry)')
        elif not flags & esc_bus.FLAG_READY and (prev is None or prev_flags & esc_bus.FLAG_READY):
            self.get_logger().error(f'motor {i}: not ready (calibration / encoder / current sense)')
        if flags & esc_bus.FLAG_WD_TRIPPED and not prev_flags & esc_bus.FLAG_WD_TRIPPED:
            if not self.estopped:
                self.get_logger().warning(f'motor {i}: watchdog tripped, re-enabling')
                self.bus.command(i, esc_bus.OP_ENABLE)
        elif prev is not None and not flags & esc_bus.FLAG_ENABLED and prev_flags & esc_bus.FLAG_ENABLED:
            self.get_logger().warning(f'motor {i}: disabled')

    # ---- subscriptions ----

    def on_cmd_vel(self, msg):
        target = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        if not np.all(np.isfinite(target)):
            self.get_logger().error(f'cmd_vel not finite, ignored: {target}', throttle_duration_sec=1.0)
            return
        self.target = target
        self.cmd_time = time.monotonic()

    def on_wheel_cmd(self, msg):
        vals = np.array((list(msg.data) + [0.0] * self.n)[:self.n], float)
        if not np.all(np.isfinite(vals)):
            self.get_logger().error(f'wheels/cmd not finite, ignored: {vals}', throttle_duration_sec=1.0)
            return
        self.wheel_cmd = vals
        self.wheel_cmd_time = time.monotonic()

    def on_estop(self, _):
        self.estopped = True
        self.bus.estop()
        self.reset_motion()
        self.get_logger().warning('ESTOP: all motors disabled (wheels/enable to clear)')

    def on_enable(self, _):
        self.estopped = False
        self.reset_motion()
        for i in range(self.n):
            self.bus.command(i, esc_bus.OP_ENABLE)
        self.get_logger().info('all motors enabled')

    def reset_motion(self):
        self.ref.reset()
        self.ref_history.clear()
        self.heading_error = 0.0
        self.yaw_expected = 0.0
        self.target = np.zeros(3)

    # ---- control step ----

    def step(self):
        now = time.monotonic()
        dt = self.dt
        self.read_bus(now)
        fresh = [t is not None and now - t < 0.2 for t in self.tlm_time]
        wheels_ok = all(fresh)

        # gyro: mean rate since the last step, bias learned while standing still
        sample = self.imu.take() if self.imu else None
        gyro = accel = None
        if sample:
            gyro, accel = sample
            self.gyro_time = now
        gyro_ok = self.gyro_time is not None and now - self.gyro_time < 0.1
        moving = bool(np.any(np.abs(self.speed) > 0.2)) or bool(np.any(self.ref.v))
        self.still_time = 0.0 if moving else self.still_time + dt
        if gyro is not None and self.still_time > 1.0 and abs(gyro[2] - self.gyro_bias) < 0.15:
            self.gyro_bias += (dt / 2.0) * (gyro[2] - self.gyro_bias)
        if gyro is not None:
            self.yaw_rate = gyro[2] - self.gyro_bias
        elif not gyro_ok:
            self.yaw_rate = None

        # state estimate: translation from the wheels, yaw from the gyro; pose from the wheel angle steps
        self.iq_smooth += 0.3 * (self.iq - self.iq_smooth)
        if wheels_ok:
            self.vel = self.model.body_velocity(self.speed, self.yaw_rate, self.iq_smooth)
            turn = self.vel[2] * dt
            d = self.model.body_velocity(self.angle - self.angle_used, turn, self.iq_smooth)
            self.angle_used = self.angle.copy()
            mid = self.pose[2] + 0.5 * turn
            c, s = math.cos(mid), math.sin(mid)
            self.pose += (c * d[0] - s * d[1], s * d[0] + c * d[1], turn)

        # command
        raw = self.wheel_cmd_time is not None and now - self.wheel_cmd_time < self.cmd_timeout
        cmd_fresh = self.cmd_time is not None and now - self.cmd_time < self.cmd_timeout
        if self.yaw_rate is not None:
            # between messages the commanded velocity stays constant over the ground
            a = self.yaw_rate * dt
            c, s = math.cos(a), math.sin(a)
            self.target[0], self.target[1] = (c * self.target[0] + s * self.target[1],
                                              -s * self.target[0] + c * self.target[1])
        target = self.target if cmd_fresh and not raw else np.zeros(3)
        acc = np.zeros(3)
        if self.estopped:
            self.ref.reset()
            self.setpoint[:] = 0.0
            self.ff[:] = 0.0
        elif raw:
            self.reset_motion()
            self.setpoint = self.wheel_cmd.copy()
            self.ff[:] = 0.0
        else:
            v, acc = self.ref.step(target, dt, self.yaw_rate, self.vel if wheels_ok else None)
            self.ref_history.append(v.copy())
            delayed = self.ref_history[0]
            # heading: the expected turn angle is followed with the gyro
            self.yaw_expected += self.yaw_blend * (delayed[2] - self.yaw_expected)
            if self.yaw_rate is not None and (np.any(v) or np.any(target) or abs(self.yaw_expected) > 0.05):
                self.heading_error = float(np.clip(self.heading_error + (self.yaw_expected - self.yaw_rate) * dt,
                                                   -self.heading_max, self.heading_max))
            else:
                self.heading_error = 0.0
            out = delayed + (0.0, 0.0, self.heading_gain * self.heading_error)
            if self.yaw_rate is not None:
                # same velocity over the ground at the time the wheels deliver it
                a = self.yaw_rate * self.aim_ahead
                c, s = math.cos(a), math.sin(a)
                out[0], out[1] = c * out[0] + s * out[1], -s * out[0] + c * out[1]
            self.ff = self.ref.current if self.use_ff else np.zeros(self.n)
            self.setpoint = self.model.wheel_speeds(out, self.ff)

        if not self.estopped:
            self.bus.setpoints(self.setpoint)
            self.bus.feedforward(self.ff)

        self.ticks += 1
        self.publish(gyro, accel, wheels_ok, fresh)
        if self.log:
            f = lambda a: ','.join(f'{x:.4f}' for x in a)
            g = gyro[2] if gyro else float('nan')
            ac = accel[:2] if accel else (float('nan'),) * 2
            self.log.write(f'{time.time():.4f},{f(target)},{f(self.ref.v)},{f(acc)},{f(self.vel)},{g:.4f},'
                           f'{self.gyro_bias:.4f},{self.heading_error:.4f},{f(self.pose)},{f(ac)},'
                           f'{f(self.setpoint)},{f(self.ff)},{f(self.speed)},{f(self.iq)},{f(self.angle)}\n')

    def publish(self, gyro, accel, wheels_ok, fresh):
        stamp = self.get_clock().now().to_msg()
        if gyro is not None:
            m = Imu()
            m.header.stamp, m.header.frame_id = stamp, 'imu'
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = gyro
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = accel
            self.imu_pub.publish(m)
        if self.ticks % 2 == 0:
            m = JointState()
            m.header.stamp = stamp
            m.name = [f'wheel_{i}' for i in range(self.n)]
            nan = math.nan
            m.position = [float(a) if ok else nan for a, ok in zip(self.angle, fresh)]
            m.velocity = [float(a) if ok else nan for a, ok in zip(self.speed, fresh)]
            m.effort = [float(a) if ok else nan for a, ok in zip(self.iq, fresh)]
            self.state_pub.publish(m)

            o = Odometry()
            o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, 'odom', 'base_link'
            o.pose.pose.position.x, o.pose.pose.position.y = float(self.pose[0]), float(self.pose[1])
            o.pose.pose.orientation.z = math.sin(self.pose[2] / 2)
            o.pose.pose.orientation.w = math.cos(self.pose[2] / 2)
            o.twist.twist.linear.x, o.twist.twist.linear.y = float(self.vel[0]), float(self.vel[1])
            o.twist.twist.angular.z = float(self.vel[2])
            self.odom_pub.publish(o)

            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = (float(x) for x in self.ref.v)
            self.ref_pub.publish(t)
        if self.ticks % round(1.0 / self.dt) == 0:
            state = 'ESTOP' if self.estopped else 'on'
            stale = [i for i, ok in enumerate(fresh) if not ok]
            line = (f'{state} vbus {self.vbus.max():.1f} V temp ' + ' '.join(f'{t:.0f}' for t in self.temp)
                    + ' C iq ' + ' '.join(f'{a:+.2f}' for a in self.iq)
                    + f' gyro {"ok" if self.yaw_rate is not None else "MISSING"} bias {self.gyro_bias:+.4f}'
                    + f' late {self.late * 1000:.1f} ms'
                    + (f' NO TELEMETRY from {stale}' if stale else ''))
            self.status_pub.publish(String(data=line))
            if stale or self.yaw_rate is None:
                self.get_logger().warning(line, throttle_duration_sec=5.0)
            self.late = 0.0

    def shutdown(self):
        self.bus.estop()
        self.bus.close()
        if self.imu:
            self.imu.close()
        if self.log:
            self.log.close()


def main(args=None):
    rclpy.init(args=args)
    node = Drive()
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(10))
    except (OSError, AttributeError):
        pass
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    next_t = time.monotonic()
    try:
        while rclpy.ok():
            node.step()
            next_t += node.dt
            while True:
                remaining = next_t - time.monotonic()
                if remaining <= 0:
                    break
                executor.spin_once(timeout_sec=remaining)
            node.late = max(node.late, -remaining)
            if remaining < -5 * node.dt:
                next_t = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
