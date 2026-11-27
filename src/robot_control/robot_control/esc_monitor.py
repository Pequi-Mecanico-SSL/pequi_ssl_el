"""Health of the ESCs, for the node that drives them: the content of /diagnostics and the stop
of a motor whose torque is reversed (stale encoder calibration, swapped phase wires)."""
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue

from . import esc_bus

OK, WARN, ERROR, STALE = range(4)
# a wheel whose speed error shrank by this much is on its way to the command [rad/s]
TRACKING_PROGRESS = 10.0
ANGLE_WARN_DEG, ANGLE_ERROR_DEG = 25, 45
ENCODER_ERRORS_WARN = 10   # new bad encoder frames between two status frames
STATUS_FW = 9              # first firmware whose status frame carries the electrical zero error


def make_status(name, hardware_id, problems, values):
    status = DiagnosticStatus()
    status.name = name
    status.hardware_id = hardware_id
    status.level = bytes([max((level for level, _ in problems), default=OK)])
    status.message = '; '.join(text for _, text in problems) or 'ok'
    status.values = [KeyValue(key=k, value=str(v)) for k, v in values]
    return status


class EscMonitor:
    """Fed with the frames of esc_bus.poll(); times are seconds of one monotonic clock."""

    def __init__(self, n, name, tracking_error, tracking_timeout, battery_warn, battery_error,
                 temp_warn, temp_error, diag_period):
        self.n, self.name = n, name
        self.tracking_error, self.tracking_timeout = tracking_error, tracking_timeout
        self.battery_warn, self.battery_error = battery_warn, battery_error
        self.temp_warn, self.temp_error = temp_warn, temp_error
        self.diag_period = diag_period
        self.telem = {}       # motor -> (speed, iq, flags, fault, time)
        self.diag = {}        # motor -> (bus V, board C, iq reference, id, uq, time)
        self.stat = {}        # motor -> (parity errors, rejected frames, interrupt us, zero error deg, ud)
        self.enc_rise = {}    # motor -> new bad encoder frames in the last status period
        self.hello = {}       # motor -> (uid, firmware)
        self.tracking = {}    # motor -> (time, speed error) since it runs against its command
        self.trip = ''        # why the monitor asked for a stop

    # ---- frames ----

    def on_tlm(self, i, speed, iq, flags, fault, now):
        self.telem[i] = (speed, iq, flags, fault, now)

    def on_diag(self, i, vbus, temp, iq_ref, i_d, uq, now):
        self.diag[i] = (vbus, temp, iq_ref, i_d, uq, now)

    def on_stat(self, i, parity, jumps, isr, angle, ud):
        if self.hello.get(i, (0, 0))[1] < STATUS_FW:
            return
        if i in self.stat:
            self.enc_rise[i] = (parity - self.stat[i][0] + jumps - self.stat[i][1]) % 0x10000
        self.stat[i] = (parity, jumps, isr, angle, ud)

    def on_hello(self, i, uid, fw):
        self.hello[i] = (uid, fw)
        self.stat.pop(i, None)      # the counters restart with the board
        self.enc_rise.pop(i, None)

    def firmware(self, i):
        return self.hello.get(i, (None, None))[1]

    def battery(self, now):
        """Mean bus voltage over the ESCs that reported lately, None without one."""
        volts = [d[0] for d in self.diag.values() if now - d[5] < 3.5 * self.diag_period]
        return sum(volts) / len(volts) if volts else None

    # ---- reversed torque ----

    def check_tracking(self, i, target, driving, now):
        """True when motor i has run against or beyond its speed command for tracking_timeout:
        its torque pushes the wrong way. Behind the command is only a blocked or accelerating wheel."""
        if self.tracking_timeout <= 0 or not driving or i not in self.telem:
            self.tracking.pop(i, None)
            return False
        speed, iq, flags, _, _ = self.telem[i]
        err = abs(speed - target)
        if (not flags & esc_bus.FLAG_ENABLED or err <= self.tracking_error
                or (speed * target >= 0 and abs(speed) <= abs(target))):
            self.tracking.pop(i, None)
            return False
        start = self.tracking.get(i)
        if start is None or err < start[1] - TRACKING_PROGRESS:
            self.tracking[i] = (now, err)
            return False
        if now - start[0] < self.tracking_timeout:
            return False
        self.trip = (f'motor {i} ran against its command for {self.tracking_timeout:.1f} s '
                     f'(command {target:+.1f}, measured {speed:+.1f} rad/s, Iq {iq:+.2f} A)')
        self.tracking.clear()
        return True

    def clear(self):
        self.trip = ''
        self.tracking.clear()

    # ---- /diagnostics ----

    def motor_status(self, i, sent, now):
        problems, values = [], []
        uid, fw = self.hello.get(i, (None, None))
        if fw is not None:
            values.append(('firmware', fw))
        entry = self.telem.get(i)
        if not entry or now - entry[4] > 1.0:
            problems.append((STALE, 'no telemetry'))
        else:
            speed, iq, flags, fault, _ = entry
            fault_name = esc_bus.FAULTS[fault] if fault < len(esc_bus.FAULTS) else str(fault)
            if fault:
                problems.append((ERROR, f'fault: {fault_name}'))
            elif not flags & esc_bus.FLAG_READY:
                problems.append((ERROR, 'not ready (calibration / encoder / current sense)'))
            elif flags & esc_bus.FLAG_WD_TRIPPED:
                problems.append((WARN, 'watchdog tripped'))
            elif not flags & esc_bus.FLAG_ENABLED:
                problems.append((WARN, 'disabled'))
            elif flags & esc_bus.FLAG_LIMITED:
                problems.append((WARN, 'current limit lowered: windings warm'))
            values += [('enabled', bool(flags & esc_bus.FLAG_ENABLED)),
                       ('fault', fault_name),
                       ('velocity (rad/s)', f'{speed:.2f}'),
                       ('command (rad/s)', f'{sent:.2f}'),
                       ('Iq (A)', f'{iq:.3f}')]
            diag = self.diag.get(i)
            if diag and now - diag[5] < 3.5 * self.diag_period:
                vbus, temp, iq_ref, i_d, uq, _ = diag
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
                        problems.append((level, f'electrical zero off by {angle:+d} deg: calibrate'))
                    values += [('Ud (V)', f'{ud:.2f}'),
                               ('encoder parity errors', parity),
                               ('encoder rejected frames', jumps),
                               ('control interrupt max (us)', f'{isr:.2f}'),
                               ('electrical zero error (deg)', 'not measured' if angle is None else angle)]
        hardware_id = f'esc {uid:08X}' if uid is not None else ''
        return make_status(f'{self.name}: motor {i}', hardware_id, problems, values)

    def bus_status(self, interface, estopped, now, extra=()):
        problems, values = [], [('interface', interface)]
        if self.trip:
            problems.append((ERROR, f'stopped: {self.trip}'))
        elif estopped:
            problems.append((WARN, 'estop'))
        battery = self.battery(now)
        if battery is not None:
            if battery < self.battery_error:
                problems.append((ERROR, f'battery at {battery:.1f} V'))
            elif battery < self.battery_warn:
                problems.append((WARN, f'battery at {battery:.1f} V'))
            values.append(('battery (V)', f'{battery:.1f}'))
        values.append(('estop', estopped))
        values += list(extra)
        for name in ('rx_errors', 'tx_errors', 'rx_over_errors'):
            try:
                with open(f'/sys/class/net/{interface}/statistics/{name}') as f:
                    values.append((f'can {name}', f.read().strip()))
            except OSError:
                pass
        return make_status(f'{self.name}: bus', interface, problems, values)

    def stale_diag(self, now):
        """Motors whose diag frames stopped (switched off by a bench tool, or the request was lost)."""
        return [i for i in range(self.n)
                if i in self.telem and now - self.telem[i][4] < 1.0
                and (i not in self.diag or now - self.diag[i][5] >= 3.5 * self.diag_period)]
