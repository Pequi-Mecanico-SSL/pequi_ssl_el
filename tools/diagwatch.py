"""Live view of /diagnostics in the terminal: diagwatch.py [domain ...]

Subscribes on every ROS domain given (default 77 and 0) and redraws in place.
can_bridge's statuses get a motor table and a bus line; any other status is
listed with its values. Run it through tools/diagwatch.
"""
import sys
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

LEVELS = (('OK', '32'), ('WARN', '33'), ('ERROR', '31'), ('STALE', '35'))
STALE_AFTER = 3.0   # s without a message before a domain's data counts as old


def paint(text, code):
    return f'\x1b[{code}m{text}\x1b[0m'


def level(status):
    name, code = LEVELS[status.level[0]] if status.level[0] < len(LEVELS) else ('?', '35')
    return paint(f'{name:<5}', code + ';1')


def motor_row(index, status):
    v = {kv.key: kv.value for kv in status.values}

    def get(key, width, unit=''):
        text = v.get(key, '')
        return f'{text + unit if text else "-":>{width}}'

    zero = v.get('electrical zero error (deg)', '')
    zero = '-' if zero in ('', 'not measured') else f'{int(zero):+d}°'
    enc = f"{v.get('encoder parity errors', '-')}/{v.get('encoder rejected frames', '-')}"
    state = {'True': 'on', 'False': 'off'}.get(v.get('enabled'), '-')
    return (f'  {index:<5} {level(status)} {state:<5}'
            f"{get('velocity (rad/s)', 8)}{get('command (rad/s)', 8)}{get('Iq (A)', 8)}"
            f"{get('bus voltage (V)', 7)}{get('board temperature (C)', 6, '°C')}"
            f'{zero:>7}{enc:>9}  fw {v.get("firmware", "-"):<3} {status.message}')


def render(domain, statuses, age):
    old = age > STALE_AFTER
    head = f'domain {domain}  /diagnostics  ' + (
        paint(f'no message for {age:.0f} s', '31;1') if old else f'{age:.1f} s ago')
    lines = [paint(head, '1'), '']
    motors = sorted((int(name.rsplit(' ', 1)[1]), s) for name, s in statuses.items()
                    if name.startswith('can_bridge: motor '))
    bus = statuses.get('can_bridge: bus')
    if bus:
        v = {kv.key: kv.value for kv in bus.values}
        cmd = f"{v['wheels/cmd age (s)']} s ago" if 'wheels/cmd age (s)' in v else 'none yet'
        can = '/'.join(v.get(f'can {k}', '-') for k in ('rx_errors', 'tx_errors', 'rx_over_errors'))
        lines.append(f"  bus   {level(bus)} battery {v.get('battery (V)', '-')} V   "
                     f"estop {'ON' if v.get('estop') == 'True' else 'off'}   "
                     f"wheels/cmd {cmd}   "
                     f"CAN rx/tx/overrun errors {can}   {bus.message}")
    if motors:
        lines.append(paint('  motor level state   rad/s     cmd    Iq A  bus V board   zero'
                           '  enc err', '2'))
        lines += [motor_row(i, s) for i, s in motors]
    for name, s in sorted(statuses.items()):
        if name.startswith('can_bridge: '):
            continue
        lines.append(f'  {level(s)} {name}: {s.message}')
        if s.values:
            lines.append('        ' + ', '.join(f'{kv.key}={kv.value}' for kv in s.values))
    return lines


class Watch:
    def __init__(self, domain):
        self.domain = domain
        self.context = Context()
        rclpy.init(context=self.context, domain_id=domain)
        self.node = Node(f'diagwatch_{domain}', context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.statuses = {}
        self.stamp = None
        self.node.create_subscription(DiagnosticArray, '/diagnostics', self.update, 10)

    def update(self, msg):
        self.stamp = time.monotonic()
        for status in msg.status:
            self.statuses[status.name] = status


def main():
    domains = [int(a) for a in sys.argv[1:]] or [77, 0]
    watches = [Watch(d) for d in domains]
    sys.stdout.write('\x1b[?25l\x1b[2J')   # hide the cursor, clear
    try:
        while True:
            for w in watches:
                w.executor.spin_once(timeout_sec=0.25 / len(watches))
            lines = []
            for w in watches:
                if w.stamp is not None:
                    lines += render(w.domain, w.statuses, time.monotonic() - w.stamp) + ['']
            if not lines:
                lines = [f"waiting for /diagnostics on domain {', '.join(map(str, domains))}"]
            sys.stdout.write('\x1b[H' + ''.join(line + '\x1b[K\n' for line in lines) + '\x1b[J')
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write('\x1b[?25h\n')
        for w in watches:
            rclpy.try_shutdown(context=w.context)


if __name__ == '__main__':
    main()
