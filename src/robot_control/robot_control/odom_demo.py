"""Offline demo: gyro + wheel fusion vs the current methods, on a synthetic
robot with known truth (no ROS, no simulator, no hardware).

    ros2 run robot_control odom_demo [--plot odom_demo.png] [--seed 0]

Compared on the same sensor data (see sim_sensors.py for the scenario):
    current      velocity_controller.py today: gyro bias learned only while
                 idle, heading = integrated (gyro - bias)
    wheels only  body velocity = pseudo-inverse of the wheel matrix
                 (what robot_pid.py does)
    fused        OdomEKF
"""
import argparse
import math

import numpy as np

from .odom_ekf import OdomEKF
from .sim_sensors import (IDLE, SIGMA_GYRO, SIGMA_WHEEL, SLIPS, STUCK, SyntheticRun,
                          gyro_bias, in_window)


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


class CurrentMethod:
    """velocity_controller.py's bias/heading logic (default parameters)."""

    def __init__(self, bias_tau=10.0, bias_max_dev=0.1, idle_time=1.0):
        self.bias_tau, self.bias_max_dev, self.idle_time = bias_tau, bias_max_dev, idle_time
        self.bias = 0.0
        self.heading = 0.0
        self.omega = 0.0
        self.last_t = None
        self.last_active = None

    def command(self, t, cmd):
        if abs(cmd[0]) > 0.01 or abs(cmd[1]) > 0.01 or abs(cmd[2]) > 0.05:
            self.last_active = t

    def imu(self, t, raw):
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        if not 0.0 < dt < 0.05:
            return
        idle = self.last_active is None or t - self.last_active > self.idle_time
        if idle and abs(raw - self.bias) < self.bias_max_dev:
            self.bias += (dt / self.bias_tau) * (raw - self.bias)
        self.omega = raw - self.bias
        self.heading = float(wrap(self.heading + self.omega * dt))


def run(seed=0, **ekf_kwargs):
    sim = SyntheticRun(seed=seed)
    ekf = OdomEKF(sim.J, sigma_gyro=SIGMA_GYRO, sigma_wheel=SIGMA_WHEEL, **ekf_kwargs)
    cur = CurrentMethod()
    J_pinv = np.linalg.pinv(sim.J)
    wheel_vel = np.zeros(3)
    wheel_heading = 0.0
    last_wheel_t = 0.0

    log = {k: [] for k in ('t', 'true', 'fused', 'wheels', 'cur_w', 'cur_th',
                           'wheel_th', 'bias', 'cur_bias_log', 'true_bias', 'used')}
    for kind, t, z in sim.events():
        k = sim.index(t)
        if kind == 'imu':
            cur.command(t, sim.cmd[k])
            cur.imu(t, z)
            ekf.update_gyro(t, z)
        else:
            valid = np.isfinite(z)
            if valid.all():
                wheel_vel = J_pinv @ z
            wheel_heading = float(wrap(wheel_heading + wheel_vel[2] * (t - last_wheel_t)))
            last_wheel_t = t
            used = ekf.update_wheels(t, z)
            log['used'].append((t, len(used)))
            continue
        log['t'].append(t)
        log['true'].append(np.concatenate([sim.vel[k], [sim.pose[k, 2]]]))
        log['fused'].append(ekf.x[[3, 4, 5, 2]].copy())
        log['wheels'].append(wheel_vel.copy())
        log['cur_w'].append(cur.omega)
        log['cur_th'].append(cur.heading)
        log['wheel_th'].append(wheel_heading)
        log['bias'].append(ekf.x[6])
        log['cur_bias_log'].append(cur.bias)
        log['true_bias'].append(gyro_bias(t))
    out = {k: np.array(v) for k, v in log.items()}
    out['ekf'] = ekf
    return out


def metrics(r):
    t = r['t']
    true, fused, wheels = r['true'], r['fused'], r['wheels']
    slip = np.array([any(a <= ti < b for _w, a, b, _e in SLIPS) for ti in t])
    stuck = (t >= STUCK[0]) & (t < STUCK[1])
    normal = ~slip & ~stuck
    rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
    lin = lambda est, mask: rms(np.linalg.norm(est[mask, :2] - true[mask, :2], axis=1))
    before_stuck = t < STUCK[0]
    return {
        'linear velocity RMS, normal (m/s)': (lin(wheels, normal), lin(fused, normal)),
        'linear velocity RMS, one wheel slipping (m/s)': (lin(wheels, slip), lin(fused, slip)),
        'linear velocity RMS, stuck on wall (m/s)': (lin(wheels, stuck), lin(fused, stuck)),
        'omega RMS (rad/s)': (rms(r['cur_w'] - true[:, 2]), rms(fused[:, 2] - true[:, 2])),
        'heading error just before stuck (deg)': (
            math.degrees(abs(wrap(r['cur_th'][before_stuck][-1] - true[before_stuck, 3][-1]))),
            math.degrees(abs(wrap(fused[before_stuck, 3][-1] - true[before_stuck, 3][-1])))),
        'gyro bias error at end (rad/s)': (
            abs(float(r['cur_bias_log'][-1] - r['true_bias'][-1])),
            abs(float(r['bias'][-1] - r['true_bias'][-1]))),
    }


def report(r):
    print('\n=== gyro + wheel fusion vs current methods (synthetic robot, 47 s) ===')
    print(f'{"":46s} {"current/wheels":>15s} {"fused":>9s}')
    for name, (base, fused) in metrics(r).items():
        print(f'{name:46s} {base:15.3f} {fused:9.3f}')
    ekf = r['ekf']
    print(f'wheel updates: {ekf.n_wheel}, rejected entirely: {ekf.n_wheel_rejected}, '
          f'single-wheel slips caught per wheel: {ekf.slips.tolist()}, '
          f'gyro jumps: {ekf.n_gyro_jumps}')
    print('(baseline column: velocities = wheels only; omega/heading/bias = '
          'velocity_controller.py today)')


def plot(r, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t, true, fused, wheels = r['t'], r['true'], r['fused'], r['wheels']
    fig, ax = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    def shade(a):
        for s, e in IDLE:
            a.axvspan(s, e, color='0.9', lw=0)
        for _w, s, e, _x in SLIPS:
            a.axvspan(s, e, color='#fdd', lw=0)
        a.axvspan(*STUCK, color='#ddf', lw=0)

    shade(ax[0])
    ax[0].plot(t, wheels[:, 0], color='tab:orange', lw=0.8, label='wheels only')
    ax[0].plot(t, fused[:, 0], color='tab:blue', lw=1.2, label='fused')
    ax[0].plot(t, true[:, 0], 'k--', lw=1, label='truth')
    ax[0].set_ylabel('vx (m/s)')
    ax[0].legend(loc='upper right', ncol=3)
    ax[0].set_title('grey = idle, red = one wheel slipping, blue = stuck on a wall '
                    '(wheels spin, robot does not move)')

    shade(ax[1])
    ax[1].plot(t, r['cur_w'] - true[:, 2], color='tab:orange', lw=0.8,
               label='current (gyro - idle-learned bias)')
    ax[1].plot(t, fused[:, 2] - true[:, 2], color='tab:blue', lw=1, label='fused')
    ax[1].set_ylabel('omega error (rad/s)')
    ax[1].set_ylim(-0.1, 0.1)
    ax[1].legend(loc='upper right')

    shade(ax[2])
    ax[2].plot(t, np.degrees(wrap(r['cur_th'] - true[:, 3])), color='tab:orange',
               label='current (integrated gyro)')
    ax[2].plot(t, np.degrees(wrap(r['wheel_th'] - true[:, 3])), color='tab:green',
               lw=0.8, label='wheels only')
    ax[2].plot(t, np.degrees(wrap(fused[:, 3] - true[:, 3])), color='tab:blue', label='fused')
    ax[2].set_ylabel('heading error (deg)')
    ax[2].legend(loc='upper left')

    shade(ax[3])
    ax[3].plot(t, r['true_bias'], 'k--', label='true gyro bias')
    ax[3].plot(t, r['cur_bias_log'], color='tab:orange', label='current (learned when idle)')
    ax[3].plot(t, r['bias'], color='tab:blue', label='fused estimate')
    ax[3].set_ylabel('gyro bias (rad/s)')
    ax[3].set_xlabel('time (s)')
    ax[3].legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f'plot saved to {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', default='', help='save a PNG plot to this path')
    args = parser.parse_args()
    r = run(seed=args.seed)
    report(r)
    if args.plot:
        plot(r, args.plot)


if __name__ == '__main__':
    main()
