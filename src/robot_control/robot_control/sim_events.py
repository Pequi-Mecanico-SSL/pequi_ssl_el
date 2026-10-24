"""Synthetic robot with known collision / skid / stall / push events, for
testing robot_events without hardware (pure python/numpy, no ROS).

30 s scenario (EVENTS lists the truth):
     3 -  8 s   hard driving (false-positive check: accelerations at the limit)
     8.0 s      head-on hit into a wall at ~1 m/s (impact)
     8 - 9.5 s  pinned: command still forward, wheels stalled, current at limit
    12 - 13 s   wheel 1 loses traction and spins 6 rad/s too fast (skid)
    15.0 s      hit on the side: +3 rad/s rotation in 10 ms (impact)
    19 - 20 s   idle robot pushed sideways at 0.3 m/s by an opponent (push)
    21 - 30 s   hard driving and 4 rad/s spins (false-positive check)

Sensor models are plausible guesses, not measurements of our hardware: the
tests check the detection logic; thresholds must be tuned on real recordings.
    accelerometer  body-frame acceleration + 0.3 m/s^2 noise + tilt offset,
                   clipped at the +-4 g range
    motor current  ESC velocity loop: 0.1 A per rad/s of tracking error,
                   clipped at the 1.3 A limit, plus friction and noise
"""
import math

import numpy as np

from .geometry import (DEFAULT_ROBOT_RADIUS, DEFAULT_WHEEL_ANGLES_DEG,
                       DEFAULT_WHEEL_RADIUS, wheel_matrix)

DURATION = 30.0
IMU_HZ = 200.0
WHEEL_HZ = 50.0
CURRENT_LIMIT = 1.3
ACCEL_RANGE = 4.0 * 9.80665
SIGMA_GYRO = 0.01
SIGMA_WHEEL = 0.3
SIGMA_ACCEL = 0.3
SIGMA_CURRENT = 0.03
TILT_OFFSET = (0.15, -0.10)   # m/s^2, gravity leaking into x/y from a tilted IMU

IMPACT_T = 8.0
STALL = (8.0, 9.5)
SKID = (1, 12.0, 13.0, 6.0)   # wheel, start, end, extra rad/s
SIDE_HIT_T = 15.0
PUSH = (19.0, 20.0)
IDLE = [(0.0, 3.0), (18.0, 21.0)]

EVENTS = [
    ('impact', IMPACT_T, IMPACT_T),
    ('stall', *STALL),
    ('skid', SKID[1], SKID[2]),
    ('impact', SIDE_HIT_T, SIDE_HIT_T),
    ('push', *PUSH),
]


def in_window(t, windows):
    return any(a <= t < b for a, b in windows)


class EventRun:

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.J = np.array(wheel_matrix(DEFAULT_WHEEL_ANGLES_DEG, DEFAULT_ROBOT_RADIUS,
                                       DEFAULT_WHEEL_RADIUS))
        n_seg = int(DURATION / 1.0) + 1
        targets = np.column_stack([rng.uniform(-1.2, 1.2, n_seg),
                                   rng.uniform(-1.2, 1.2, n_seg),
                                   rng.uniform(-4.0, 4.0, n_seg)])

        dt = 0.001
        n = int(DURATION / dt) + 1
        self.t = np.arange(n) * dt
        self.cmd = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        v = np.zeros(3)
        for k, t in enumerate(self.t):
            if in_window(t, IDLE):
                cmd = np.zeros(3)
            elif IMPACT_T - 1.0 <= t < STALL[1]:
                cmd = np.array([1.0, 0.0, 0.0])      # driving straight into the wall
            else:
                cmd = targets[int(t)]
            self.cmd[k] = cmd

            if STALL[0] <= t < STALL[1]:
                v = v * max(0.0, 1.0 - dt / 0.02) if t < STALL[0] + 0.02 else np.zeros(3)
            elif PUSH[0] <= t < PUSH[1]:
                v = v + (np.array([0.0, 0.3, 0.0]) - v) * min(1.0, dt / 0.05)
            else:
                dv = (cmd - v) * (dt / 0.1)
                lim = np.array([3.0, 3.0, 20.0]) * dt
                v = v + np.clip(dv, -lim, lim)
            if abs(t - SIDE_HIT_T) < 0.01 - 1e-9 and t >= SIDE_HIT_T:
                v = v + np.array([0.0, 0.0, 3.0 * dt / 0.01])
            self.vel[k] = v

        # body-frame acceleration a = dv/dt + omega x v, averaged over 5 ms
        dv = np.gradient(self.vel, dt, axis=0)
        acc = np.column_stack([dv[:, 0] - self.vel[:, 2] * self.vel[:, 1],
                               dv[:, 1] + self.vel[:, 2] * self.vel[:, 0]])
        kernel = np.ones(5) / 5.0
        acc = np.column_stack([np.convolve(acc[:, i], kernel, mode='same') for i in range(2)])

        self.imu = []     # (t, gyro_z, ax, ay)
        for t in np.arange(0.0, DURATION, 1.0 / IMU_HZ):
            k = self.index(t)
            ax, ay = acc[k] + np.array(TILT_OFFSET) + rng.normal(0, SIGMA_ACCEL, 2)
            self.imu.append((t, self.vel[k, 2] + 0.02 + rng.normal(0, SIGMA_GYRO),
                             float(np.clip(ax, -ACCEL_RANGE, ACCEL_RANGE)),
                             float(np.clip(ay, -ACCEL_RANGE, ACCEL_RANGE))))

        self.wheels = []  # (t, speeds, currents, setpoints)
        for t in np.arange(0.0, DURATION, 1.0 / WHEEL_HZ):
            k = self.index(t)
            true_speed = self.J @ self.vel[k]
            setpoint = self.J @ self.cmd[k]
            current = np.clip(0.1 * (setpoint - true_speed), -CURRENT_LIMIT, CURRENT_LIMIT)
            current = current + 0.05 * np.sign(true_speed) + rng.normal(0, SIGMA_CURRENT, 4)
            speeds = true_speed + rng.normal(0, SIGMA_WHEEL, 4)
            wheel, a, b, extra = SKID
            if a <= t < b:
                speeds[wheel] += extra
            self.wheels.append((t, speeds, current, setpoint))

    def index(self, t):
        return min(int(round(t * 1000.0)), len(self.t) - 1)

    def stream(self):
        """Time-ordered ('imu', t, (gyro, ax, ay)) / ('wheels', t, (speeds, currents, setpoints))."""
        ev = ([('imu', t, (g, ax, ay)) for t, g, ax, ay in self.imu]
              + [('wheels', t, (s, c, sp)) for t, s, c, sp in self.wheels])
        return sorted(ev, key=lambda e: (e[1], e[0] == 'imu'))
