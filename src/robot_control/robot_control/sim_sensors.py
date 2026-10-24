"""Synthetic robot with known ground truth, for testing odometry without a
simulator or hardware (pure python/numpy, no ROS).

A 47 s drive with realistic defects:
    gyro bias      starts at 0.02 rad/s and warms up to 0.045 rad/s
    idle           0-10 s (power-on), 25-27 and 39-41 s: robot not commanded
    wheel 2 slip   17-19 s: spins 6 rad/s faster than the ground speed
    wheel 0 slip   32-32.5 s: short traction loss
    wheel 3 stale  22-22.5 s: no telemetry (NaN)
    stuck          43-44 s: robot pinned against a wall, all wheels still
                   spinning at the commanded speed (undetectable without vision)
"""
import math

import numpy as np

from .geometry import (DEFAULT_ROBOT_RADIUS, DEFAULT_WHEEL_ANGLES_DEG,
                       DEFAULT_WHEEL_RADIUS, wheel_matrix)

DURATION = 47.0
IMU_HZ = 200.0
WHEEL_HZ = 50.0
SIGMA_GYRO = 0.01
SIGMA_WHEEL = 0.3
IDLE = [(0.0, 10.0), (25.0, 27.0), (39.0, 41.0)]
SLIPS = [(2, 17.0, 19.0, 6.0), (0, 32.0, 32.5, -8.0)]  # wheel, start, end, extra rad/s
STALE = [(3, 22.0, 22.5)]
STUCK = (43.0, 44.0)
ACCEL_LIMIT = 3.0       # m/s^2
ANG_ACCEL_LIMIT = 20.0  # rad/s^2


def gyro_bias(t):
    return 0.02 + 0.025 * min(1.0, t / 40.0)


def in_window(t, windows):
    return any(a <= t < b for a, b in windows)


class SyntheticRun:
    """Precomputed truth at 1 kHz plus sensor streams."""

    def __init__(self, seed=0, duration=DURATION):
        rng = np.random.default_rng(seed)
        self.J = np.array(wheel_matrix(DEFAULT_WHEEL_ANGLES_DEG, DEFAULT_ROBOT_RADIUS,
                                       DEFAULT_WHEEL_RADIUS))
        # commanded body velocity: a new random target every 1.5 s
        n_seg = int(duration / 1.5) + 1
        targets = np.column_stack([rng.uniform(-1.0, 1.0, n_seg),
                                   rng.uniform(-1.0, 1.0, n_seg),
                                   rng.uniform(-4.0, 4.0, n_seg)])

        dt = 0.001
        n = int(duration / dt) + 1
        self.t = np.arange(n) * dt
        self.cmd = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))     # true body velocity (vx, vy, w)
        self.pose = np.zeros((n, 3))    # true odometry-frame pose
        v = np.zeros(3)
        pose = np.zeros(3)
        for k, t in enumerate(self.t):
            cmd = np.zeros(3) if in_window(t, IDLE) else targets[int(t / 1.5)]
            self.cmd[k] = cmd
            if STUCK[0] <= t < STUCK[1]:
                v = np.zeros(3)
            else:
                # first-order response with acceleration limits
                dv = (cmd - v) * (dt / 0.1)
                lim = np.array([ACCEL_LIMIT, ACCEL_LIMIT, ANG_ACCEL_LIMIT]) * dt
                v = v + np.clip(dv, -lim, lim)
            self.vel[k] = v
            self.pose[k] = pose
            c, s = math.cos(pose[2]), math.sin(pose[2])
            pose = pose + dt * np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])
            pose[2] = (pose[2] + math.pi) % (2 * math.pi) - math.pi

        self.imu = []     # (t, gyro_z)
        for t in np.arange(0.0, duration, 1.0 / IMU_HZ):
            k = self.index(t)
            self.imu.append((t, self.vel[k, 2] + gyro_bias(t) + rng.normal(0, SIGMA_GYRO)))

        self.wheels = []  # (t, [4 speeds])
        for t in np.arange(0.0, duration, 1.0 / WHEEL_HZ):
            k = self.index(t)
            if STUCK[0] <= t < STUCK[1]:
                # wheels keep turning as commanded (ESC tracks setpoint), robot does not move
                speeds = self.J @ self.cmd[k]
            else:
                speeds = self.J @ self.vel[k]
            speeds = speeds + rng.normal(0, SIGMA_WHEEL, len(speeds))
            for wheel, a, b, extra in SLIPS:
                if a <= t < b:
                    speeds[wheel] += extra
            for wheel, a, b in STALE:
                if a <= t < b:
                    speeds[wheel] = math.nan
            self.wheels.append((t, speeds))

    def index(self, t):
        return min(int(round(t * 1000.0)), len(self.t) - 1)

    def events(self):
        """All sensor readings in time order: ('imu', t, gyro) / ('wheels', t, speeds)."""
        ev = [('imu', t, z) for t, z in self.imu] + [('wheels', t, z) for t, z in self.wheels]
        return sorted(ev, key=lambda e: (e[1], e[0] == 'imu'))
