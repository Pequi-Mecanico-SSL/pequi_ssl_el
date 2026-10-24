"""Per-robot extended Kalman filter over SSL-Vision poses.

Pure numpy, no ROS imports, so it can be unit-tested anywhere.

State (6):  [x, y, theta, vx, vy, omega]
    x, y, theta   field frame (m, rad), right-handed, the /vision/poses frame
    vx, vy        body frame (m/s), REP-103: x forward, y left, the cmd_vel frame
    omega         rad/s, CCW positive

Motion model: the body velocity is rotated into the field frame by theta,
which is the nonlinear part (hence EKF). Velocities follow the last
commanded body velocity through a first-order lag when a fresh command is
known (our own robots), and are held constant otherwise (opponents, or no
command stream); either way random acceleration is process noise.

Measurement: vision pose [x, y, theta], stamped with its capture time.
Innovations beyond a Mahalanobis gate are rejected (vision id swaps, glare
ghosts); after `max_rejects` consecutive rejections the filter snaps to the
measurement instead (robot picked up and moved, or a persistent id swap).
Measurements older than the newest one processed (multi-camera latency
differences) are handled by rewinding to a stored posterior and replaying.
"""
import math
from collections import deque

import numpy as np

X, Y, TH, VX, VY, W = range(6)

# chi-square, 3 dof, 99.9%
DEFAULT_GATE = 16.27


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class RobotEKF:

    def __init__(self,
                 sigma_xy=0.005,          # m, vision position noise (1 sigma)
                 sigma_theta=0.03,        # rad, vision orientation noise
                 # White-noise acceleration densities, not peak accelerations:
                 # higher = follows maneuvers faster, noisier velocity.
                 accel_noise=0.5,         # m/s^2/sqrt(Hz)
                 ang_accel_noise=3.0,     # rad/s^2/sqrt(Hz)
                 cmd_tau=0.15,            # s, body velocity lag behind cmd_vel
                 cmd_timeout=0.3,         # s, a command older than this is ignored
                 gate=DEFAULT_GATE,
                 max_rejects=5,
                 max_step=0.01,           # s, prediction sub-step
                 history_s=0.5):          # s, rewind window for late measurements
        self.R = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_theta ** 2])
        self.accel_q = accel_noise ** 2
        self.ang_accel_q = ang_accel_noise ** 2
        self.cmd_tau = cmd_tau
        self.cmd_timeout = cmd_timeout
        self.gate = gate
        self.max_rejects = max_rejects
        self.max_step = max_step
        self.history_s = history_s

        self.x = None
        self.P = None
        self.t = None
        # (t, z, x_post, P_post) per processed measurement, oldest first
        self.history = deque()
        # (t, u) commanded body velocity, oldest first
        self.commands = deque()

        self.consecutive_rejects = 0
        self.n_updates = 0
        self.n_rejected = 0
        self.n_resets = 0
        self.n_dropped_late = 0

    @property
    def initialized(self):
        return self.x is not None

    # ---- commands ---------------------------------------------------------

    def add_command(self, t, vx, vy, omega):
        """Commanded body velocity (cmd_vel) sent at time t."""
        self.commands.append((t, np.array([vx, vy, omega], dtype=float)))
        horizon = t - self.history_s - self.cmd_timeout
        while len(self.commands) > 1 and self.commands[1][0] <= horizon:
            self.commands.popleft()

    def _command_at(self, t):
        u = None
        for tu, cmd in self.commands:
            if tu > t:
                break
            u = (tu, cmd)
        if u is None or t - u[0] > self.cmd_timeout:
            return None
        return u[1]

    # ---- model ------------------------------------------------------------

    def _step(self, x, P, t, dt):
        """Propagate (x, P) from t by dt (dt <= max_step)."""
        c, s = math.cos(x[TH]), math.sin(x[TH])
        vx, vy, w = x[VX], x[VY], x[W]

        F = np.eye(6)
        F[X, TH] = (-vx * s - vy * c) * dt
        F[X, VX] = c * dt
        F[X, VY] = -s * dt
        F[Y, TH] = (vx * c - vy * s) * dt
        F[Y, VX] = s * dt
        F[Y, VY] = c * dt
        F[TH, W] = dt

        xn = x.copy()
        xn[X] += (vx * c - vy * s) * dt
        xn[Y] += (vx * s + vy * c) * dt
        xn[TH] = wrap(x[TH] + w * dt)

        u = self._command_at(t)
        if u is not None and self.cmd_tau > 0.0:
            a = math.exp(-dt / self.cmd_tau)
            xn[VX:] = a * x[VX:] + (1.0 - a) * u
            F[VX, VX] = F[VY, VY] = F[W, W] = a

        Q = np.diag([0.0, 0.0, 0.0,
                     self.accel_q * dt, self.accel_q * dt,
                     self.ang_accel_q * dt])
        return xn, F @ P @ F.T + Q

    def _predict(self, x, P, t0, t1):
        t = t0
        while t1 - t > 1e-9:
            dt = min(self.max_step, t1 - t)
            x, P = self._step(x, P, t, dt)
            t += dt
        return x, P

    def predict(self, t):
        """(x, P) extrapolated to time t; does not change the filter."""
        if not self.initialized:
            return None, None
        if t <= self.t:
            return self.x.copy(), self.P.copy()
        return self._predict(self.x, self.P, self.t, t)

    # ---- measurements -----------------------------------------------------

    def reset(self, t, z):
        self.x = np.array([z[0], z[1], wrap(z[2]), 0.0, 0.0, 0.0])
        # velocity unknown at start: 1 m/s, 5 rad/s
        self.P = np.diag([self.R[0, 0], self.R[1, 1], self.R[2, 2],
                          1.0, 1.0, 25.0])
        self.t = t
        self.history.clear()
        self.history.append((t, np.asarray(z, dtype=float), self.x.copy(), self.P.copy()))
        self.consecutive_rejects = 0

    def _update(self, x, P, z):
        """Returns (x, P, accepted)."""
        y = np.asarray(z, dtype=float) - x[:3]
        y[2] = wrap(y[2])
        S = P[:3, :3] + self.R
        S_inv = np.linalg.inv(S)
        if float(y @ S_inv @ y) > self.gate:
            return x, P, False
        K = P[:, :3] @ S_inv
        x = x + K @ y
        x[TH] = wrap(x[TH])
        # Joseph form keeps P symmetric positive definite
        I_KH = np.eye(6)
        I_KH[:, :3] -= K
        P = I_KH @ P @ I_KH.T + K @ self.R @ K.T
        return x, P, True

    def update(self, t, z):
        """Vision pose z = (x, y, theta) captured at time t.

        Returns 'init', 'accepted', 'rejected', 'reset' or 'dropped'.
        """
        z = np.asarray(z, dtype=float)
        if not self.initialized:
            self.reset(t, z)
            return 'init'

        if t < self.t:
            return self._update_late(t, z)

        x, P = self._predict(self.x, self.P, self.t, t)
        x, P, ok = self._update(x, P, z)
        if not ok:
            self.n_rejected += 1
            self.consecutive_rejects += 1
            if self.consecutive_rejects >= self.max_rejects:
                self.n_resets += 1
                self.reset(t, z)
                return 'reset'
            return 'rejected'

        self.consecutive_rejects = 0
        self.n_updates += 1
        self.x, self.P, self.t = x, P, t
        self.history.append((t, z, x.copy(), P.copy()))
        while self.history and self.history[0][0] < t - self.history_s:
            self.history.popleft()
        return 'accepted'

    def _update_late(self, t, z):
        """Rewind to the last posterior before t, insert z, replay the rest."""
        entries = list(self.history)
        base = None
        for i, (te, _z, _x, _P) in enumerate(entries):
            if te <= t:
                base = i
            else:
                break
        if base is None:
            self.n_dropped_late += 1
            return 'dropped'

        t0, _z, x, P = entries[base]
        x, P = self._predict(x, P, t0, t)
        x, P, ok = self._update(x, P, z)
        if not ok:
            self.n_rejected += 1
            return 'rejected'
        replay = [(t, z, x.copy(), P.copy())]
        tc = t
        for te, ze, _x, _P in entries[base + 1:]:
            x, P = self._predict(x, P, tc, te)
            x, P, _ok = self._update(x, P, ze)
            tc = te
            replay.append((te, ze, x.copy(), P.copy()))

        self.history = deque(entries[:base + 1] + replay)
        self.x, self.P, self.t = x, P, tc
        self.n_updates += 1
        return 'accepted'
