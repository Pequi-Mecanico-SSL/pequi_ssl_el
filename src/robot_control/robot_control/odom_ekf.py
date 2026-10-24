"""On-robot odometry filter: gyro + 4 wheel speeds (pure numpy, no ROS).

State (7):  [x, y, theta, vx, vy, omega, gyro_bias]
    x, y, theta   odometry frame (starts at 0 where the filter starts)
    vx, vy        body frame (m/s), REP-103: x forward, y left
    omega         rad/s, CCW positive
    gyro_bias     rad/s, gyro reads omega + gyro_bias

Measurements:
    gyro      z = omega + gyro_bias                        (imu, 200 Hz)
    wheels    z_i = J_i . [vx, vy, omega]  for each wheel  (ESC telemetry, 50 Hz)

The rotation of body velocity into the odometry frame is the nonlinear
(EKF) part; both measurements are linear.

Why fuse: the gyro gives omega precisely but has a drifting bias; the
wheels give all three velocities but slip. The wheels' omega makes the bias
observable at all times, not only while the robot is idle, and 4 wheels
for 3 degrees of freedom (plus the gyro) leave redundancy to detect a
slipping wheel: if the 4-wheel update fails the consistency gate, each
leave-one-out subset is tried and the most consistent one is used; a wheel
that was excluded in the previous update stays excluded while that still
passes (a skidding wheel keeps skidding, and with a single spare
measurement two subsets can fit about equally well).
If none passes (all wheels slipping, robot pushed), the wheel update is
skipped and velocity coasts on the motion model and gyro; after
`max_wheel_rejects` rejections in a row the linear velocity uncertainty is
reset so the wheels can pull the estimate back (recovery from a bad state).
Omega and bias are left alone: the gyro is the trusted omega sensor.

Not detectable here: all four wheels spinning together while the robot is
stuck (e.g. pushing a wall). The wheels agree with each other, so the
filter believes them; only vision can catch that.

Position and heading have no absolute reference here, so they drift slowly;
vision (vision_filter) is the absolute reference.
"""
import itertools
import math

import numpy as np

X, Y, TH, VX, VY, W, B = range(7)

# chi-square 99.9% by degrees of freedom
CHI2 = {1: 10.83, 2: 13.82, 3: 16.27, 4: 18.47}


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class OdomEKF:

    def __init__(self, wheel_rows,
                 sigma_gyro=0.01,        # rad/s, gyro noise
                 sigma_wheel=0.3,        # rad/s, ESC wheel speed noise
                 # white-noise densities: higher = follows faster, noisier
                 accel_noise=3.0,        # m/s^2/sqrt(Hz)
                 # small on purpose: the 200 Hz gyro tracks omega, so wheels
                 # that disagree with it on omega get rejected instead of
                 # dragging the bias estimate
                 ang_accel_noise=3.0,    # rad/s^2/sqrt(Hz)
                 bias_noise=0.002,       # rad/s/sqrt(s), gyro bias drift
                 initial_bias_sigma=0.05,
                 max_wheel_rejects=25,   # consecutive, ~0.5 s at 50 Hz
                 max_step=0.01):
        self.J = np.asarray(wheel_rows, dtype=float)   # (n_wheels, 3)
        self.r_gyro = sigma_gyro ** 2
        self.r_wheel = sigma_wheel ** 2
        self.accel_q = accel_noise ** 2
        self.ang_accel_q = ang_accel_noise ** 2
        self.bias_q = bias_noise ** 2
        self.max_step = max_step
        self.max_wheel_rejects = max_wheel_rejects
        self.consecutive_rejects = 0

        self.x = np.zeros(7)
        self.P = np.diag([0.0, 0.0, 0.0, 0.01, 0.01, 0.01, initial_bias_sigma ** 2])
        self.t = None

        self.n_gyro = 0
        self.n_gyro_jumps = 0
        self.n_wheel = 0
        self.n_wheel_rejected = 0
        self.slips = np.zeros(len(self.J), dtype=int)
        self.last_excluded = None
        self.last_residuals = np.zeros(len(self.J))

    # ---- model ------------------------------------------------------------

    def _step(self, x, P, dt):
        c, s = math.cos(x[TH]), math.sin(x[TH])
        vx, vy, w = x[VX], x[VY], x[W]
        F = np.eye(7)
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

        Q = np.diag([0.0, 0.0, 0.0, self.accel_q * dt, self.accel_q * dt,
                     self.ang_accel_q * dt, self.bias_q * dt])
        return xn, F @ P @ F.T + Q

    def _advance(self, t):
        if self.t is None:
            self.t = t
            return
        while t - self.t > 1e-9:
            dt = min(self.max_step, t - self.t)
            self.x, self.P = self._step(self.x, self.P, dt)
            self.t += dt

    def predict(self, t):
        """(x, P) extrapolated to t without changing the filter."""
        x, P, t0 = self.x.copy(), self.P.copy(), self.t
        if t0 is None:
            return x, P
        while t - t0 > 1e-9:
            dt = min(self.max_step, t - t0)
            x, P = self._step(x, P, dt)
            t0 += dt
        return x, P

    def _apply(self, H, z, R):
        """Standard update; returns normalized innovation squared."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.x[TH] = wrap(self.x[TH])
        I_KH = np.eye(7) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return float(y @ S_inv @ y)

    # ---- measurements -----------------------------------------------------

    def update_gyro(self, t, omega_measured):
        self._advance(t)
        H = np.zeros((1, 7))
        H[0, W] = 1.0
        H[0, B] = 1.0
        z = np.array([omega_measured])
        R = np.array([[self.r_gyro]])
        y = z[0] - self.x[W] - self.x[B]
        if y * y / (self.P[W, W] + 2 * self.P[W, B] + self.P[B, B] + self.r_gyro) > CHI2[1]:
            # sudden rotation change (collision, wheel lift): real motion,
            # not bias; let omega jump without touching the bias estimate
            self.P[W, :] = 0.0
            self.P[:, W] = 0.0
            self.P[W, W] = 10.0
            self.n_gyro_jumps += 1
        self._apply(H, z, R)
        self.n_gyro += 1

    def _wheel_H(self, idx):
        H = np.zeros((len(idx), 7))
        H[:, VX:W + 1] = self.J[idx]
        return H

    def _nis(self, idx, z):
        H = self._wheel_H(idx)
        y = z[idx] - H @ self.x
        S = H @ self.P @ H.T + self.r_wheel * np.eye(len(idx))
        return float(y @ np.linalg.solve(S, y))

    def update_wheels(self, t, speeds):
        """speeds: measured wheel speeds (rad/s), NaN for stale wheels.

        Returns the indices of the wheels used (empty if all rejected).
        """
        self._advance(t)
        z = np.asarray(speeds, dtype=float)
        valid = [i for i in range(len(z)) if math.isfinite(z[i])]
        self.last_residuals = z - self.J @ self.x[VX:W + 1]
        if len(valid) < 3:
            return []

        used = None
        if self._nis(valid, z) <= CHI2[len(valid)]:
            used = valid
        elif len(valid) == 4:
            candidates = []
            for subset in itertools.combinations(valid, 3):
                nis = self._nis(list(subset), z)
                if nis <= CHI2[3]:
                    candidates.append((nis, list(subset)))
            if candidates:
                keep = [c for c in candidates if self.last_excluded not in c[1]]
                used = min(keep or candidates)[1]
                for i in set(valid) - set(used):
                    self.slips[i] += 1
        if used is None:
            self.n_wheel_rejected += 1
            self.consecutive_rejects += 1
            if self.consecutive_rejects >= self.max_wheel_rejects:
                self.consecutive_rejects = 0
                self.P[VX:VY + 1, :] = 0.0
                self.P[:, VX:VY + 1] = 0.0
                self.P[VX, VX] = self.P[VY, VY] = 1.0
            return []
        self.consecutive_rejects = 0
        excluded = set(valid) - set(used)
        self.last_excluded = excluded.pop() if len(valid) == 4 and excluded else None

        self._apply(self._wheel_H(used), z[used], self.r_wheel * np.eye(len(used)))
        # after the update: wheels that agree with the fused velocity are ~noise,
        # a slipping (excluded) wheel stands out
        self.last_residuals = z - self.J @ self.x[VX:W + 1]
        self.n_wheel += 1
        return used
