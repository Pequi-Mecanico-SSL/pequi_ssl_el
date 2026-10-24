"""Collision / skid / stall / push detection from on-robot sensors (no ROS).

Events (dicts) returned by the input methods:
  {'event': 'impact', 'kind': 'linear', 't', 'accel', 'direction_deg', 'saturated'}
  {'event': 'impact', 'kind': 'rotational', 't', 'ang_accel'}
  {'event': 'skid',  'state': 'start'|'end', 't', 'wheel', 'candidates'}
  {'event': 'stall', 'state': 'start'|'end', 't', 'wheels'}
  {'event': 'push',  'state': 'start'|'end', 't'}

How each is detected, and why that signal:
  impact               reported impact_window_s after the first threshold
                       crossing, with the peak value inside that window
  impact (linear)      horizontal acceleration beyond what the motors can
                       produce, after removing the centripetal part
                       (omega x v, from gyro and wheels: driving fast while
                       spinning reaches ~7 m/s^2). The along-track part is not
                       removed: in a head-on hit the wheels stop with the
                       robot, so it would hide the hit. Axis signs matter for
                       the centripetal term and direction_deg (see accel_axes).
  impact (rotational)  gyro angular acceleration beyond what the motors can
                       produce (an opponent hitting a corner)
  skid                 a wheel far from the fused velocity
                       (odom/wheel_residuals after the EKF update). With two
                       over the threshold only the worse is flagged; three or
                       more at once is not a wheel skid (model mismatch,
                       impact) and is ignored.
                       Wheels that drive in nearly opposite directions (ours:
                       0/3 and 1/2, 195 deg apart) produce almost the same
                       signature, so a skid is certain but which wheel of such
                       a pair is not: 'candidates' lists both, 'wheel' is the
                       best guess.
  stall                the setpoints ask for motion, >= 2 wheels have their
                       current near the limit, and the robot's velocity (from
                       the wheels) is not getting closer to the commanded one.
                       Hard acceleration also saturates the current, but the
                       velocity error then shrinks quickly. Judged on the
                       whole robot: single wheels can briefly move away from
                       their setpoint during hard maneuvers.
  push                 robot commanded not to translate (for settle_s) but the
                       wheels turn, or the motors fight back with current.
                       Rotation commands do not count as motion here: when a
                       shove spins the robot, velocity_controller damps the
                       spin, and that must not end the push.
"""
import math
from collections import deque

import numpy as np


class Hysteresis:
    """On after the condition holds for on_s, off after it is false for off_s."""

    def __init__(self, on_s, off_s):
        self.on_s, self.off_s = on_s, off_s
        self.active = False
        self.since = None

    def update(self, t, condition):
        """Returns 'start', 'end' or None."""
        if condition != self.active:
            if self.since is None:
                self.since = t
            if t - self.since >= (self.on_s if condition else self.off_s):
                self.active = condition
                self.since = None
                return 'start' if condition else 'end'
        else:
            self.since = None
        return None


class EventDetector:

    def __init__(self, wheel_rows,
                 current_limit=1.3,          # A, must match can_bridge current_limit_a
                 accel_range=4.0 * 9.80665,  # m/s^2, imu.py full scale
                 impact_accel=8.0,           # m/s^2 horizontal, ~0.8 g
                 impact_ang_accel=60.0,      # rad/s^2
                 impact_refractory_s=0.3,
                 impact_window_s=0.02,
                 accel_offset_tau_s=5.0,     # tilt/gravity offset tracking
                 accel_axes=('+x', '+y'),    # imu axis giving robot forward, left
                 sigma_wheel=0.3,
                 skid_sigmas=3.0,
                 robot_radius=0.077,         # m, scales omega into m/s
                 stall_min_error=0.3,        # m/s, commanded vs measured
                 stall_current_frac=0.8,
                 stall_max_progress=0.5,     # m/s^2, error shrink rate
                 stall_window_s=0.1,         # s, window the shrink rate is measured over
                 stall_on_s=0.15,            # s, condition must hold this long to start
                 stall_off_s=0.2,            # s, and be false this long to end
                 push_speed=0.1,             # m/s
                 push_current_frac=0.4,
                 push_cmd_speed=0.05,        # m/s, commanded translation below this = still
                 settle_s=0.5):
        J = np.asarray(wheel_rows, dtype=float)
        self.J_pinv = np.linalg.pinv(J)
        self.n = len(wheel_rows)
        drive = J[:, :2] / np.linalg.norm(J[:, :2], axis=1, keepdims=True)
        self.candidates = [sorted({i} | {j for j in range(self.n)
                                          if j != i and abs(drive[i] @ drive[j]) > 0.9})
                           for i in range(self.n)]
        self.current_limit = current_limit
        self.saturation = 0.98 * accel_range
        self.impact_accel = impact_accel
        self.impact_ang_accel = impact_ang_accel
        self.refractory = impact_refractory_s
        self.impact_window = impact_window_s
        self.pending = None              # impact being measured
        self.offset_tau = accel_offset_tau_s
        self.axes = [self._axis(a) for a in accel_axes]
        self.skid_threshold = skid_sigmas * sigma_wheel
        self.robot_radius = robot_radius
        self.stall_min_error = stall_min_error
        self.stall_current = stall_current_frac * current_limit
        self.stall_max_progress = stall_max_progress
        self.stall_window = stall_window_s
        self.push_speed = push_speed
        self.push_current = push_current_frac * current_limit
        self.push_cmd_speed = push_cmd_speed
        self.settle_s = settle_s

        self.accel_offset = np.zeros(2)
        self.last_imu = None             # (t, gyro)
        self.last_impact = -math.inf
        self.skid = [Hysteresis(0.03, 0.1) for _ in range(self.n)]
        self.stall = Hysteresis(stall_on_s, stall_off_s)
        self.push = Hysteresis(0.1, 0.3)
        self.stall_wheels = []
        self.error_history = deque()     # (t, |commanded - measured| body velocity)
        self.setpoints = np.zeros(self.n)
        self.body = np.zeros(3)          # latest body velocity from the wheels
        self.moving_cmd_t = -math.inf    # last time a translation was commanded

    @staticmethod
    def _axis(spec):
        sign = -1.0 if spec.startswith('-') else 1.0
        return sign, 'xyz'.index(spec[-1])

    # ---- inputs ------------------------------------------------------------

    def imu(self, t, gyro_z, accel_xyz):
        events = []
        a = np.array([s * accel_xyz[i] for s, i in self.axes])
        vx, vy = self.body[0], self.body[1]
        a -= np.array([-gyro_z * vy, gyro_z * vx])   # centripetal
        saturated = any(abs(v) >= self.saturation for v in accel_xyz[:2])
        horizontal = a - self.accel_offset
        magnitude = float(np.hypot(*horizontal))

        linear_hit = magnitude > self.impact_accel or saturated
        ang_accel = None
        if self.last_imu is not None:
            dt = t - self.last_imu[0]
            if 1e-4 < dt < 0.05:
                ang_accel = (gyro_z - self.last_imu[1]) / dt
        self.last_imu = (t, gyro_z)
        rot_hit = ang_accel is not None and abs(ang_accel) > self.impact_ang_accel

        if (linear_hit or rot_hit) and self.pending is None \
                and t - self.last_impact > self.refractory:
            self.last_impact = t
            self.pending = {'event': 'impact', 't': t, 'kind': 'rotational',
                            'accel': 0.0, 'saturated': False, 'ang_accel': 0.0}
        if self.pending is not None:
            p = self.pending
            if linear_hit or magnitude > p['accel']:
                if magnitude > p['accel']:
                    p['accel'] = magnitude
                    # direction the robot was pushed toward, body frame
                    p['direction_deg'] = math.degrees(math.atan2(horizontal[1], horizontal[0]))
            if linear_hit:
                p['kind'] = 'linear'
            p['saturated'] = p['saturated'] or saturated
            if ang_accel is not None and abs(ang_accel) > abs(p['ang_accel']):
                p['ang_accel'] = ang_accel
            if t - p['t'] >= self.impact_window:
                if p['kind'] == 'linear':
                    events.append({'event': 'impact', 'kind': 'linear', 't': p['t'],
                                   'accel': round(p['accel'], 2), 'saturated': p['saturated'],
                                   'direction_deg': round(p['direction_deg'], 1)})
                else:
                    events.append({'event': 'impact', 'kind': 'rotational', 't': p['t'],
                                   'ang_accel': round(p['ang_accel'], 1)})
                self.pending = None
        if not linear_hit and self.last_imu is not None and t - self.last_impact > self.refractory:
            # slowly follow tilt / gravity leakage; frozen around impacts
            k = min(1.0, 0.005 / self.offset_tau)
            self.accel_offset += k * (a - self.accel_offset)
        return events

    def commanded(self):
        """Body velocity the setpoints ask for (vx, vy, omega)."""
        return self.J_pinv @ self.setpoints

    def wheel_setpoints(self, t, setpoints):
        self.setpoints = np.asarray(setpoints, dtype=float)
        if np.hypot(*self.commanded()[:2]) >= self.push_cmd_speed:
            self.moving_cmd_t = t

    def wheels(self, t, speeds, currents):
        events = []
        speeds = np.asarray(speeds, dtype=float)
        currents = np.asarray(currents, dtype=float)
        cmd = self.commanded()
        if np.hypot(cmd[0], cmd[1]) >= self.push_cmd_speed:
            self.moving_cmd_t = t
        valid = np.isfinite(speeds)
        body = (self.J_pinv[:, valid] @ speeds[valid]) if valid.sum() >= 3 else None
        if body is not None:
            self.body = body

        # stall
        saturated = [i for i in range(self.n)
                     if math.isfinite(currents[i]) and abs(currents[i]) > self.stall_current]
        stalling = False
        if body is not None:
            scale = np.array([1.0, 1.0, self.robot_radius])
            asks_for_motion = float(np.linalg.norm(cmd * scale)) > self.stall_min_error
            error = float(np.linalg.norm((cmd - body) * scale))
            self.error_history.append((t, error))
            while self.error_history[0][0] < t - self.stall_window - 1e-9:
                self.error_history.popleft()
            t_old, old_error = self.error_history[0]
            if t - t_old >= 0.5 * self.stall_window:
                progress = (old_error - error) / (t - t_old)
                stalling = (asks_for_motion and error > self.stall_min_error
                            and len(saturated) >= 2 and progress < self.stall_max_progress)
        change = self.stall.update(t, stalling)
        if change == 'start':
            self.stall_wheels = saturated
        if change:
            events.append({'event': 'stall', 'state': change, 't': t,
                           'wheels': self.stall_wheels})

        # push
        commanded_still = t - self.moving_cmd_t > self.settle_s
        speed = float(np.hypot(body[0], body[1])) if body is not None else 0.0
        fighting = int(np.sum(np.abs(np.nan_to_num(currents)) > self.push_current))
        # current from a commanded spin is not the motors fighting a push
        spinning = abs(cmd[2]) * self.robot_radius > self.push_cmd_speed
        change = self.push.update(t, commanded_still and (
            speed > self.push_speed or (fighting >= 2 and not spinning)))
        if change:
            events.append({'event': 'push', 'state': change, 't': t})
        return events

    def residuals(self, t, residuals):
        res = np.nan_to_num(np.abs(np.asarray(residuals, dtype=float)))
        over = res > self.skid_threshold
        if over.sum() >= 3:
            over[:] = False  # not a single-wheel problem
        elif over.sum() == 2:
            # one spare measurement cannot always tell which of two wheels is
            # off; blaming the worse one avoids flagging an innocent neighbour
            over[:] = False
            over[int(np.argmax(res))] = True
        events = []
        for i in range(self.n):
            change = self.skid[i].update(t, bool(over[i]))
            if change:
                events.append({'event': 'skid', 'state': change, 't': t, 'wheel': i,
                               'candidates': self.candidates[i]})
        return events

    def status(self):
        return {'skid': [i for i in range(self.n) if self.skid[i].active],
                'stalled': self.stall.active,
                'pushed': self.push.active,
                'last_impact': None if self.last_impact == -math.inf else self.last_impact}
