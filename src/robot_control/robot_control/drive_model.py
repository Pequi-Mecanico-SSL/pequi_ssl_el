"""Motion model of an omniwheel base: wheel speeds, body velocity and motor currents.

Conventions (REP 103): x forward, y left, yaw positive = CCW from top; a body
velocity is (vx, vy, w) in m/s and rad/s. Positive wheel speed = clockwise seen
from outside the robot. Wheel angle = position around the robot, CCW from the
front, one entry per motor id.

Kinematics: on carpet a wheel turns faster than its contact point moves over
the ground, by a factor 1 + oblique_loss x |sin a| + creep x traction current.
a is the angle between the contact point's motion and the wheel's rolling
direction (the rollers turning under it); the traction current is the motor
current beyond what keeps the wheel rolling, positive when driving, negative
when braking. Both terms are zero on a hard floor.

Dynamics, as motor currents: the rollers resist turning with a constant force
along each wheel's axis (roller_friction, N), each wheel takes a constant
current to keep rolling (rolling_current, A), and the rest accelerates the
body (mass, inertia). Of the force that accelerates a translation only the
share efficiency arrives: the carpet takes the rest under load.

Grip: a wheel puts on the ground at most grip x its load; the load is the
static one (centre of mass at com) shifted by the body's acceleration acting at
com_height. Accelerating towards a wheel takes load off it.
"""
import math

import numpy as np


def soft_sign(x, x0):
    return x / np.sqrt(x * x + x0 * x0)


def feasible_scale(base, delta, limit):
    """Largest s in [0, 1] keeping |base + s * delta| <= limit (one per entry) on every entry; 0 when there is none."""
    lo, hi = 0.0, 1.0
    for b, d, lim in zip(base, delta, limit):
        if abs(d) < 1e-9:
            continue
        s1, s2 = (-lim - b) / d, (lim - b) / d
        if s1 > s2:
            s1, s2 = s2, s1
        lo, hi = max(lo, s1), min(hi, s2)
    return hi if hi >= lo else 0.0


class Model:
    LAT_SPEED = 0.03     # m/s, roller friction is fully developed above this
    ROLL_SPEED = 1.0     # rad/s, same for the rolling current

    def __init__(self, wheel_angles_deg, robot_radius, wheel_radius, oblique_loss=0.0, creep=0.0,
                 mass=2.0, inertia=0.008, motor_kt=0.0347, roller_friction=0.0, rolling_current=0.0,
                 grip=0.0, com=(0.0, 0.0), com_height=0.0, efficiency=1.0):
        a = np.radians(np.asarray(wheel_angles_deg, float))
        self.n = len(a)
        per_wheel = lambda v: np.full(self.n, float(v)) if np.isscalar(v) else np.asarray(v, float)
        self.r = per_wheel(wheel_radius)
        self.axis_dir = np.c_[np.cos(a), np.sin(a)]
        # contact-point speed along each wheel's rolling direction per unit of (vx, vy, w)
        self.J = np.c_[-np.sin(a), np.cos(a), per_wheel(robot_radius)]
        self.K = self.J / self.r[:, None]
        self.K_pinv = np.linalg.pinv(self.K)
        self.Kxy_pinv = np.linalg.pinv(self.K[:, :2])
        self.oblique = per_wheel(oblique_loss)
        self.creep = per_wheel(creep)
        self.roller_friction = per_wheel(roller_friction)
        self.rolling_current = per_wheel(rolling_current)
        # motor currents for a body wrench (Fx, Fy, torque)
        self.wrench_map = np.linalg.pinv(self.J.T) * (self.r / motor_kt)[:, None]
        self.accel_map = self.wrench_map * np.array([mass / efficiency, mass / efficiency, inertia])
        # wheel loads (N): static, and what each m/s^2 of body acceleration moves between the wheels
        pos = np.c_[np.cos(a), np.sin(a)] * per_wheel(robot_radius)[:, None]
        balance = np.linalg.pinv(np.vstack([np.ones(self.n), pos.T]))
        weight = mass * 9.81
        self.load = balance @ (weight, weight * com[0], weight * com[1])
        self.load_shift = -balance[:, 1:] * (mass * com_height)
        self.grip_current = float(grip) * self.r / motor_kt       # A per N of load; 0 = no grip limit

    def _slip_factor(self, v, current):
        """Wheel surface speed over contact-point speed, per wheel, at body velocity v."""
        drive = self.J @ v
        lat = self.axis_dir @ v[:2]
        factor = 1.0 + self.oblique * np.abs(lat) / (np.hypot(drive, lat) + 1e-9)
        if current is not None:
            rolling = soft_sign(drive / self.r, self.ROLL_SPEED)
            factor += self.creep * (np.asarray(current, float) * rolling - self.rolling_current * np.abs(rolling))
        return drive, lat, np.maximum(factor, 0.5)

    def wheel_speeds(self, v, current=None):
        """Wheel speeds (rad/s) that move the body at v while the motors carry current (A)."""
        drive, _, factor = self._slip_factor(np.asarray(v, float), current)
        return drive / self.r * factor

    def body_velocity(self, wheels, yaw_rate=None, current=None):
        """Body velocity from wheel speeds and motor currents; with yaw_rate (gyro) only the translation comes from the wheels."""
        wheels = np.asarray(wheels, float)
        v = np.zeros(3)
        factor = np.ones(self.n)
        for _ in range(4):
            w = wheels / factor
            if yaw_rate is None:
                v = self.K_pinv @ w
            else:
                v[:2] = self.Kxy_pinv @ (w - self.K[:, 2] * yaw_rate)
                v[2] = yaw_rate
            factor = self._slip_factor(v, current)[2]
        return v

    def grip_currents(self, accel):
        """Motor current (A) each wheel can put on the ground while the body accelerates (ax, ay)."""
        if not self.grip_current.any():
            return np.full(self.n, np.inf)
        load = np.maximum(self.load + self.load_shift @ np.asarray(accel, float)[:2], 0.0)
        return self.grip_current * load + self.rolling_current

    def friction_currents(self, v):
        """Motor currents (A) that hold the body velocity v against friction."""
        v = np.asarray(v, float)
        drive = self.J @ v
        lat = self.axis_dir @ v[:2]
        push = self.roller_friction * soft_sign(lat, self.LAT_SPEED)
        wrench = np.array([push @ self.axis_dir[:, 0], push @ self.axis_dir[:, 1], 0.0])
        return self.wrench_map @ wrench + self.rolling_current * soft_sign(drive / self.r, self.ROLL_SPEED)


class Reference:
    """Velocity reference: follows the command as fast as the motors' current allows.

    Each step moves the reference towards the target with the largest
    acceleration that keeps every motor's predicted current (friction +
    acceleration) inside current_budget and inside what the wheel can put on
    the ground: the wheels neither saturate nor spin, so the motion keeps its
    direction. Yaw acceleration is served first.
    Between steps the reference is constant over the ground, not in the turning robot frame.
    The reference never runs more than max_lead ahead of the measured velocity:
    where the robot accelerates less than the model says, it waits.
    """

    def __init__(self, model, current_budget, max_accel, max_ang_accel, max_wheel_speed, max_lead, max_lead_yaw):
        self.model = model
        self.budget = float(current_budget)
        self.max_accel = float(max_accel)
        self.max_ang_accel = float(max_ang_accel)
        self.max_wheel = float(max_wheel_speed)
        self.max_lead = float(max_lead)
        self.max_lead_yaw = float(max_lead_yaw)
        self.reset()

    def reset(self):
        self.v = np.zeros(3)
        self.acc = np.zeros(3)                    # acceleration granted in the last step
        self.current = np.zeros(self.model.n)     # predicted motor currents of the last step

    def step(self, target, dt, yaw_rate=None, measured=None):
        m = self.model
        target = np.asarray(target, float)
        # wheel speed limit: scale the target, keeping its direction
        peak = np.max(np.abs(m.wheel_speeds(target)))
        if peak > self.max_wheel:
            target = target * (self.max_wheel / peak)

        v = self.v.copy()
        turn = (v[2] if yaw_rate is None else yaw_rate) * dt
        c, s = math.cos(turn), math.sin(turn)
        v[0], v[1] = c * v[0] + s * v[1], -s * v[0] + c * v[1]

        dw = float(np.clip(target[2] - v[2], -self.max_ang_accel * dt, self.max_ang_accel * dt))
        dv = target[:2] - v[:2]
        n, cap = math.hypot(dv[0], dv[1]), self.max_accel * dt
        if n > cap:
            dv = dv * (cap / n)

        # friction where the step is heading: already there when starting from rest
        friction = m.friction_currents(v + (dv[0], dv[1], dw))
        want_w, want_v = dw, dv
        acc_xy = self.acc[:2]
        for _ in range(3):      # the grip depends on the acceleration it allows
            limit = np.minimum(self.budget, m.grip_currents(acc_xy))
            dw = want_w * feasible_scale(friction, m.accel_map[:, 2] * (want_w / dt), limit)
            base = friction + m.accel_map[:, 2] * (dw / dt)
            dv = want_v * feasible_scale(base, m.accel_map[:, :2] @ (want_v / dt), limit)
            acc_xy = dv / dt

        acc = np.array([dv[0], dv[1], dw]) / dt
        self.acc = acc
        self.v = v + acc * dt
        self.current = m.friction_currents(self.v) + m.accel_map @ acc

        if measured is not None:
            lead = self.v[:2] - measured[:2]
            n = math.hypot(lead[0], lead[1])
            if n > self.max_lead:
                self.v[:2] = measured[:2] + lead * (self.max_lead / n)
            self.v[2] = measured[2] + float(np.clip(self.v[2] - measured[2], -self.max_lead_yaw, self.max_lead_yaw))
        return self.v, acc
