"""Tracking of vision objects and dead reckoning of our own robot while vision does not see it. No ROS imports: the
node uses it live, and the same code can be run offline over recorded vision."""
import collections
import math


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Track:
    """pose samples of one object -> position and a velocity from the newest sample at least `base` s older"""
    def __init__(self, base=0.1, v_max=4.0):
        self.q = collections.deque()
        self.base, self.v_max = base, v_max
        self.vel = (0.0, 0.0)

    def add(self, t, x, y, th=0.0):
        if self.q and math.hypot(x - self.q[-1][1], y - self.q[-1][2]) > self.v_max * max(t - self.q[-1][0], 0.02) + 0.05:
            self.q.clear()                  # a jump: a new object, or a teleport
            self.vel = (0.0, 0.0)
        self.q.append((t, x, y, th))
        while len(self.q) > 2 and self.q[1][0] <= t - self.base:
            self.q.popleft()
        t0, x0, y0, _ = self.q[0]
        if t - t0 >= 0.5 * self.base:
            self.vel = ((x - x0) / (t - t0), (y - y0) / (t - t0))

    def seen(self, now, timeout):
        return bool(self.q) and now - self.q[-1][0] < timeout

    def age(self, now):
        return now - self.q[-1][0] if self.q else float('inf')

    def state(self):
        _, x, y, th = self.q[-1]
        return x, y, th, self.vel[0], self.vel[1]

    def coast(self, now, horizon):
        """last pose, moved on with the last velocity for at most `horizon` s; the velocity is zero after that"""
        x, y, th, vx, vy = self.state()
        a = min(max(now - self.q[-1][0], 0.0), horizon)
        if now - self.q[-1][0] > horizon:
            return x + vx * a, y + vy * a, th, 0.0, 0.0
        return x + vx * a, y + vy * a, th, vx, vy


class DeadReckoning:
    """Our robot's pose between vision samples, carried forward from the body-frame commands it was sent.

    Model of the robot (measured with step commands on the real robot): it acts on a command `delay` s after it was
    sent; its velocity moves towards the commanded one with at most `accel` m/s^2 (vector); the turn rate follows the
    command at once (angular acceleration is 30-60 rad/s^2). At every vision sample the pose is reset to the sample and
    the velocity to a mix of the vision velocity (weight `vel_mix`) and the model's own velocity."""
    def __init__(self, delay=0.1, accel=3.0, vel_mix=0.5, step=1 / 60):
        self.delay, self.accel, self.vel_mix, self.step = delay, accel, vel_mix, step
        self.cmds = collections.deque()             # (t sent, body vx, body vy, w)
        self.t = None                               # time of the state below
        self.x = self.y = self.th = 0.0
        self.vx = self.vy = 0.0                     # field frame
        self.t_seen = None

    def command(self, t, vx, vy, w):
        self.cmds.append((t, vx, vy, w))
        while len(self.cmds) > 2 and self.cmds[1][0] <= t - self.delay - 2.0:
            self.cmds.popleft()

    def _cmd_at(self, tau):
        c = (0.0, 0.0, 0.0)
        for tc, vx, vy, w in self.cmds:
            if tc > tau:
                break
            c = (vx, vy, w)
        return c

    def observe(self, t, x, y, th, vis_vx, vis_vy):
        if self.t is not None and t >= self.t:
            self._advance(t)                        # the model's velocity at this moment
        m = self.vel_mix if self.t is not None else 1.0
        self.vx = m * vis_vx + (1 - m) * self.vx
        self.vy = m * vis_vy + (1 - m) * self.vy
        self.x, self.y, self.th, self.t, self.t_seen = x, y, th, t, t

    def _advance(self, now):
        while self.t < now - 1e-9:
            dt = min(self.step, now - self.t)
            bvx, bvy, w = self._cmd_at(self.t - self.delay)
            c, s = math.cos(self.th), math.sin(self.th)
            tx, ty = c * bvx - s * bvy, s * bvx + c * bvy
            dvx, dvy = tx - self.vx, ty - self.vy
            dv, lim = math.hypot(dvx, dvy), self.accel * dt
            if dv > lim:
                dvx, dvy = dvx * lim / dv, dvy * lim / dv
            ax, ay = self.vx, self.vy
            self.vx += dvx; self.vy += dvy
            self.x += 0.5 * (ax + self.vx) * dt; self.y += 0.5 * (ay + self.vy) * dt
            self.th = wrap(self.th + w * dt)
            self.t += dt

    def predict(self, now):
        """-> (x, y, th, vx, vy) at `now`, or None before the first sample"""
        if self.t is None:
            return None
        if now > self.t:
            self._advance(now)
        return self.x, self.y, self.th, self.vx, self.vy

    def age(self, now):
        return now - self.t_seen if self.t_seen is not None else float('inf')
