"""Our robot's pose from its wheel encoders (and optionally its gyro) while vision does not see it. No ROS imports: the
node feeds it live, and the same code can be run offline over recordings. Used by solo_node's recovery only, which is
off by default (recover:=false) and has been checked offline on recorded data only, not yet on the robot.

Wheels: /wheels/state positions (rad, integrated on the robot), so a lost Wi-Fi packet loses no motion: the next sample
carries it. Body-frame displacement = M @ (wheel position change); M was fitted against vision on recorded field runs of
the three-wheel robot on carpet (the nominal kinematics -60/60/180 deg, 81.4 mm, r 24.5 mm give the same matrix up to a
sign on x and 10-20 % more travel than the fit, the omni rollers' slip on carpet).

Heading: from the wheels (third row of M), or from the gyro when gyro() is fed: /imu is the raw rate (the robot's
velocity controller removes its bias itself), so the bias is estimated here from the gyro against vision over the last
`bias_window` s of vision.

The pose is kept in an arbitrary odometry frame with a short history; anchor() takes a vision pose, places it at the
time the robot was there (receive time - vis_latency), and sets the transform odometry frame -> field frame so that the
odometry pose at that moment coincides with vision. pose(t) is then the field pose now. wheel_lag: how much older a
wheel sample is than its receive time.

Robot source: robot_pose() takes the pose the robot integrates itself (tools/solo/onboard_odom.py, /solo/odom) instead
of wheels() / gyro(); anchoring, the self-check against vision, pose() and stale() are the same."""
import bisect
import collections
import math

# fitted against vision on recorded field runs of the three-wheel robot: body (dx, dy, dth) = M @ wheel (dpos0, dpos1, dpos2)
M_FIELD = ((-0.01293, 0.01128, -0.00072),
           (0.00802, 0.00801, -0.01342),
           (0.08944, 0.08871, 0.08843))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Odometry:
    def __init__(self, M=M_FIELD, vis_latency=0.1, wheel_lag=0.02, history=4.0, max_wheel=80.0, bias_window=4.0,
                 bias_max=0.1):
        self.M = [list(r) for r in M]
        self.vis_latency, self.wheel_lag, self.history, self.max_wheel = vis_latency, wheel_lag, history, max_wheel
        self.bias_window, self.bias_max = bias_window, bias_max
        self.hist = collections.deque()        # (t robot, x, y, th) in the odometry frame
        self.x = self.y = self.th = 0.0
        self.t_w = None                         # receive time of the last wheel sample
        self.pos = None
        self.anchor_T = None                    # (ox, oy, oth): field = rot(oth) * odom + (ox, oy)
        self.t_anchor = None                    # receive time of the last vision anchor
        self.use_gyro, self.g_t, self.g_ang, self.g_raw, self.bias = False, None, 0.0, 0.0, 0.0
        self.ghist = collections.deque()        # (t robot, gyro angle less the bias, raw gyro angle)
        self.vhist = collections.deque()        # (t robot, unwrapped vision heading, gyro angle then)
        self.resets = 0
        self.chk, self.qual = None, None        # self-check against vision: (t, odom pose, vision pose); error / path, smoothed

    # ---- inputs
    def wheels(self, t, pos):
        """one /wheels/state sample: t = receive time, pos = three wheel positions (rad)"""
        if not all(math.isfinite(p) for p in pos):
            return
        if self.pos is None or len(pos) != len(self.pos):
            self.pos, self.t_w = list(pos), t
            self._push(t - self.wheel_lag)
            return
        d = [p - q for p, q in zip(pos, self.pos)]
        dt = max(t - self.t_w, 1e-3)
        self.pos, self.t_w = list(pos), t
        if max(abs(v) for v in d) > self.max_wheel * dt + 1.0:   # the robot's driver restarted (positions back to 0)
            self.resets += 1
            self._push(t - self.wheel_lag)
            return
        bx = sum(m * v for m, v in zip(self.M[0], d))
        by = sum(m * v for m, v in zip(self.M[1], d))
        if self.use_gyro:
            dth = wrap(self._gyro_at(t - self.wheel_lag) - self.th)
        else:
            dth = sum(m * v for m, v in zip(self.M[2], d))
        mid = self.th + 0.5 * dth
        c, s = math.cos(mid), math.sin(mid)
        self.x += c * bx - s * by
        self.y += s * bx + c * by
        self.th += dth                          # not wrapped: the history interpolates it
        self._push(t - self.wheel_lag)

    def robot_pose(self, t, x, y, th):
        """one /solo/odom sample (the robot's own odometry, tools/solo/onboard_odom.py): t = receive time, (x, y, th) its
        pose in its odom frame. Used instead of wheels()/gyro() (recover_source robot); wheel_lag is then the age of
        that pose when it arrives. A jump (the robot's program restarted: its frame is back at zero) drops the anchor
        and the history: no field pose until vision sees the robot again."""
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(th)):
            return
        if self.t_w is not None:
            dt = max(t - self.t_w, 1e-3)
            if math.hypot(x - self.x, y - self.y) > 3.0 * dt + 0.3 or abs(wrap(th - self.th)) > 15.0 * dt + 0.5:
                self.resets += 1
                self.hist.clear()
                self.anchor_T, self.chk = None, None
                self.x, self.y, self.th = x, y, th
                self.t_w = t
                self._push(t - self.wheel_lag)
                return
            th = self.th + wrap(th - self.th)    # unwrapped: the history interpolates it
        self.x, self.y, self.th, self.t_w = x, y, th, t
        self._push(t - self.wheel_lag)

    def gyro(self, t, gz):
        """one /imu sample (raw rate, rad/s); switches the heading to the gyro"""
        if not math.isfinite(gz):
            return
        self.use_gyro = True
        tr = t - self.wheel_lag
        if self.g_t is not None and 0.0 < tr - self.g_t < 0.1:
            self.g_ang += (gz - self.bias) * (tr - self.g_t)
            self.g_raw += gz * (tr - self.g_t)
        self.g_t = tr
        self.ghist.append((tr, self.g_ang, self.g_raw))
        while len(self.ghist) > 2 and self.ghist[0][0] < tr - self.history - self.bias_window:
            self.ghist.popleft()

    def anchor(self, t, x, y, th):
        """a vision pose received at t (taken at t - vis_latency)"""
        ta = t - self.vis_latency
        o = self._odom_at(ta)
        if o is None:
            return
        ox, oy, oth = o
        self._check(ta, o, (x, y, th))
        dth = th - oth
        c, s = math.cos(dth), math.sin(dth)
        self.anchor_T = (x - (c * ox - s * oy), y - (s * ox + c * oy), dth)
        self.t_anchor = t
        if self.use_gyro:
            self._bias_update(ta, th)

    def _check(self, ta, o, v):
        """every second while vision sees the robot: the odometry's displacement (rotated into the field by the heading
        offset then) against vision's, as error / distance (only when it moved > 0.15 m, and not when the wheels say it
        hardly moved: a robot carried by hand); qual is its running mean"""
        if self.chk is not None and ta - self.chk[0] < 1.0:
            return
        if self.chk is not None and ta - self.chk[0] < 1.5:
            _, o0, v0 = self.chk
            dvx, dvy = v[0] - v0[0], v[1] - v0[1]
            d = math.hypot(dvx, dvy)
            if d > 0.15:
                r = v0[2] - o0[2]
                c, s = math.cos(r), math.sin(r)
                px, py = c * (o[0] - o0[0]) - s * (o[1] - o0[1]), s * (o[0] - o0[0]) + c * (o[1] - o0[1])
                if math.hypot(px, py) > 0.3 * d:          # else the wheels did not turn: carried by hand, not an odometry fault
                    e = math.hypot(px - dvx, py - dvy) / d
                    self.qual = e if self.qual is None else 0.8 * self.qual + 0.2 * e
        self.chk = (ta, o, v)

    def healthy(self, limit=0.7):
        """False when the odometry has disagreed with vision (wrong wheel order or sign, a wheel not reporting...) or has
        not yet been checked against it"""
        return self.qual is not None and self.qual < limit     # unknown until it has been checked once while driving

    # ---- outputs
    def pose(self, t):
        """field pose (x, y, th) of the robot at receive-clock time t (the newest wheel sample, not extrapolated), or
        None before the first anchor"""
        if self.anchor_T is None or not self.hist:
            return None
        ox, oy, oth = self.anchor_T
        c, s = math.cos(oth), math.sin(oth)
        return ox + c * self.x - s * self.y, oy + s * self.x + c * self.y, wrap(self.th + oth)

    def stale(self, t, limit=0.3):
        return self.t_w is None or t - self.t_w > limit

    def travelled(self, since_t):
        """odometry path length since robot-time since_t"""
        pts = [h for h in self.hist if h[0] >= since_t]
        return sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(pts, pts[1:]))

    # ---- internals
    def _push(self, tr):
        self.hist.append((tr, self.x, self.y, self.th))
        while len(self.hist) > 2 and self.hist[0][0] < tr - self.history:
            self.hist.popleft()

    def _odom_at(self, tr):
        h = self.hist
        if not h:
            return None
        if tr >= h[-1][0]:
            return h[-1][1:]
        if tr <= h[0][0]:
            return h[0][1:]
        i = bisect.bisect_left([e[0] for e in h], tr)
        a, b = h[i - 1], h[i]
        f = (tr - a[0]) / max(b[0] - a[0], 1e-9)
        return tuple(a[k] + f * (b[k] - a[k]) for k in (1, 2, 3))

    def _gyro_at(self, tr, k=1):
        g = self.ghist
        if not g:
            return self.th if k == 1 else 0.0
        if tr >= g[-1][0]:
            return g[-1][k]
        i = bisect.bisect_left([e[0] for e in g], tr)
        if i == 0:
            return g[0][k]
        a, b = g[i - 1], g[i]
        f = (tr - a[0]) / max(b[0] - a[0], 1e-9)
        return a[k] + f * (b[k] - a[k])

    def _bias_update(self, ta, th):
        """bias: raw gyro angle change minus vision heading change over the last bias_window s (vision heading
        unwrapped; a gap in vision longer than 0.3 s restarts the window)"""
        gr = self._gyro_at(ta, 2)
        if self.vhist:
            if ta - self.vhist[-1][0] > 0.3:
                self.vhist.clear()
            else:
                th = self.vhist[-1][1] + wrap(th - self.vhist[-1][1])
        self.vhist.append((ta, th, gr))
        while len(self.vhist) > 2 and self.vhist[1][0] < ta - self.bias_window:
            self.vhist.popleft()
        t0, th0, g0 = self.vhist[0]
        T = ta - t0
        if T > 0.5 * self.bias_window:
            b = ((gr - g0) - (th - th0)) / T
            self.bias = max(-self.bias_max, min(self.bias_max, b))
