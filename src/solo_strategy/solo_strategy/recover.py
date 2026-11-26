"""Vision recovery: our robot has left the camera's view; drive it slowly, steered by wheel odometry, back to a place the
camera is known to see. No ROS imports: solo_node runs it live, and the same code can be replayed offline over
recordings. Off by default in solo_node (recover:=false); checked offline on recorded data only, not yet on the robot.

The node calls step() every tick while the robot has been unseen for blind_limit s (predict true) or pose_timeout s
(predict false) or longer, and while a recovery is under way, until vision has had the robot for `frames` detections in
a row. step() returns the field-frame velocity to send
(w is always 0: the robot's own heading hold, on its gyro, keeps the heading while it translates) and a debug dict.

Target, chosen once when the recovery starts, from the last position vision saw:
  1. the vision coverage map (tools/solo/vision_map.py, imported from a path, any failure ignored; map_path empty: the
     module's default map): nearest_good();
  2. otherwise (or if the map's point fails the checks below) the last seen point clamped into the field shrunk by
     `margin` (0.5 m inside the nearest line).
  Either way the target is kept `margin_map` inside the field lines and out of THEIR defense area (+ 0.2 m).
  Behind a goal line within the goal's width + 0.3 m, a waypoint first takes the robot sideways clear of the goal frame.
Limits: stands still after max_time s or max_dist m of odometry travel without a solid reacquisition, always in HALT and
TIMEOUT (whatever `states` holds), in any state not in `states` (default STOP, RUNNING, FREE_KICK, KICKOFF, PENALTY;
not BALL_PLACEMENT, BREAK, ...), when the wheel odometry is stale, with no odometry pose, or when the odometry's
once-a-second check against vision (odometry.Odometry.healthy) says it disagrees, and when the camera sees nothing at
all (no ball, no other robot either: the vision is down, not our robot out of view). In STOP the speed is capped
at stop_speed (the rule's limit is 0.75 m/s)."""
import importlib.util
import math

HALT, STOP, RUNNING, TIMEOUT = 1, 2, 3, 7        # pequi_msgs/GameState state values (solo_node.py: HALT 1, STOP 2, RUNNING 3)
FIELD_L, FIELD_W = 4.52, 3.0
DA_DEPTH, DA_WIDTH = 0.4975, 1.35
GOAL_W, GOAL_D = 0.8, 0.18


def load_map(module_path, map_path):
    """-> a vision map object or None (no module, no map file, any error)"""
    try:
        spec = importlib.util.spec_from_file_location('vision_map_recover', module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m = mod.load(map_path) if map_path else mod.load()
        m.nearest_good(0.0, 0.0)                  # it works at all
        return m
    except Exception:
        return None


def in_their_area(x, y, own_sign, pad=0.2):
    """own_sign: +1 if we defend the positive-x half"""
    return -own_sign * x > FIELD_L / 2 - DA_DEPTH - pad and abs(y) < DA_WIDTH / 2 + pad


def fallback_target(x, y, margin):
    hx, hy = FIELD_L / 2 - margin, FIELD_W / 2 - margin
    return max(-hx, min(hx, x)), max(-hy, min(hy, y))


def choose_target(x, y, own_sign, vmap=None, margin=0.5, margin_map=0.3, min_rate=0.9, min_evidence=30):
    """-> (tx, ty, source)"""
    t, src = None, 'field'
    if vmap is not None:
        try:
            tx, ty = vmap.nearest_good(x, y, min_rate=min_rate, min_evidence=min_evidence)
            tx, ty = fallback_target(float(tx), float(ty), margin_map)     # at least margin_map inside the lines
            r = vmap.rate(tx, ty)
            if r == r and r >= min_rate and vmap.evidence(tx, ty) >= min_evidence:
                t, src = (tx, ty), 'map'
        except Exception:
            t = None
    if t is None:
        t = fallback_target(x, y, margin)
    tx, ty = t
    if in_their_area(tx, ty, own_sign):
        tx = -own_sign * (FIELD_L / 2 - DA_DEPTH - 0.25)            # in front of their area, same y
        src += '+out of their area'
    return tx, ty, src


def goal_waypoint(x, y, tx, ty):
    """robot behind a goal line, inside the goal frame's width: first sideways to clear the frame"""
    if abs(x) > FIELD_L / 2 - 0.02 and abs(y) < GOAL_W / 2 + 0.3:
        side = 1.0 if (ty > y if abs(ty - y) > 1e-3 else y >= 0) else -1.0
        return x, side * (GOAL_W / 2 + 0.4)
    return None


class Recovery:
    def __init__(self, speed=0.4, accel=0.5, max_time=6.0, max_dist=2.5, frames=3, stop_speed=0.6, gain=1.5,
                 tol=0.05, margin=0.5, margin_map=0.3, min_rate=0.9, min_evidence=30, vmap=None,
                 states=(STOP, RUNNING, 4, 5, 6)):
        self.speed, self.accel, self.max_time, self.max_dist, self.frames = speed, accel, max_time, max_dist, frames
        self.stop_speed, self.gain, self.tol, self.margin, self.margin_map = stop_speed, gain, tol, margin, margin_map
        self.min_rate, self.min_evidence, self.vmap, self.states = min_rate, min_evidence, vmap, tuple(states)
        self.reset()

    def reset(self):
        self.active = False                 # an episode is under way (started, not yet solidly reacquired)
        self.t0 = None
        self.target = self.waypoint = None
        self.src = ''
        self.dist, self.last_p = 0.0, None
        self.v = (0.0, 0.0)
        self.gave_up = ''

    def reacquired(self, run):
        """vision has had the robot for `run` frames in a row: end the episode"""
        if self.active and run >= self.frames:
            self.reset()
            return True
        return False

    def step(self, t, pose, last_seen, state, own_sign, odom_stale, dt, odom_bad=False, vision_down=False):
        """pose: odometry (x, y, th) or None; last_seen: the last vision pose (x, y); -> ((vfx, vfy), info)"""
        if not self.active:
            self.active, self.t0 = True, t
            lx, ly = last_seen
            self.target = choose_target(lx, ly, own_sign, self.vmap, self.margin, self.margin_map, self.min_rate, self.min_evidence)
            self.src = self.target[2]
            self.target = self.target[:2]
            self.waypoint = goal_waypoint(lx, ly, *self.target)
        info = dict(mode='recovering', target=[round(self.target[0], 2), round(self.target[1], 2)], src=self.src,
                    rec_t=round(t - self.t0, 2), rec_dist=round(self.dist, 2))
        if self.waypoint is not None:
            info['waypoint'] = [round(self.waypoint[0], 2), round(self.waypoint[1], 2)]
        if pose is not None:
            info['odom'] = [round(pose[0], 3), round(pose[1], 3), round(pose[2], 3)]
            if self.last_p is not None:
                self.dist += math.hypot(pose[0] - self.last_p[0], pose[1] - self.last_p[1])
            self.last_p = pose[:2]
        why = ''
        if state in (HALT, TIMEOUT) or state not in self.states:     # HALT and TIMEOUT whatever self.states holds
            why = f'state {state}'
        elif odom_stale or pose is None:
            why = 'odometry stale'
        elif odom_bad:
            why = 'odometry disagrees with vision'
        elif vision_down:
            why = 'vision sees nothing at all'
        elif self.gave_up:
            why = self.gave_up
        elif t - self.t0 > self.max_time:
            why = self.gave_up = f'gave up after {self.max_time:g} s'
        elif self.dist > self.max_dist:
            why = self.gave_up = f'gave up after {self.max_dist:g} m'
        if why:
            self.v = (0.0, 0.0)
            info['mode'] = f'recovering: standing still ({why})'
            return (0.0, 0.0), info
        x, y = pose[0], pose[1]
        goal = self.target
        if self.waypoint is not None:
            if math.hypot(self.waypoint[0] - x, self.waypoint[1] - y) < 0.1 or abs(y) > GOAL_W / 2 + 0.35:
                self.waypoint = None
            else:
                goal = self.waypoint
        ex, ey = goal[0] - x, goal[1] - y
        d = math.hypot(ex, ey)
        cap = min(self.speed, self.stop_speed) if state == STOP else self.speed
        if d < self.tol:
            self.v = (0.0, 0.0)
            info['mode'] = 'recovering: at the target, waiting for vision'
            return (0.0, 0.0), info
        sp = min(cap, self.gain * d)
        tvx, tvy = sp * ex / d, sp * ey / d
        # ramp the vector: change by at most accel * dt per tick (slowing down at once is allowed)
        pvx, pvy = self.v
        if math.hypot(tvx, tvy) > math.hypot(pvx, pvy):
            dvx, dvy = tvx - pvx, tvy - pvy
            dv, lim = math.hypot(dvx, dvy), self.accel * dt
            if dv > lim:
                tvx, tvy = pvx + dvx * lim / dv, pvy + dvy * lim / dv
        self.v = (tvx, tvy)
        info['v_field'] = [round(tvx, 2), round(tvy, 2)]
        return self.v, info
