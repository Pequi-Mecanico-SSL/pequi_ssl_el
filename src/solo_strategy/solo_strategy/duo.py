"""The ROS-free part of duo_node.py (two robots, one strategy process): what each robot counts as for the policy and
for its command, and the turn of the policy's field-frame answer into one robot's body-frame command. No ROS imports:
the node uses it live, the tests (tools/solo/tests/test_duo.py) run it without ROS.

A robot is in exactly one of these conditions at each tick (Robot.view):
    benched     named in the node's bench parameter (a card, taken off by hand): zero command, absent for the policy
    gone        never seen, or not seen for gone_after s: zero command, absent for the policy
    off field   last seen more than off_margin m outside the field lines (being carried away): zero, absent
    unseen      not seen for blind_limit s (predict) or pose_timeout s (no predict), but for less than gone_after s:
                zero command; the policy still has it, at its last vision position, standing (a short camera dropout
                does not hand its role to the other robot and back)
    play        seen: the vision pose, or while unseen for pred_after .. blind_limit s the dead-reckoned one with the
                speed and turn caps lowered (blind_slow .. blind_limit), exactly as solo_node does for its robot
"""
import math
from dataclasses import dataclass

from solo_strategy.tracking import DeadReckoning, Track

BENCHED, GONE, OFF_FIELD, UNSEEN, PLAY = 'benched', 'gone', 'off field', 'unseen', 'play'


@dataclass
class Limits:
    predict: bool = True
    pose_timeout: float = 0.2
    pred_after: float = 0.05
    blind_slow: float = 0.25
    blind_limit: float = 0.5
    blind_floor: float = 0.4
    gone_after: float = 2.0
    off_margin: float = 0.45
    half_length: float = 2.25
    half_width: float = 1.5
    pred_policy_vel: str = 'model'


@dataclass
class View:
    status: str
    pose: tuple = None          # (x, y, th, vx, vy), field frame; None when absent
    scale: float = 1.0          # speed and turn caps x this (blind slow-down)
    age: float = float('inf')   # s since the last vision pose

    @property
    def present(self):
        """the policy counts this robot as on the field"""
        return self.status in (PLAY, UNSEEN)

    @property
    def drives(self):
        """this robot gets the policy's command (every other condition: zero)"""
        return self.status == PLAY


class Robot:
    """one of our robots: its vision track, its dead reckoning, and the command state its ramp needs"""
    def __init__(self, vid, delay=0.1, accel=3.0, vel_mix=0.5):
        self.vid = vid
        self.track = Track()
        self.dr = DeadReckoning(delay=delay, accel=accel, vel_mix=vel_mix)
        self.last_vf = (0.0, 0.0)       # the field-frame velocity last sent (for the ramp)
        self.prev_vf = (0.0, 0.0)
        self.last_w = 0.0

    def observe(self, t, x, y, th):
        self.track.add(t, x, y, th)
        self.dr.observe(t, x, y, th, *self.track.vel)

    def sent(self, t, vx, vy, w):
        self.dr.command(t, vx, vy, w)

    def view(self, now, lim, benched=False):
        age = self.dr.age(now) if lim.predict else self.track.age(now)
        if benched:
            return View(BENCHED, age=age)
        if not self.track.q or age >= lim.gone_after:
            return View(GONE, age=age)
        _, lx, ly, _ = self.track.q[-1]
        if abs(lx) > lim.half_length + lim.off_margin or abs(ly) > lim.half_width + lim.off_margin:
            return View(OFF_FIELD, age=age)
        limit = lim.blind_limit if lim.predict else lim.pose_timeout
        if age >= limit:
            x, y, th, _, _ = self.track.state()
            return View(UNSEEN, pose=(x, y, th, 0.0, 0.0), age=age)
        if not lim.predict or age <= lim.pred_after:
            return View(PLAY, pose=self.track.state(), age=age)
        x, y, th, vx, vy = self.dr.predict(now)
        if lim.pred_policy_vel == 'vision':
            vx, vy = self.track.vel
        scale = 1.0
        if age > lim.blind_slow:
            scale = 1.0 - (1.0 - lim.blind_floor) * min(1.0, (age - lim.blind_slow) / max(lim.blind_limit - lim.blind_slow, 1e-3))
        return View(PLAY, pose=(x, y, th, vx, vy), scale=scale, age=age)


def ramp(prev, tgt, a, dt):
    """one axis: towards zero at once; away from zero by at most a * dt per tick (a <= 0: no ramp). As solo_node."""
    if a <= 0.0 or abs(tgt) <= abs(prev) and prev * tgt >= 0.0:
        return tgt
    if prev * tgt < 0.0:
        prev = 0.0
    step = a * dt
    return prev + max(-step, min(step, tgt - prev))


def body_command(robot, vfx, vfy, w, th, scale, speed_cap, angular_cap, latency, ramps, dt):
    """field-frame velocity (vfx, vfy) and turn rate w for a robot at heading th -> its body-frame (x, y, w), capped,
    turned into the frame the robot will have `latency` s from now, and ramped along its own axes; updates the
    robot's last_vf / last_w. The same arithmetic as solo_node's play path, except that a non-finite input gives zero:
    the caps and the ramp would turn a NaN into a finite command (min / max keep the bound) that grows every tick."""
    if not all(math.isfinite(v) for v in (vfx, vfy, w, th, scale)):
        robot.last_vf, robot.last_w = (0.0, 0.0), 0.0
        return 0.0, 0.0, 0.0
    sp, cap = math.hypot(vfx, vfy), speed_cap * scale
    if sp > cap:
        vfx, vfy = vfx * cap / sp, vfy * cap / sp
    w = max(-angular_cap * scale, min(angular_cap * scale, w))
    thp = th + robot.last_w * latency
    c, s = math.cos(thp), math.sin(thp)
    bx, by = c * vfx + s * vfy, -s * vfx + c * vfy
    px, py = c * robot.prev_vf[0] + s * robot.prev_vf[1], -s * robot.prev_vf[0] + c * robot.prev_vf[1]
    bx, by = ramp(px, bx, ramps[0], dt), ramp(py, by, ramps[1], dt)
    robot.last_vf = (c * bx - s * by, s * bx + c * by)
    w = max(-angular_cap, min(angular_cap, w))
    robot.last_w = w
    return bx, by, w


def slots(ids, keeper):
    """the policy's slot order: the keeper first (duo_nk slot 0), then the others in the order given"""
    return [keeper] + [i for i in ids if i != keeper]


def check_ids(ids, domains, topics, keeper, bench, own_domain):
    """"" when the robot set-up parameters are consistent, else the reason. Two robots that would listen to the same
    topic on the same ROS domain are refused: that is the cross-talk this node exists to rule out."""
    ids, domains, topics, bench = list(ids), list(domains), list(topics), list(bench)
    if len(ids) < 1 or len(ids) > 2:
        return f"ids must name one or two vision ids, not {ids}"
    if len(set(ids)) != len(ids):
        return f"ids repeat a vision id: {ids}"
    if any(not 0 <= i <= 15 for i in ids):
        return f"vision ids are 0 .. 15: {ids}"
    if len(domains) != len(ids):
        return f"domains must give one ROS domain per robot ({len(ids)}), not {domains}"
    if any(not (d == -1 or 0 <= d <= 232) for d in domains):
        return f"a ROS domain is 0 .. 232 (or -1: this process's own): {domains}"
    if len(topics) != len(ids) or any(not t.startswith('/') for t in topics):
        return f"cmd_topics must give one absolute topic per robot ({len(ids)}), not {topics}"
    pairs = [(own_domain if d == -1 else d, t) for d, t in zip(domains, topics)]
    if len(set(pairs)) != len(pairs):
        return f"two robots would listen to the same topic on the same ROS domain: {pairs}"
    if keeper not in ids:
        return f"keeper {keeper} is not one of ids {ids}"
    if any(b not in ids for b in bench):
        return f"bench {bench} names a robot that is not one of ids {ids}"
    return ""


def load_live(path, team, ids, domains, topics, own_domain):
    """the keeper and bench a node with the same team and ids saved in `path` (save_live) -> (keeper, bench) or None
    (no file, unreadable, another set-up, or not consistent with check_ids). A restart of the node (the page's STRATEGY
    button) then goes on with the roles set live instead of the start-up ones."""
    import json
    if not path:
        return None
    try:
        with open(path) as f:
            s = json.load(f)
        if s.get('team') != team or [int(i) for i in s.get('ids', [])] != [int(i) for i in ids]:
            return None
        keeper, bench = int(s['keeper']), set(int(b) for b in s.get('bench', []))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if check_ids(ids, domains, topics, keeper, bench, own_domain):
        return None
    return keeper, bench


def save_live(path, team, ids, keeper, bench):
    """write the live keeper and bench for load_live (atomically: a node reading it never sees half a file)"""
    import json
    import os
    if not path:
        return
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, 'w') as f:
        json.dump(dict(team=team, ids=[int(i) for i in ids], keeper=int(keeper), bench=sorted(int(b) for b in bench)), f)
    os.replace(tmp, path)


def stack_problems(counts):
    """{ROS domain: robot control stacks heard there (publishers of the robots' wheels/state)} -> (domains whose
    robots must get zero, warning text). Two stacks on one domain means two robots hear the same commands: neither
    is driven. None on a domain is only a warning (a stack still starting, or a simulator without robot stacks)."""
    two = sorted(d for d, n in counts.items() if n > 1)
    none = sorted(d for d, n in counts.items() if n == 0)
    txt = []
    if two:
        txt.append(f"{', '.join(f'{counts[d]} robot stacks on domain {d}' for d in two)}: zero to the robot(s) on it")
    if none:
        txt.append(f"no robot stack heard on domain {', '.join(str(d) for d in none)}")
    return set(two), '; '.join(txt)
