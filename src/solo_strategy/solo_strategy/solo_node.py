"""One-robot strategy node: vision and referee topics in, cmd_vel out. The decisions are made by policy_server.py
(the simulator's scripted policy, a separate process: it needs the simulator's Python); this node builds the world in
the team's attack frame, asks the server once per tick and turns the answer into a body-frame Twist.

Subscribes:
    (vision_prefix, default /vision/poses)
    /vision/poses/<team>/robot<robot_index>      geometry_msgs/Pose2D   our robot (m, rad, field frame)
    /vision/poses/<other team>/robot<0..2>       geometry_msgs/Pose2D   their robots
    /vision/poses/ball                           geometry_msgs/Pose2D
    /referee/game_state                          pequi_msgs/GameState
Publishes:
    cmd_topic (default /<team>/robot<robot_index>/cmd_vel)   geometry_msgs/Twist, body frame (x forward, y left, z CCW)
    /solo/debug                                  std_msgs/String (JSON, 5 Hz)

Safety: zero velocity when the server does not answer, in HALT, and when our robot has not been seen for too long:
    predict false: unseen for pose_timeout -> zero (the robot stands while vision does not see it);
    predict true (default): while unseen for more than pred_after s, its pose is carried forward from the commands it was sent (tracking.py:
        pred_delay, pred_accel, pred_vel_mix); the speed and turn caps fall linearly from 1 at blind_slow s to
        blind_floor at blind_limit s, and from blind_limit on the command is zero. Their robots stay in the world for
        opp_hold s after their last sighting (moved on with their velocity for at most 0.1 s), and with ball_coast > 0
        (default 0) a ball unseen for under 0.2 s is moved on with its velocity for at most ball_coast s.
Without a referee on the network the robot stands still, unless force_state names a state (running, stop, halt): that is
for tests and for a field without a game controller.
side: which half we defend: auto (from the referee), negative or positive (field x)."""
import json
import math
import socket
import time

import rclpy
from geometry_msgs.msg import Pose2D, Twist
from pequi_msgs.msg import GameState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from solo_strategy.tracking import DeadReckoning, Track, wrap

HALT, STOP, RUNNING, FREE_KICK, KICKOFF, PENALTY = 1, 2, 3, 4, 5, 6
STATES = {'halt': HALT, 'stop': STOP, 'running': RUNNING}
PREPARE = (4, 5, 6, 7)                      # wire commands PREPARE_KICKOFF_* / PREPARE_PENALTY_*
NEXT = {FREE_KICK: 1, PENALTY: 2, KICKOFF: 3}       # the simulator's rules.Next; anything else: 4 (force start)
TOUCH = 0.09 + 0.0215 + 0.03


class SoloNode(Node):
    def __init__(self):
        super().__init__('solo_strategy')
        d = self.declare_parameter
        d('team', 'blue'); d('robot_index', 0); d('cmd_topic', ''); d('rate_hz', 30.0); d('pose_timeout', 0.2); d('latency', 0.1)
        d('predict', True); d('blind_slow', 0.25); d('blind_limit', 0.5); d('blind_floor', 0.4)
        d('pred_after', 0.05); d('pred_delay', 0.1); d('pred_accel', 3.0); d('pred_vel_mix', 0.5); d('opp_hold', 0.5); d('ball_coast', 0.0)
        d('pred_policy_vel', 'model')
        d('speed_cap', 1.0); d('angular_cap', 4.0); d('ramp_fwd', 1.7); d('ramp_side', 1.5); d('side', 'auto'); d('force_state', ''); d('server_port', 47001); d('cmd_reliable', False)
        d('vision_prefix', '/vision/poses')
        d('foe_all', False)                                        # shared test field: every robot but ours counts as one of theirs
        g = lambda n: self.get_parameter(n).value
        self.team = g('team')
        self.other = 'yellow' if self.team == 'blue' else 'blue'
        self.team_id = 2 if self.team == 'blue' else 1             # referee: 1 YELLOW, 2 BLUE
        idx = int(g('robot_index'))
        self.pose_timeout = float(g('pose_timeout'))
        self.speed_cap, self.angular_cap = float(g('speed_cap')), float(g('angular_cap'))
        self.ramp = (float(g('ramp_fwd')), float(g('ramp_side'))); self.dt = 1.0 / float(g('rate_hz'))
        self.last_vf = (0.0, 0.0); self.prev_vf = (0.0, 0.0)      # the field-frame velocity last sent (for the ramp)
        self.side, self.force_state = g('side'), g('force_state')
        self.predict = bool(g('predict'))
        self.blind_slow, self.blind_limit, self.blind_floor = float(g('blind_slow')), float(g('blind_limit')), float(g('blind_floor'))
        self.pred_policy_vel = g('pred_policy_vel')
        self.opp_hold, self.ball_coast, self.pred_after = float(g('opp_hold')), float(g('ball_coast')), float(g('pred_after'))
        self.dr = DeadReckoning(delay=float(g('pred_delay')), accel=float(g('pred_accel')), vel_mix=float(g('pred_vel_mix')))
        self.us, self.ball = Track(), Track(base=0.08, v_max=8.0)
        self.gs, self.gs_time = None, 0.0
        self.t0 = time.monotonic()
        self.cmd_t, self.cmd_counter, self.prev_state = 0.0, None, None
        self.kick_pos = (0.0, 0.0)
        self.still_since, self.still_pos = 0.0, (0.0, 0.0)
        self.last_team = 0
        self.mode, self.fails, self.n = '', 0, 0
        self.latency, self.last_w, self.seq = float(g('latency')), 0.0, 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.02)
        self.server = ('127.0.0.1', int(g('server_port')))
        vp = g('vision_prefix')
        topic = g('cmd_topic') or f'/{self.team}/robot{idx}/cmd_vel'
        # best effort: over Wi-Fi a reliable stream stalls on a lost packet and then delivers stale commands in a burst
        self.pub = self.create_publisher(Twist, topic, 10 if g('cmd_reliable') else qos_profile_sensor_data)
        self.dbg = self.create_publisher(String, '/solo/debug', 10)
        self.create_subscription(Pose2D, f'{vp}/{self.team}/robot{idx}', self.on_us, 10)
        srcs = [(self.other, i) for i in range(3)]
        if bool(g('foe_all')):                                     # both colours, ids 0..5, ours left out
            srcs = [(tm, j) for tm in (self.other, self.team) for j in range(6) if (tm, j) != (self.team, idx)]
        self.opp = [Track() for _ in srcs]
        for i, (tm, j) in enumerate(srcs):
            self.create_subscription(Pose2D, f'{vp}/{tm}/robot{j}',
                                     lambda m, i=i: self.opp[i].add(self.now(), m.x, m.y, m.theta), 10)
        self.create_subscription(Pose2D, f'{vp}/ball', lambda m: self.ball.add(self.now(), m.x, m.y), 10)
        self.create_subscription(GameState, '/referee/game_state', self.on_game_state, 10)
        self.create_timer(1.0 / float(g('rate_hz')), self.tick)
        self.get_logger().info(f'solo strategy up: team {self.team}, robot index {idx}, commands on {topic}, side {self.side}, '
                               f'force_state {self.force_state or "-"}, speed cap {self.speed_cap} m/s, ramp {self.ramp[0]} / {self.ramp[1]} m/s2, '
                               + (f'unseen: dead reckoning, slower from {self.blind_slow} s, zero from {self.blind_limit} s' if self.predict
                                  else f'unseen for {self.pose_timeout} s: zero'))

    def on_us(self, m):
        t = self.now()
        self.us.add(t, m.x, m.y, m.theta)
        self.dr.observe(t, m.x, m.y, m.theta, *self.us.vel)

    def now(self):
        return time.monotonic() - self.t0

    def on_game_state(self, m):
        self.gs, self.gs_time = m, self.now()

    def tick(self):
        now = self.now()
        out = Twist()
        self.prev_vf, self.last_vf = self.last_vf, (0.0, 0.0)      # every path that sends zero restarts the ramp
        info = self.decide(now, out)
        self.pub.publish(out)
        self.dr.command(now, out.linear.x, out.linear.y, out.angular.z)
        self.n += 1
        if info['mode'] != self.mode:
            self.mode = info['mode']
            self.get_logger().info(f"mode: {self.mode}")
        if self.n % 6 == 0:
            self.dbg.publish(String(data=json.dumps(info)))

    def _ramp(self, prev, tgt, a):
        """one axis: towards zero at once; away from zero by at most a * dt per tick (a <= 0: no ramp)"""
        if a <= 0.0 or abs(tgt) <= abs(prev) and prev * tgt >= 0.0:
            return tgt
        if prev * tgt < 0.0:
            prev = 0.0
        step = a * self.dt
        return prev + max(-step, min(step, tgt - prev))

    def decide(self, now, out):
        # ---- game state ----
        gs = self.gs if self.gs is not None and now - self.gs_time < 1.0 else None
        mirror = self.side == 'positive'
        nxt = None
        if self.force_state:
            state, ours, command = STATES.get(self.force_state, HALT), False, -1
            counter = -1
        elif gs is None:
            return dict(mode='no referee: standing still', t=round(now, 2))
        else:
            state, ours, command, counter = int(gs.state), gs.for_team == self.team_id and gs.state != RUNNING, int(gs.command), int(gs.command_counter)
            if self.side == 'auto':
                mirror = bool(gs.blue_team_on_positive_half) == (self.team == 'blue')
            if gs.has_next_state:
                nxt = (NEXT.get(int(gs.next_state), 4), gs.next_for_team == self.team_id)
        sx = -1.0 if mirror else 1.0
        ball_seen = bool(self.ball.q)
        bx, by, _, bvx, bvy = self.ball.state() if ball_seen else (0.0, 0.0, 0.0, 0.0, 0.0)
        if ball_seen and now - self.ball.q[-1][0] > 0.2:
            bvx = bvy = 0.0                                       # hidden (under a robot): where it was last seen, at rest
        elif ball_seen and self.predict and self.ball_coast > 0:  # a blink: where it has rolled to since
            a = min(now - self.ball.q[-1][0], self.ball_coast)
            bx, by = bx + bvx * a, by + bvy * a
        # the kick's clock: from the command that allows it (not from the PREPARE_* before a kickoff / penalty)
        if counter != self.cmd_counter or state != self.prev_state:
            if state != RUNNING or self.prev_state is None:
                self.kick_pos = (bx, by)
            self.cmd_counter, self.prev_state, self.cmd_t = counter, state, now
        if command in PREPARE:
            self.cmd_t = now
        # the no-progress clock and the last touch
        if math.hypot(bx - self.still_pos[0], by - self.still_pos[1]) > 0.05 or state != RUNNING:
            self.still_pos, self.still_since = (bx, by), now
        age = self.dr.age(now) if self.predict else self.us.age(now)
        scale = 1.0
        if self.predict:
            us_seen = age < self.blind_limit
            if us_seen and age <= self.pred_after:
                x, y, th, vx, vy = self.us.state()                # fresh: the vision pose, as without prediction
            elif us_seen:
                x, y, th, vx, vy = self.dr.predict(now)
                if self.pred_policy_vel == 'vision':           # the policy gets the last vision velocity, as without prediction
                    vx, vy = self.us.vel
                if age > self.blind_slow:
                    scale = 1.0 - (1.0 - self.blind_floor) * min(1.0, (age - self.blind_slow) / max(self.blind_limit - self.blind_slow, 1e-3))
        else:
            us_seen = self.us.seen(now, self.pose_timeout)
            if us_seen:
                x, y, th, vx, vy = self.us.state()
        if us_seen and math.hypot(bx - x, by - y) < TOUCH:
            self.last_team = 1
        opp = []
        for tr in self.opp:
            if self.predict and tr.seen(now, self.opp_hold):
                ox, oy, oth, ovx, ovy = tr.coast(now, 0.1)
            elif tr.seen(now, self.pose_timeout):
                ox, oy, oth, ovx, ovy = tr.state()
            else:
                continue
            if math.hypot(bx - ox, by - oy) < TOUCH:
                self.last_team = -1
            opp.append([sx * ox, oy, wrap(math.pi - oth) if mirror else oth, sx * ovx, ovy, 1])
        if len(opp) > 3:                                           # the policy takes three: those nearest the ball
            opp.sort(key=lambda o: math.hypot(sx * bx - o[0], by - o[1]))
            opp = opp[:3]
        if not us_seen:
            return dict(mode='our robot is not in vision: standing still', t=round(now, 2), state=state, unseen=round(min(age, 99.0), 2))
        if not ball_seen:
            return dict(mode='no ball seen yet: standing still', t=round(now, 2), state=state)
        q = dict(t=now, state=state, ours=bool(ours), cmd_t=self.cmd_t, still_since=self.still_since, last_team=self.last_team,
                 kick_pos=[sx * self.kick_pos[0], self.kick_pos[1]],
                 us=[sx * x, y, wrap(math.pi - th) if mirror else th, sx * vx, vy, 1], opp=opp, ball=[sx * bx, by, sx * bvx, bvy])
        if nxt is not None:
            q['next'] = [nxt[0], bool(nxt[1]), sx * bx, by]
        self.seq += 1
        q['seq'] = self.seq
        try:
            self.sock.sendto(json.dumps(q).encode(), self.server)
            r = json.loads(self.sock.recv(65535))
            while r.get('seq') != self.seq:                       # a late answer to an earlier tick: not this one's
                r = json.loads(self.sock.recv(65535))
            self.fails = 0
        except (OSError, ValueError):
            self.last_w = 0.0
            self.fails += 1
            try:                                                  # drop a late answer so the next one is not stale
                self.sock.setblocking(False); self.sock.recv(65535)
            except OSError:
                pass
            finally:
                self.sock.settimeout(0.02)
            return dict(mode='policy server does not answer: standing still', t=round(now, 2), state=state)
        # attack frame -> field frame -> body frame, capped
        vfx, vfy, w = sx * r['v'][0], r['v'][1], (-r['w'] if mirror else r['w'])
        sp, cap = math.hypot(vfx, vfy), self.speed_cap * scale
        if sp > cap:
            vfx, vfy = vfx * cap / sp, vfy * cap / sp
        w = max(-self.angular_cap * scale, min(self.angular_cap * scale, w))
        # the robot acts on this command about `latency` s after the pose was taken: turn the field velocity into the
        # body frame it will have then
        thp = th + self.last_w * self.latency
        c, s = math.cos(thp), math.sin(thp)
        bx, by = c * vfx + s * vfy, -s * vfx + c * vfy
        # ramp: the robot cannot follow a jump in the speed command (its wheels slip and it yaws), so a growing command is
        # raised at ramp_fwd / ramp_side m/s2 along the robot's forward / sideways axis; slowing down is not limited
        px, py = c * self.prev_vf[0] + s * self.prev_vf[1], -s * self.prev_vf[0] + c * self.prev_vf[1]
        bx, by = self._ramp(px, bx, self.ramp[0]), self._ramp(py, by, self.ramp[1])
        self.last_vf = (c * bx - s * by, s * bx + c * by)
        out.linear.x, out.linear.y = bx, by
        out.angular.z = max(-self.angular_cap, min(self.angular_cap, w))
        self.last_w = out.angular.z
        return dict(mode=f"{r['mode']} (state {state}{', our kick' if ours else ''})", t=round(now, 2), state=state, ours=bool(ours),
                    mirror=mirror, us=[round(v, 3) for v in (x, y, th)], ball=[round(bx, 3), round(by, 3)], opp=len(opp),
                    v_field=[round(vfx, 2), round(vfy, 2)], v_body=[round(bx, 2), round(by, 2)], w=round(out.angular.z, 2), cmd_age=round(now - self.cmd_t, 1), ms=r.get('ms'),
                    unseen=round(age, 3), scale=round(scale, 2))


def main(args=None):
    rclpy.init(args=args)
    node = SoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
