"""Two-robot strategy node: one process, one vision frame, one referee state -> a command for each of our two robots.
The decisions are made by policy_server.py running a team bot (--bot duo_nk, params {"d_role": -1}); this node builds
the world in the team's attack frame with both robots, asks the server once per tick for both, and sends each robot its
own body-frame Twist on its own ROS domain, so that the two robots never hear each other's commands.

    ros2 run solo_strategy duo_node --ros-args -p team:=yellow -p ids:=[1,3] -p keeper:=1 -p domains:=[77,78]

Robots (start-up parameters, nothing is assumed about them):
    ids         [a, b]   our robots' vision ids (one id: one robot, the team plays as solo_nk)
    keeper      a        the vision id of the keeper (duo_nk slot 0); the game controller must be told the same id.
                         Live: `ros2 param set /duo_strategy keeper <id>` swaps the roles (after the game controller's
                         goalkeeper id was changed)
    domains     [77, 78] the ROS domain each robot's control stack listens on (-1: this process's own domain);
                         the robot is started with the same number (SOLO_DOMAIN=<n> tools/solo/robot_stack.sh start)
    cmd_topics  ['/cmd_vel', '/cmd_vel']   the topic each robot listens to, on its domain
    speed_caps  []       per robot speed cap (m/s); empty: speed_cap for both
    bench       []       live: vision ids that get zero and are off the field for the policy (a card, a robot taken
                         off): `ros2 param set /duo_strategy bench [3]`, back with `... bench "[]"`
                         (an integer array parameter: give [-1] for "nobody" if the empty list is refused)
Subscribes (on this process's domain):
    /vision/poses/<team>/robot<id>    our robots BY VISION ID: the vision connector must run with
                                      <team>_robot_count:=16, so that its topic number is the vision id itself (with a
                                      smaller count the number is a rank among the ids seen, and it shifts when a robot
                                      leaves the field or a ghost id appears)
    /vision/poses/<other team>/robot<0..2>, /vision/poses/ball, /referee/game_state   as solo_node
Publishes:
    each robot's cmd_topic on its domain             geometry_msgs/Twist, body frame, best effort (cmd_reliable: reliable)
    /duo/robot<id>/cmd_vel (this domain)              a copy of each robot's command, for the recorders and the page
    /solo/debug (this domain)                         std_msgs/String JSON, 5 Hz: the mode line and each robot's state

What each robot counts as (benched, gone, off field, unseen, play) is in duo.py. Only a robot in "play" is driven;
every other one gets zero. Zero for both: HALT, TIMEOUT and every state the policy does not act in, no referee for 1 s
(unless force_state), no ball seen, the policy server not answering or answering for the wrong number of robots, a
tick that raises, and at the end (Ctrl-C, SIGINT, SIGTERM): five zeros to each robot over 0.2 s. If this process dies
without them (SIGKILL), each robot's own command timeout (0.3 s) stops it.
A robot that is not in play is off the field for the policy only once it is benched, off the field, or unseen for
gone_after s (2): then the other plays alone (the keeper alone is exactly solo_nk; the field robot alone stays a field
robot and keeps out of our defense area: swap `keeper` to make it the keeper).
"""
import json
import math
import signal
import socket
import time

import rclpy
from geometry_msgs.msg import Pose2D, Twist
from pequi_msgs.msg import GameState
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String

from solo_strategy.duo import Limits, Robot, body_command, check_ids, slots
from solo_strategy.tracking import Track, wrap

HALT, STOP, RUNNING, FREE_KICK, KICKOFF, PENALTY, TIMEOUT, BALL_PLACEMENT = 1, 2, 3, 4, 5, 6, 7, 8
ASK = (STOP, RUNNING, FREE_KICK, KICKOFF, PENALTY, BALL_PLACEMENT)      # the states the policy server acts in
STATES = {'halt': HALT, 'stop': STOP, 'running': RUNNING}
PREPARE = (4, 5, 6, 7)
NEXT = {FREE_KICK: 1, PENALTY: 2, KICKOFF: 3}
TOUCH = 0.09 + 0.0215 + 0.03
L2, W2 = 2.25, 1.5


class DuoNode(Node):
    def __init__(self, own_domain):
        super().__init__('duo_strategy')
        d = self.declare_parameter
        d('team', 'blue'); d('ids', [0, 1]); d('keeper', -1); d('domains', [-1, -1]); d('cmd_topics', ['/cmd_vel', '/cmd_vel'])
        d('speed_caps', [0.0]); d('bench', [-1])
        d('rate_hz', 30.0); d('pose_timeout', 0.2); d('latency', 0.1)
        d('predict', True); d('blind_slow', 0.25); d('blind_limit', 0.5); d('blind_floor', 0.4); d('gone_after', 2.0); d('off_margin', 0.25)
        d('pred_after', 0.05); d('pred_delay', 0.1); d('pred_accel', 3.0); d('pred_vel_mix', 0.5); d('opp_hold', 0.5)
        d('pred_policy_vel', 'model')
        d('speed_cap', 1.0); d('angular_cap', 4.0); d('ramp_fwd', 1.7); d('ramp_side', 1.5); d('side', 'auto'); d('force_state', '')
        d('server_port', 47001); d('cmd_reliable', False); d('vision_prefix', '/vision/poses')
        g = lambda n: self.get_parameter(n).value
        self.own_domain = own_domain
        self.team = g('team')
        self.other = 'yellow' if self.team == 'blue' else 'blue'
        self.team_id = 2 if self.team == 'blue' else 1
        self.ids = [int(i) for i in g('ids')]
        self.domains = [int(v) for v in g('domains')]
        self.topics = list(g('cmd_topics'))
        keeper = int(g('keeper'))
        self.keeper = self.ids[0] if keeper == -1 else keeper
        self.bench = set(int(b) for b in g('bench') if int(b) != -1)
        why = check_ids(self.ids, self.domains, self.topics, self.keeper, self.bench, own_domain)
        if why:
            raise ValueError(why)
        caps = [float(v) for v in g('speed_caps') if float(v) > 0.0]
        if caps and len(caps) != len(self.ids):
            raise ValueError(f"speed_caps must give one cap per robot ({len(self.ids)}) or none, not {caps}")
        self.speed_cap = dict(zip(self.ids, caps)) if caps else {i: float(g('speed_cap')) for i in self.ids}
        self.angular_cap = float(g('angular_cap'))
        self.ramps = (float(g('ramp_fwd')), float(g('ramp_side')))
        self.dt = 1.0 / float(g('rate_hz'))
        self.latency = float(g('latency'))
        self.side, self.force_state = g('side'), g('force_state')
        self.lim = Limits(predict=bool(g('predict')), pose_timeout=float(g('pose_timeout')), pred_after=float(g('pred_after')),
                          blind_slow=float(g('blind_slow')), blind_limit=float(g('blind_limit')), blind_floor=float(g('blind_floor')),
                          gone_after=float(g('gone_after')), off_margin=float(g('off_margin')), half_length=L2, half_width=W2,
                          pred_policy_vel=g('pred_policy_vel'))
        self.opp_hold = float(g('opp_hold'))
        self.robots = {i: Robot(i, delay=float(g('pred_delay')), accel=float(g('pred_accel')), vel_mix=float(g('pred_vel_mix')))
                       for i in self.ids}
        self.ball = Track(base=0.08, v_max=8.0)
        self.gs, self.gs_time = None, 0.0
        self.t0 = time.monotonic()
        self.cmd_t, self.cmd_counter, self.prev_state = 0.0, None, None
        self.kick_pos = (0.0, 0.0)
        self.still_since, self.still_pos = 0.0, (0.0, 0.0)
        self.last_team = 0
        self.mode, self.fails, self.n, self.seq, self.errors = '', 0, 0, 0, 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.02)
        self.server = ('127.0.0.1', int(g('server_port')))
        qos = 10 if g('cmd_reliable') else qos_profile_sensor_data
        # each robot's command publisher, on the robot's own domain: a context (one DDS participant) per other domain.
        # Publishing needs no executor, so these nodes are never spun.
        self.contexts, self.pub = [], {}
        extra = {}
        for vid, dom, topic in zip(self.ids, self.domains, self.topics):
            if dom == -1 or dom == own_domain:
                node = self
            else:
                if dom not in extra:
                    ctx = rclpy.Context()
                    _init(ctx, dom)
                    extra[dom] = Node(f'duo_strategy_d{dom}', context=ctx)
                    self.contexts.append(ctx)
                node = extra[dom]
            self.pub[vid] = node.create_publisher(Twist, topic, qos)
        self.extra_nodes = list(extra.values())
        self.echo = {vid: self.create_publisher(Twist, f'/duo/robot{vid}/cmd_vel', qos_profile_sensor_data) for vid in self.ids}
        self.dbg = self.create_publisher(String, '/solo/debug', 10)
        vp = g('vision_prefix')
        for vid in self.ids:
            self.create_subscription(Pose2D, f'{vp}/{self.team}/robot{vid}',
                                     lambda m, r=self.robots[vid]: r.observe(self.now(), m.x, m.y, m.theta), 10)
        self.opp = [Track() for _ in range(3)]
        for i in range(3):
            self.create_subscription(Pose2D, f'{vp}/{self.other}/robot{i}', lambda m, i=i: self.opp[i].add(self.now(), m.x, m.y, m.theta), 10)
        self.create_subscription(Pose2D, f'{vp}/ball', lambda m: self.ball.add(self.now(), m.x, m.y), 10)
        self.create_subscription(GameState, '/referee/game_state', self.on_game_state, 10)
        self.add_on_set_parameters_callback(self.on_params)
        self.create_timer(self.dt, self.tick)
        where = ', '.join(f"id {v} on domain {own_domain if d == -1 else d} {t}" for v, d, t in zip(self.ids, self.domains, self.topics))
        self.get_logger().info(f"duo strategy up: team {self.team}, keeper id {self.keeper}, {where}; bench {sorted(self.bench) or '-'}; "
                               f"speed caps {self.speed_cap}; force_state {self.force_state or '-'}; side {self.side}; "
                               f"our robots read from {vp}/{self.team}/robot<vision id> (connector {self.team}_robot_count:=16)")

    # ---- live parameters ----
    def on_params(self, params):
        keeper, bench = self.keeper, set(self.bench)
        for p in params:
            if p.name == 'keeper':
                keeper = int(p.value)
            elif p.name == 'bench':
                bench = set(int(b) for b in p.value if int(b) != -1)
            elif p.name in ('ids', 'domains', 'cmd_topics', 'team', 'speed_caps', 'speed_cap', 'server_port'):
                return SetParametersResult(successful=False, reason=f"{p.name} is read at start only: restart the node")
        why = check_ids(self.ids, self.domains, self.topics, keeper, bench, self.own_domain)
        if why:
            return SetParametersResult(successful=False, reason=why)
        if keeper != self.keeper or bench != self.bench:
            self.get_logger().info(f"keeper id {self.keeper} -> {keeper}, bench {sorted(self.bench)} -> {sorted(bench)}")
        self.keeper, self.bench = keeper, bench
        return SetParametersResult(successful=True)

    def now(self):
        return time.monotonic() - self.t0

    def on_game_state(self, m):
        self.gs, self.gs_time = m, self.now()

    # ---- the tick ----
    def tick(self):
        now = self.now()
        for r in self.robots.values():
            r.prev_vf, r.last_vf = r.last_vf, (0.0, 0.0)          # every path that sends zero restarts the ramp
        cmds = {vid: (0.0, 0.0, 0.0) for vid in self.ids}
        try:
            info = self.decide(now, cmds)
        except Exception as e:                                    # a bug must not leave a command standing
            cmds = {vid: (0.0, 0.0, 0.0) for vid in self.ids}
            for r in self.robots.values():
                r.last_vf, r.last_w = (0.0, 0.0), 0.0
            self.errors += 1
            info = dict(mode=f'error in the tick: standing still ({type(e).__name__}: {e})'[:200], t=round(now, 2))
            if self.errors % 30 == 1:
                self.get_logger().error(info['mode'])
        for vid, (vx, vy, w) in cmds.items():
            m = Twist()
            m.linear.x, m.linear.y, m.angular.z = float(vx), float(vy), float(w)
            self.pub[vid].publish(m)
            self.echo[vid].publish(m)
            self.robots[vid].sent(now, vx, vy, w)
        self.n += 1
        if info['mode'] != self.mode:
            self.mode = info['mode']
            self.get_logger().info(f"mode: {self.mode}")
        if self.n % 6 == 0:
            self.dbg.publish(String(data=json.dumps(info)))

    def zeros(self, n=5, gap=0.05):
        for k in range(n):
            for vid in self.ids:
                self.pub[vid].publish(Twist())
                self.echo[vid].publish(Twist())
            if k < n - 1:
                time.sleep(gap)

    def decide(self, now, cmds):
        """fills cmds {vision id: (body vx, vy, w)} for the robots that play; -> the debug line"""
        gs = self.gs if self.gs is not None and now - self.gs_time < 1.0 else None
        mirror = self.side == 'positive'
        nxt = None
        if self.force_state:
            state, ours, command, counter = STATES.get(self.force_state, HALT), False, -1, -1
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
            bvx = bvy = 0.0
        if counter != self.cmd_counter or state != self.prev_state:
            if state != RUNNING or self.prev_state is None:
                self.kick_pos = (bx, by)
            self.cmd_counter, self.prev_state, self.cmd_t = counter, state, now
        if command in PREPARE:
            self.cmd_t = now
        if math.hypot(bx - self.still_pos[0], by - self.still_pos[1]) > 0.05 or state != RUNNING:
            self.still_pos, self.still_since = (bx, by), now
        order = slots(self.ids, self.keeper)
        views = {vid: self.robots[vid].view(now, self.lim, benched=vid in self.bench) for vid in order}
        robots = {str(vid): dict(status=v.status, keeper=vid == self.keeper, unseen=round(min(v.age, 99.0), 2)) for vid, v in views.items()}
        base = dict(t=round(now, 2), state=state, ours=bool(ours), mirror=mirror, robots=robots)
        if state not in ASK:
            return dict(base, mode=f'state {state}: both standing still')
        if not any(v.present for v in views.values()):
            return dict(base, mode='none of our robots on the field: standing still')
        if not ball_seen:
            return dict(base, mode='no ball seen yet: standing still')
        for v in views.values():
            if v.present and math.hypot(bx - v.pose[0], by - v.pose[1]) < TOUCH:
                self.last_team = 1
        opp = []
        for tr in self.opp:
            if self.lim.predict and tr.seen(now, self.opp_hold):
                ox, oy, oth, ovx, ovy = tr.coast(now, 0.1)
            elif tr.seen(now, self.lim.pose_timeout):
                ox, oy, oth, ovx, ovy = tr.state()
            else:
                continue
            if math.hypot(bx - ox, by - oy) < TOUCH:
                self.last_team = -1
            opp.append([sx * ox, oy, wrap(math.pi - oth) if mirror else oth, sx * ovx, ovy, 1])
        us = []
        for k, vid in enumerate(order):
            v = views[vid]
            if v.present:
                x, y, th, vx, vy = v.pose
                us.append([sx * x, y, wrap(math.pi - th) if mirror else th, sx * vx, vy, 1])
            else:
                us.append([-L2 - 1.0, 1.0 - k, 0.0, 0.0, 0.0, 0])       # off the field, as the server parks absent slots
        q = dict(t=now, state=state, ours=bool(ours), cmd_t=self.cmd_t, still_since=self.still_since, last_team=self.last_team,
                 kick_pos=[sx * self.kick_pos[0], self.kick_pos[1]], us=us, opp=opp, ball=[sx * bx, by, sx * bvx, bvy])
        if nxt is not None:
            q['next'] = [nxt[0], bool(nxt[1]), sx * bx, by]
        self.seq += 1
        q['seq'] = self.seq
        try:
            self.sock.sendto(json.dumps(q).encode(), self.server)
            r = json.loads(self.sock.recv(65535))
            while r.get('seq') != self.seq:
                r = json.loads(self.sock.recv(65535))
            self.fails = 0
        except (OSError, ValueError):
            self.fails += 1
            try:
                self.sock.setblocking(False); self.sock.recv(65535)
            except OSError:
                pass
            finally:
                self.sock.settimeout(0.02)
            for rb in self.robots.values():
                rb.last_w = 0.0
            return dict(base, mode='policy server does not answer: standing still')
        if not (isinstance(r.get('v'), list) and len(r['v']) == len(order) and all(isinstance(e, list) for e in r['v'])
                and isinstance(r.get('w'), list) and len(r['w']) == len(order) and int(r.get('team_size', 1)) >= len(order)):
            return dict(base, mode=f"policy server does not play {len(order)} robots (start it with --bot duo_nk): standing still; {str(r.get('mode'))[:80]}")
        modes = r.get('mode') if isinstance(r.get('mode'), list) else [r.get('mode')] * len(order)
        for k, vid in enumerate(order):
            v, rb = views[vid], self.robots[vid]
            robots[str(vid)]['policy'] = modes[k]
            if not v.drives:
                rb.last_w = 0.0
                continue
            vfx, vfy, w = sx * r['v'][k][0], r['v'][k][1], (-r['w'][k] if mirror else r['w'][k])
            cmds[vid] = body_command(rb, vfx, vfy, w, v.pose[2], v.scale, self.speed_cap[vid], self.angular_cap, self.latency, self.ramps, self.dt)
            robots[str(vid)].update(us=[round(c, 3) for c in v.pose[:3]], v_field=[round(vfx, 2), round(vfy, 2)],
                                    v_body=[round(cmds[vid][0], 2), round(cmds[vid][1], 2)], w=round(cmds[vid][2], 2), scale=round(v.scale, 2))
        line = '; '.join(f"{'K' if vid == self.keeper else 'F'}{vid} {views[vid].status}"
                         + (f" {modes[k]}" if views[vid].drives else '') for k, vid in enumerate(order))
        return dict(base, mode=f"{line} (state {state}{', our kick' if ours else ''})", opp=len(opp), ms=r.get('ms'))


def _stop_signal(signum, frame):
    raise KeyboardInterrupt


def _init(context, domain_id):
    """rclpy.init on a context and a domain, without rclpy's own signal handlers where the option exists (they shut
    the context down before the final zeros are published)"""
    try:
        from rclpy.signals import SignalHandlerOptions
    except ImportError:
        rclpy.init(context=context, domain_id=domain_id)
        return
    rclpy.init(context=context, domain_id=domain_id, signal_handler_options=SignalHandlerOptions.NO)


def main(args=None):
    import os
    own = int(os.environ.get('ROS_DOMAIN_ID', '0') or 0)
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        signal.signal(signal.SIGINT, _stop_signal)
        signal.signal(signal.SIGTERM, _stop_signal)
    except ImportError:
        rclpy.init(args=args)
    node = DuoNode(own)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.zeros()                        # zeros over 0.2 s to each robot: one lost message does not leave a command standing
        for n in node.extra_nodes:
            n.destroy_node()
        for ctx in node.contexts:
            rclpy.shutdown(context=ctx)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
