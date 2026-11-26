"""Live dashboard: aggregates the stack's topics into one JSON frame and streams it to a web page.

    http://<host>:<port>/          the page (web/index.html)
    ws://<host>:<port>/ws          frames at ~20 Hz

Frame: field, ball, robots (both teams: pose, relayed cmd_vel), referee (game state + score/command), routers
(active behavior per team), and each team's active behavior debug (e.g. obstacles, targets, planned paths), from
whatever behaviors publish on /<team>/behaviors/<name>/debug (for the names in BEHAVIORS).
"""
import asyncio
import json
import threading
import time
from http import HTTPStatus
from pathlib import Path

import rclpy
import websockets
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose2D, Twist
from pequi_msgs.msg import ActiveBehavior, GameState, RefereeState
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

BEHAVIORS = ('stop', 'kickoff', 'free_kick', 'penalty', 'bench')


class Dashboard(Node):
    def __init__(self):
        super().__init__('dashboard')
        self.declare_parameters(namespace='', parameters=[('port', 8093), ('robot_count', 3), ('rate', 20)])
        gp = lambda n: self.get_parameter(n).get_parameter_value()
        self.port = gp('port').integer_value; self.n = gp('robot_count').integer_value
        self.frame = {'field': None, 'ball': None, 'robots': {'blue': {}, 'yellow': {}}, 'cmd': {'blue': {}, 'yellow': {}},
                      'game': None, 'referee': None, 'active': {}, 'debug': {}}
        self.debug_all = {'blue': {}, 'yellow': {}}      # team -> behavior name -> last debug message
        self.lock = threading.Lock()
        self.stamps = {}
        for color in ('blue', 'yellow'):
            for i in range(self.n):
                self.create_subscription(Pose2D, f'/vision/poses/{color}/robot{i}', lambda m, c=color, i=i: self._pose(m, c, i), 10)
                self.create_subscription(Twist, f'/{color}/robot{i}/cmd_vel', lambda m, c=color, i=i: self._cmd(m, c, i), 10)
            self.create_subscription(ActiveBehavior, f'/{color}/router/active', lambda m, c=color: self._active(m, c), 10)
            for b in BEHAVIORS:
                self.create_subscription(String, f'/{color}/behaviors/{b}/debug', lambda m, c=color: self._debug(m, c), 5)
        self.create_subscription(Pose2D, '/vision/poses/ball', self._ball, 10)
        self.create_subscription(Float32MultiArray, '/vision/field', self._field, 10)
        self.create_subscription(GameState, '/referee/game_state', self._game, 10)
        self.create_subscription(RefereeState, '/referee/state', self._referee, 10)
        self.html = (Path(get_package_share_directory('dashboard')) / 'web' / 'index.html').read_bytes()
        threading.Thread(target=self._serve, daemon=True).start()
        self.get_logger().info(f'dashboard on http://0.0.0.0:{self.port}')

    # ---- inputs -------------------------------------------------------------
    def _pose(self, m, c, i):
        with self.lock:
            self.frame['robots'][c][i] = [round(m.x, 3), round(m.y, 3), round(m.theta, 3)]; self.stamps[(c, i)] = time.time()

    def _cmd(self, m, c, i):
        with self.lock:
            self.frame['cmd'][c][i] = [round(m.linear.x, 3), round(m.linear.y, 3), round(m.angular.z, 3)]

    def _ball(self, m):
        with self.lock:
            self.frame['ball'] = [round(m.x, 3), round(m.y, 3)]; self.stamps['ball'] = time.time()

    def _field(self, m):
        with self.lock:
            self.frame['field'] = [float(x) for x in m.data]

    def _game(self, m):
        with self.lock:
            self.frame['game'] = {'state': m.state_name, 'for_team': m.for_team_name, 'ball_in_play': m.ball_in_play, 'command': int(m.command),
                                  'placement': [m.placement_x, m.placement_y] if m.has_placement_position else None, 'blue_positive': m.blue_team_on_positive_half}

    def _referee(self, m):
        with self.lock:
            self.frame['referee'] = {'command': m.command_name, 'stage': m.stage_name, 'score': [m.yellow.score, m.blue.score],
                                     'next': m.next_command_name if m.has_next_command else None, 'action_left': round(m.current_action_time_remaining, 1),
                                     'max_bots': [m.yellow.max_allowed_bots, m.blue.max_allowed_bots], 'events': [e.type for e in m.game_events]}

    def _active(self, m, c):
        with self.lock:
            self.frame['active'][c] = {'behavior': m.behavior, 'id': m.activation_id, 'ours': m.ours, 'benched': list(m.benched)}

    def _debug(self, m, c):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        with self.lock:
            self.debug_all[c][d.get('behavior', '?')] = d; self.stamps[('debug', c, d.get('behavior', '?'))] = time.time()

    # ---- server ---------------------------------------------------------------
    def snapshot(self):
        now = time.time()
        with self.lock:
            f = json.loads(json.dumps(self.frame))
            for c in ('blue', 'yellow'):
                # the active behavior's debug (per the router), plus bench for the display of benched robots
                act = (f['active'].get(c) or {}).get('behavior')
                d = self.debug_all[c].get(act)
                if d is not None:
                    d = dict(d); d['stale'] = now - self.stamps.get(('debug', c, act), 0) > 0.5; f['debug'][c] = d
                else:
                    f['debug'].pop(c, None)
                # benched robots are driven by `bench` whatever the active behavior: show bench's plan for them
                benched = set((f['active'].get(c) or {}).get('benched', []))
                b = self.debug_all[c].get('bench')
                if b is not None and act != 'bench' and benched and f['debug'].get(c):
                    keep = [p for p in f['debug'][c]['plans'] if p['i'] not in benched]
                    for p in b['plans']:
                        if p['i'] in benched:
                            p = dict(p); p['kind'] = 'bench: ' + p['kind']; keep.append(p)
                    f['debug'][c]['plans'] = sorted(keep, key=lambda p: p['i'])
        for c in ('blue', 'yellow'):
            f['robots'][c] = {i: p for i, p in f['robots'][c].items() if now - self.stamps.get((c, int(i)), 0) < 0.5}
        f['t'] = now
        return f

    def _serve(self):
        async def handler(ws):
            period = 1.0 / self.get_parameter('rate').get_parameter_value().integer_value
            try:
                while True:
                    await ws.send(json.dumps(self.snapshot(), separators=(',', ':')))
                    await asyncio.sleep(period)
            except websockets.ConnectionClosed:
                pass

        async def process_request(connection, request):
            if request.path.startswith('/ws'):
                return None                                   # proceed with the websocket handshake
            resp = connection.respond(HTTPStatus.OK, self.html.decode())
            resp.headers['Content-Type'] = 'text/html; charset=utf-8'
            resp.headers['Cache-Control'] = 'no-store'
            return resp

        async def main():
            async with websockets.serve(handler, '0.0.0.0', self.port, process_request=process_request, max_size=None):
                await asyncio.Future()
        asyncio.run(main())


def main(args=None):
    rclpy.init(args=args)
    node = Dashboard()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
