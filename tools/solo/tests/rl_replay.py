"""A trained policy in policy_server.py against the same policy in the strategy simulator: one simulated game per
colour is played with the checkpoint going through the simulator's sslenv.deploy, every tick is turned into the request
solo_node would send (their robots out of order, with vision ids 7, 3, 5 for the simulator's slots 0, 1, 2 and 7 the
game controller's keeper id for them), and a policy_server.py process started with the same checkpoint answers them. What the server
shows the policy must be what the simulator showed it: every entry of the observation frame that does not depend on
the commands sent before must agree within --tol. The other entries and the commands themselves agree only roughly:
the server is not in the loop (the simulated robot follows the simulator's commands, not the server's), and the two
compiled programs round differently, which the policy's own last command in its observation feeds back (seen: up to
0.1 m/s for a second or two, then back to 1e-4). Run with the simulator's Python, from the simulator checkout (no ROS,
no robot):

    cd <strategy simulator checkout> && PYTHONPATH=src:ssl-rules/src JAX_PLATFORMS=cpu SSL_ROBOTS=robots/three_wheel_fast.json \\
        .venv/bin/python <this repository>/tools/solo/tests/rl_replay.py <checkpoint> --robot three_wheel_fast

Exit 0 when those frame entries agree within --tol on every tick and the commands within --cmd-tol (m/s, rad/s)."""
import argparse
import dataclasses
import json
import os
import socket
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from sslenv import deploy
from sslenv.env import observe, reset, seen, step
from sslenv.league.entrants import load_entrant
from sslenv.league.match import match_cfg, stop_request_fn, swapped, tourney_cfg
from sslenv.scripted.frame import attack_view
from sslenv.train.ppo import env_cfg_of, load_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--robot", default="three_wheel_fast")
ap.add_argument("--opponent", default="scripted:chase")
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--port", type=int, default=47171)
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--cmd-tol", type=float, default=0.5)
ap.add_argument("--pitch", default="carpet")
args = ap.parse_args()
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src", "solo_strategy", "solo_strategy", "policy_server.py")

d = load_ckpt(args.ckpt)
own = env_cfg_of(d)
cfg = tourney_cfg(level="realistic", pitch=args.pitch)
cfg = dataclasses.replace(cfg, **{f: max(getattr(cfg, f), getattr(own, f)) for f in ("obs_v2", "stack", "obs_kicker", "obs_clocks")})
E = deploy.entrant(d, cfg, "real", "", "-", args.robot)
O = load_entrant("opp", args.opponent, cfg)
cfg = match_cfg(E, O, cfg)
N, tim = cfg.n, cfg.rules.timing
team = jnp.array([0, 1], jnp.int32)                      # game 0: the checkpoint plays blue; game 1: yellow
pick = lambda x, t: jnp.where((t == 0).reshape((-1,) + (1,) * (x.ndim - 2)), x[:, 0], x[:, 1])


@jax.jit
def tick(st, mA, mB, k):
    obs = jnp.clip(observe(st, cfg), -5.0, 5.0)
    sn = seen(st)
    ka, kb = jax.random.split(k)
    aA, mA = E.act(E.params, mA, pick(obs, team), sn, team, ka)
    aB, mB = O.act(O.params, mB, pick(obs, 1 - team), sn, 1 - team, kb)
    sel = (team == 0)[:, None, None]
    req = stop_request_fn(E, O, E.params, O.params, mA, mB, sn, team, cfg)
    av = attack_view(sn._replace(pv=None), team)        # what the node sees: vision poses, no latency compensation
    g = sn.game
    mine = lambda x: jnp.where((team == 1).reshape((-1,) + (1,) * (x.ndim - 1)), x[:, ::-1], x)
    ref = (mine(g.score), mine(g.fouls), mine(jnp.clip((g.yellow_until - g.stage_t[:, None, None]).max(-1), 0.0, tim.yellow_card_duration)))
    st, _, _, _, _ = step(st, jnp.stack([jnp.where(sel, aA, aB), jnp.where(sel, aB, aA)], 1), cfg, 0.0, stop_request=req, auto_reset=False)
    return st, mA, mB, av, ref, mA["out"], mA["bot"]["stack"][:, 0, 0]


key = jax.random.PRNGKey(3)
s0, s1 = reset(key, 1, cfg), reset(key, 1, swapped(cfg))
st = jax.tree.map(lambda x, y: jnp.concatenate([x, y]) if (x.ndim > 0 and x.shape[0] == 1) else x, s0._replace(key=None), s1._replace(key=None))._replace(key=s0.key)
mA, mB = E.init_mem(2, N), O.init_mem(2, N)
T = int(args.seconds * 30)
VID = (7, 3, 5)                                  # their vision ids by simulator slot: the keeper, then ascending
games, q_t = ([], []), [None, None]
FED = deploy.own_fed(cfg)                        # frame entries that depend on the commands sent before
f = lambda x: [float(v) for v in np.asarray(x).ravel()]
print(f"simulating 2 games of {args.seconds:.0f} s ...", flush=True)
for i, k in enumerate(jax.random.split(jax.random.fold_in(key, 1), T)):
    st, mA, mB, av, ref, out, frame = tick(st, mA, mB, k)
    av, ref, out, frame = jax.device_get((av, ref, out, frame))
    for b in (0, 1):
        q = dict(t=float(av.t[b]), state=int(av.state[b]), ours=bool(av.ours[b]), cmd_t=float(av.cmd_t[b]), still_since=float(av.still_since[b]),
                 last_team=int(av.last_team[b]), kick_pos=f(av.kick_pos[b]),
                 us=f(av.pos[b, 0]) + [float(av.th[b, 0])] + f(av.vel[b, 0]) + [int(av.act[b, 0])],
                 opp=[f(av.opos[b, j]) + [float(av.oth[b, j])] + f(av.ovel[b, j]) + [int(av.oact[b, j]), VID[j]] for j in (2, 0, 1)],
                 ball=f(av.ball[b]) + f(av.bvel[b]), ref=dict(score=[int(v) for v in ref[0][b]], fouls=[int(v) for v in ref[1][b]], cards=f(ref[2][b]), keeper=[0, 7]),
                 reset=i == 0, seq=i, frame=True, placed=bool(i > 0 and q_t[b] is not None and float(av.t[b]) - q_t[b] > 0.1))
        q_t[b] = float(av.t[b])
        games[b].append((q, out[b, 0, :2] * cfg.v_cap, float(out[b, 0, 2]) * cfg.w_cap, frame[b]))

env = dict(os.environ)
srv = subprocess.Popen([sys.executable, SERVER, "--ckpt", args.ckpt, "--robot", args.robot, "--port", str(args.port)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    for line in srv.stdout:
        if "listening" in line:
            break
    else:
        sys.exit("the policy server ended before it listened")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.settimeout(10.0)
    worst, bad, n, modes, cmds, placed = 0.0, 0, 0, {}, [], 0
    for b in (0, 1):
        for q, v, w, frame in games[b]:
            sock.sendto(json.dumps(q).encode(), ("127.0.0.1", args.port))
            r = json.loads(sock.recv(65535))
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
            placed += q["placed"]
            e = float(np.abs(np.asarray(r["frame"]) - frame)[~FED].max())
            if e > args.tol and bad < 5:
                j = int(np.argmax(np.where(FED, 0.0, np.abs(np.asarray(r["frame"]) - frame))))
                print(f"game {b} tick {q['seq']} state {q['state']}: frame entry {j} ({[p[0] for p in deploy.frame_parts(cfg) if p[1] <= j][-1]}) server {r['frame'][j]:.5f} simulator {frame[j]:.5f}")
            worst, bad, n = max(worst, e), bad + (e > args.tol), n + 1
            cmds.append(max(abs(r["v"][0] - v[0]), abs(r["v"][1] - v[1]), abs(r["w"] - w)))
finally:
    srv.terminate()
cmds = np.array(cmds)
print(f"{n} ticks ({placed} restarts), modes {modes}; frame entries that do not depend on earlier commands: largest difference {worst:.1e}, "
      f"beyond {args.tol:g} on {bad} ticks; commands (m/s, rad/s): median difference {np.median(cmds):.1e}, largest {cmds.max():.1e}")
sys.exit(1 if bad or cmds.max() > args.cmd_tol else 0)
