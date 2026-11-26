"""The one-robot policy (sslenv.scripted.solo_nk) as a local UDP service.

Runs in the Python environment of the strategy simulator package (import name sslenv; JAX), next to the ROS node (solo_node.py), which sends it the world in the
team's ATTACK frame (our goal at -x) once per control tick and gets the robot's velocity command back.

    cd <strategy simulator checkout> && PYTHONPATH=src:ssl-rules/src JAX_PLATFORMS=cpu .venv/bin/python \
        <this repository>/src/solo_strategy/solo_strategy/policy_server.py [--params '{"nk_sp_x": -9}'] [--port 47001]

Request (JSON): {"t", "state", "ours", "cmd_t", "still_since", "last_team", "kick_pos": [x, y],
                 "us": [x, y, th, vx, vy, seen], "opp": [[x, y, th, vx, vy, seen] x 3], "ball": [x, y, vx, vy],
                 "next": [kind, ours, x, y] (STOP only: the restart that follows), "reset": bool}
Reply (JSON):   {"v": [vx, vy] attack-frame field velocity m/s, "w": rad/s (attack frame), "mode": text, "ms": compute time}
States are sslrules.GameState values. The policy acts in RUNNING / FREE_KICK / KICKOFF / PENALTY; in STOP the robot
goes to the team's restart pose (at --stop-speed at most) and keeps 0.6 m from the ball; in every other state it stands still."""
import argparse
import json
import socket
import time

import jax
import jax.numpy as jnp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bot", default="solo_nk")
ap.add_argument("--params", default="{}", help="JSON overrides of the bot's PARAMS")
ap.add_argument("--port", type=int, default=47001)
ap.add_argument("--robot", default="three_wheel")
ap.add_argument("--stop-speed", type=float, default=0.6, help="top speed while the game is stopped, m/s (the limit is 0.75)")
args = ap.parse_args()

import dataclasses
import importlib

from sslrules.state import GameState as GS
from sslenv.league.match import tourney_cfg
from sslenv.realism import robot as robot_cfg
from sslenv.rules import Next
from sslenv.scripted.frame import AV, restart_view

mod = importlib.import_module("sslenv.scripted." + args.bot)
cfg = dataclasses.replace(tourney_cfg(level="realistic"), robots=None, **robot_cfg(args.robot))
G = cfg.rules.geometry
L2, W2 = G.field_length / 2, G.field_width / 2
N = 3
P = {**mod.PARAMS, **{k: jnp.float32(v) for k, v in json.loads(args.params).items() if k in mod.PARAMS}}
unknown = [k for k in json.loads(args.params) if k not in mod.PARAMS]
if unknown:
    raise SystemExit(f"unknown params: {unknown}")


def make_av(q):
    f32 = lambda x: jnp.asarray(x, jnp.float32)
    us, opp, ball = q["us"], q["opp"], q["ball"]
    park = [[-L2 - 1.0, 1.0], [-L2 - 1.0, -1.0]]                      # our two absent robots, off the field
    pos = f32([[us[:2]] + park]); vel = f32([[us[3:5], [0, 0], [0, 0]]]); th = f32([[us[2], 0.0, 0.0]])
    act = jnp.asarray([[bool(us[5]), False, False]])
    opp = (opp + [[L2 + 1.0, 0.0, 0.0, 0.0, 0.0, 0]] * N)[:N]
    opos = f32([[o[:2] for o in opp]]); ovel = f32([[o[3:5] for o in opp]]); oth = f32([[o[2] for o in opp]])
    oact = jnp.asarray([[bool(o[5]) for o in opp]])
    return AV(pos=pos, vel=vel, th=th, act=act, opos=opos, ovel=ovel, oth=oth, oact=oact,
              ball=f32([ball[:2]]), bvel=f32([ball[2:4]]), state=jnp.asarray([int(q["state"])], jnp.int32),
              ours=jnp.asarray([bool(q["ours"])]), kick_pos=f32([q["kick_pos"]]), t=f32([q["t"]]), cmd_t=f32([q["cmd_t"]]),
              still_since=f32([q["still_since"]]), last_team=jnp.asarray([int(q["last_team"])], jnp.int32),
              team=jnp.zeros((1,), jnp.int32), charged=jnp.ones((1, N), bool))


@jax.jit
def act(P, mem, av):
    cmd, mem = mod.act(P, mem, av, cfg)
    return cmd[0, 0, :2] * cfg.v_cap, cmd[0, 0, 2] * cfg.w_cap, mem


@jax.jit
def stop_pose(P, mem, av, kind, kteam, spot):
    tg, hd, use = mod.stop_pose(P, mem, restart_view(av, kind, kteam, spot), cfg)
    return tg[0, 0], hd[0, 0]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def stop_cmd(q, target, heading):
    """drive to the restart pose (<= --stop-speed), never nearer than 0.6 m to the ball"""
    p, b = np.array(q["us"][:2]), np.array(q["ball"][:2])
    d = target - p
    dist = float(np.linalg.norm(d))
    v = d / max(dist, 1e-6) * min(args.stop_speed, np.sqrt(2 * 1.2 * dist), 3.0 * dist)
    rb = p - b
    db = float(np.linalg.norm(rb))
    if db < 0.6:
        v = v + rb / max(db, 1e-6) * 2.5 * (0.6 - db)
    sp = float(np.linalg.norm(v))
    if sp > args.stop_speed:
        v = v * args.stop_speed / sp
    return v, float(np.clip(5.0 * wrap(heading - q["us"][2]), -4.0, 4.0))


mem0 = mod.init_mem(1, N)
ACTS = (int(GS.RUNNING), int(GS.FREE_KICK), int(GS.KICKOFF), int(GS.PENALTY))
STOPS = (int(GS.STOP), int(GS.BALL_PLACEMENT))


def step(q, mem):
    st = int(q["state"])
    if not q["us"][5]:
        return dict(v=[0.0, 0.0], w=0.0, mode="robot not seen"), mem
    if st in ACTS:
        v, w, mem = act(P, mem, make_av(q))
        return dict(v=[float(v[0]), float(v[1])], w=float(w), mode="play"), mem
    if st in STOPS:
        kind, ours, sx, sy = q.get("next") or [int(Next.FORCE_START), 0, q["ball"][0], q["ball"][1]]
        tg, hd = stop_pose(P, mem, make_av(q), jnp.asarray([int(kind)], jnp.int32), jnp.asarray([0 if ours else 1], jnp.int32),
                           jnp.asarray([[sx, sy]], jnp.float32))
        vv, w = stop_cmd(q, np.asarray(tg), float(hd))
        return dict(v=[float(vv[0]), float(vv[1])], w=w, mode="stop"), mem
    return dict(v=[0.0, 0.0], w=0.0, mode="halt"), mem


Q0 = dict(t=0.0, state=int(GS.RUNNING), ours=False, cmd_t=0.0, still_since=0.0, last_team=0, kick_pos=[0.0, 0.0],
          us=[-L2 + 0.4, 0.0, 0.0, 0.0, 0.0, 1], opp=[[1.0, 0.5, 3.14, 0.0, 0.0, 1]], ball=[0.0, 0.0, 0.0, 0.0])
t0 = time.time()
print(f"compiling {args.bot} ...", flush=True)
m = mem0
for _ in range(3):                                              # the same calls the loop makes, until nothing recompiles
    _, m = step(Q0, m)
    _, m = step({**Q0, "state": int(GS.STOP), "next": [int(Next.FREE_KICK), 0, -1.0, 1.0]}, m)
t1 = time.perf_counter(); step(Q0, m); warm_ms = (time.perf_counter() - t1) * 1000
print(f"compiled in {time.time() - t0:.0f} s (a step now takes {warm_ms:.1f} ms); field {G.field_length} x {G.field_width} m; "
      f"listening on 127.0.0.1:{args.port}", flush=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", args.port))
mem = mem0
n = 0
while True:
    data, addr = sock.recvfrom(65535)
    t1 = time.perf_counter()
    q = None
    try:
        q = json.loads(data)
        if q.get("reset"):
            mem = mem0
        out, mem = step(q, mem)
    except Exception as e:                                      # a bad packet must not stop the server
        out = dict(v=[0.0, 0.0], w=0.0, mode=f"error: {e!r}"[:200])
    out["ms"] = round((time.perf_counter() - t1) * 1000, 2)
    if isinstance(q, dict) and "seq" in q:
        out["seq"] = q["seq"]
    sock.sendto(json.dumps(out).encode(), addr)
    n += 1
    if n % 300 == 1:
        print(f"{n} ticks; last: state {q.get('state') if isinstance(q, dict) else '?'} {out['mode']} v ({out['v'][0]:+.2f}, {out['v'][1]:+.2f}) "
              f"w {out['w']:+.2f} {out['ms']} ms", flush=True)
