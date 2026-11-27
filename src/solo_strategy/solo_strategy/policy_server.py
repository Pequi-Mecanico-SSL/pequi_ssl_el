"""The one-robot policy (sslenv.scripted.solo_nk) as a local UDP service.

Runs in the Python environment of the strategy simulator package (import name sslenv; JAX), next to the ROS node (solo_node.py), which sends it the world in the
team's ATTACK frame (our goal at -x) once per control tick and gets the robot's velocity command back.

    cd <strategy simulator checkout> && PYTHONPATH=src:ssl-rules/src JAX_PLATFORMS=cpu .venv/bin/python \
        <this repository>/src/solo_strategy/solo_strategy/policy_server.py [--params '{"nk_sp_x": -9}'] [--port 47001]

Request (JSON): {"t", "state", "ours", "cmd_t", "still_since", "last_team", "kick_pos": [x, y],
                 "us": [x, y, th, vx, vy, seen], "opp": [[x, y, th, vx, vy, seen] x 3], "ball": [x, y, vx, vy],
                 "next": [kind, ours, x, y] (STOP only: the restart that follows), "reset": bool}
Reply (JSON):   {"v": [vx, vy] attack-frame field velocity m/s, "w": rad/s (attack frame), "mode": text, "ms": compute time}
Two robots (a team bot such as duo_nk with d_role -1): "us" is a list of robots, [[x, y, th, vx, vy, seen], ...] in
the team's slot order (duo_nk: slot 0 the keeper, slot 1 the field robot); the reply then has "v": [[vx, vy], ...],
"w": [w, ...] and "mode" for each robot in the same order, and "team_size", the number of robots the bot plays (its TEAM_SIZE). A robot not seen (seen 0) is off the field for the policy
and gets zero; in a stop each robot keeps --own-gap m from the other (centre to centre) as well as 0.6 m from the ball.
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
ap.add_argument("--own-gap", type=float, default=0.35, help="two robots: in a stop each keeps this far from the other, m (centres)")
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
    if is_team(q):                                                  # our robots in slot order, the rest parked off the field
        park = [[-L2 - 1.0, 1.0 - k] for k in range(N)]
        team = (us + [park[k] + [0.0, 0.0, 0.0, 0] for k in range(len(us), N)])[:N]
        pos = f32([[r[:2] for r in team]]); vel = f32([[r[3:5] for r in team]]); th = f32([[r[2] for r in team]])
        act = jnp.asarray([[bool(r[5]) for r in team]])
    else:
        park = [[-L2 - 1.0, 1.0], [-L2 - 1.0, -1.0]]                  # our two absent robots, off the field
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


def is_team(q):
    """a request for several robots: "us" is a list of robots"""
    return bool(q["us"]) and isinstance(q["us"][0], list)


@jax.jit
def act_team(P, mem, av):
    cmd, mem = mod.act(P, mem, av, cfg)
    return cmd[0, :, :2] * cfg.v_cap, cmd[0, :, 2] * cfg.w_cap, mem


@jax.jit
def stop_pose_team(P, mem, av, kind, kteam, spot):
    tg, hd, use = mod.stop_pose(P, mem, restart_view(av, kind, kteam, spot), cfg)
    return tg[0], hd[0]


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


def keep_apart(v, p, others, gap):
    """add a push away from each of our other robots nearer than gap (centres), as stop_cmd does for the ball"""
    for o in others:
        d = p - np.asarray(o)
        dd = float(np.linalg.norm(d))
        if dd < gap:
            v = v + d / max(dd, 1e-6) * 2.5 * (gap - dd)
    return v


def step_team(q, mem):
    """several robots: the same states and stop rules as step(), per robot; a robot not seen gets zero"""
    st, us = int(q["state"]), q["us"]
    seen = [bool(r[5]) for r in us]
    zero = dict(v=[[0.0, 0.0] for _ in us], w=[0.0 for _ in us])
    if not any(seen):
        return dict(zero, mode=["robot not seen"] * len(us)), mem
    if st in ACTS:
        v, w, mem = act_team(P, mem, make_av(q))
        v, w = np.asarray(v), np.asarray(w)
        return dict(v=[[float(v[i, 0]), float(v[i, 1])] if seen[i] else [0.0, 0.0] for i in range(len(us))],
                    w=[float(w[i]) if seen[i] else 0.0 for i in range(len(us))],
                    mode=["play" if s else "robot not seen" for s in seen]), mem
    if st in STOPS:
        kind, ours, sx, sy = q.get("next") or [int(Next.FORCE_START), 0, q["ball"][0], q["ball"][1]]
        tg, hd = stop_pose_team(P, mem, make_av(q), jnp.asarray([int(kind)], jnp.int32), jnp.asarray([0 if ours else 1], jnp.int32),
                                jnp.asarray([[sx, sy]], jnp.float32))
        tg, hd = np.asarray(tg), np.asarray(hd)
        out = dict(v=[], w=[], mode=[])
        for i, r in enumerate(us):
            if not seen[i]:
                out["v"].append([0.0, 0.0]); out["w"].append(0.0); out["mode"].append("robot not seen"); continue
            vv, w = stop_cmd({**q, "us": r}, tg[i], float(hd[i]))
            vv = keep_apart(vv, np.asarray(r[:2]), [o[:2] for j, o in enumerate(us) if j != i and seen[j]], args.own_gap)
            sp = float(np.linalg.norm(vv))
            if sp > args.stop_speed:
                vv = vv * args.stop_speed / sp
            out["v"].append([float(vv[0]), float(vv[1])]); out["w"].append(w); out["mode"].append("stop")
        return out, mem
    return dict(zero, mode=["halt"] * len(us)), mem


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
TEAM_SIZE = int(getattr(mod, "TEAM_SIZE", 1))
TEAM = TEAM_SIZE > 1                       # a team bot: the requests name several robots
if TEAM:
    Q0 = {**Q0, "us": [Q0["us"], [-0.5, 0.3, 0.0, 0.0, 0.0, 1]]}
STEP = step_team if TEAM else step
for _ in range(3):                                              # the same calls the loop makes, until nothing recompiles
    _, m = STEP(Q0, m)
    _, m = STEP({**Q0, "state": int(GS.STOP), "next": [int(Next.FREE_KICK), 0, -1.0, 1.0]}, m)
t1 = time.perf_counter(); STEP(Q0, m); warm_ms = (time.perf_counter() - t1) * 1000
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
        out, mem = (step_team if is_team(q) else step)(q, mem)
        if is_team(q):
            out["team_size"] = TEAM_SIZE                        # the client checks that the bot plays its number of robots
    except Exception as e:                                      # a bad packet must not stop the server
        n_us = len(q["us"]) if isinstance(q, dict) and isinstance(q.get("us"), list) and q["us"] and isinstance(q["us"][0], list) else 0
        out = dict(v=[[0.0, 0.0]] * n_us, w=[0.0] * n_us, mode=[f"error: {e!r}"[:200]] * n_us) if n_us else dict(v=[0.0, 0.0], w=0.0, mode=f"error: {e!r}"[:200])
    out["ms"] = round((time.perf_counter() - t1) * 1000, 2)
    if isinstance(q, dict) and "seq" in q:
        out["seq"] = q["seq"]
    sock.sendto(json.dumps(out).encode(), addr)
    n += 1
    if n % 300 == 1:
        if isinstance(out["w"], list):
            print(f"{n} ticks; last: state {q.get('state') if isinstance(q, dict) else '?'} {out['mode']} v {out['v']} w {out['w']} {out['ms']} ms", flush=True)
        else:
            print(f"{n} ticks; last: state {q.get('state') if isinstance(q, dict) else '?'} {out['mode']} v ({out['v'][0]:+.2f}, {out['v'][1]:+.2f}) "
                  f"w {out['w']:+.2f} {out['ms']} ms", flush=True)
