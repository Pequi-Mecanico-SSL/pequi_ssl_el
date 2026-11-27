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
and gets zero; in a stop no robot's target is nearer the ball than --stop-keep m (0.7), and each robot keeps --own-gap m
(0.35) from the other (centres) as well as 0.6 m from the ball. In BALL_PLACEMENT a request may carry "place": [x, y]
(the placement spot, attack frame): then the whole line from the ball to the spot is kept off as the ball is.
Parameters live (no restart, no recompile): a request {"cfg": "get"}, {"cfg": "set", "set": {key: number, ...}} or
{"cfg": "profile"} (every key back to the start value; "keys": [...] for some only) is answered with the bot's
parameters now in force, never with a robot command, and leaves the strategy's memory as it is. Every parameter is
a traced float32 scalar of the compiled program, so a new value takes effect on the next tick with no recompile (the
reply's "compiles" counts the program's compiled variants: it does not change). A set is refused whole ("error",
nothing changed) unless every key is a parameter of the bot (not d_role: one robot or the team is fixed at start)
and every value a finite number. The values in force
last as long as this process: a node restart or a request's reset keeps them; a new start of the server is back to
--params. Reply: {"bot", "profile": the --params keys at their start values, "now": the same keys and every changed
one at the values in force, "changed": [keys], "default": the bot's own value of each changed key not in --params,
"report": the strategy's per-robot-batch scalars of its memory now (duo_att: vo_m their fastest speed seen in play,
mode 0 cautious / 1 free / 2 pick, ...), "keys": every parameter name, "compiles", "error" when refused}. Each
accepted change is printed to the log with the clock time.
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
ap.add_argument("--stop-keep", type=float, default=0.7, help="two robots: in a stop no robot's target is nearer the ball than this, m (centres; the rule is 0.5 m + the robot's radius)")
ap.add_argument("--keep", default="", help="comma-separated top-level memory keys a request's reset leaves as they are (game-long "
                "knowledge the node's restart must not erase, e.g. duo_att's vo_m,vb_m,om: their fastest speeds seen in play); "
                "a restart of this server starts them over")
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
PROFILE = {k: float(v) for k, v in json.loads(args.params).items()}   # the start values, for the live settings
P0 = dict(P)                                                            # the start values of every key


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


def off_ball(target, ball, pos, keep):
    """a stop target nearer the ball than keep (centres) moved out to keep, along the ball -> target line (the ball ->
    robot line when the target is on the ball): stop_cmd's push off the ball then never has to fight the target"""
    d = target - ball
    dd = float(np.linalg.norm(d))
    if dd >= keep:
        return target
    if dd < 1e-6:
        d = pos - ball
        dd = max(float(np.linalg.norm(d)), 1e-6)
    return ball + d / dd * keep


def nearest(p, a, b):
    """the point of the segment a -> b nearest to p"""
    ab = b - a
    l2 = float(ab @ ab)
    t = 0.0 if l2 < 1e-12 else min(1.0, max(0.0, float((p - a) @ ab) / l2))
    return a + t * ab


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
        place = np.asarray(q["place"], float) if st == int(GS.BALL_PLACEMENT) and q.get("place") is not None else None
        out = dict(v=[], w=[], mode=[])
        for i, r in enumerate(us):
            if not seen[i]:
                out["v"].append([0.0, 0.0]); out["w"].append(0.0); out["mode"].append("robot not seen"); continue
            b, p_i = np.asarray(q["ball"][:2]), np.asarray(r[:2])
            if place is not None:                               # the corridor ball -> spot counts as the ball
                tgt = off_ball(tg[i], nearest(tg[i], b, place), p_i, args.stop_keep + 0.1)
                vv, w = stop_cmd({**q, "us": r, "ball": list(nearest(p_i, b, place))}, tgt, float(hd[i]))
                vv = keep_apart(vv, p_i, [nearest(p_i, b, place)], args.stop_keep + 0.05)   # the line moves with the ball: more margin
            else:
                vv, w = stop_cmd({**q, "us": r}, off_ball(tg[i], b, p_i, args.stop_keep), float(hd[i]))
            vv = keep_apart(vv, np.asarray(r[:2]), [o[:2] for j, o in enumerate(us) if j != i and seen[j]], args.own_gap)
            sp = float(np.linalg.norm(vv))
            if sp > args.stop_speed:
                vv = vv * args.stop_speed / sp
            out["v"].append([float(vv[0]), float(vv[1])]); out["w"].append(w); out["mode"].append("stop")
        return out, mem
    return dict(zero, mode=["halt"] * len(us)), mem


mem0 = mod.init_mem(1, N)
KEEP = [k for k in args.keep.split(",") if k]
missing = [k for k in KEEP if not isinstance(mem0, dict) or k not in mem0]
if missing:
    raise SystemExit(f"--keep: {missing} are not memory keys of {args.bot}")


def reset_mem(mem):
    """the start memory, with the --keep keys carried over from mem"""
    return {**mem0, **{k: mem[k] for k in KEEP}} if KEEP else mem0


FIXED = {"d_role"}           # what the server plays (one robot or the team) is set at start: run_duo.sh checks it there


def finite(v):
    """a finite JSON number (not a bool, not a string) as float, else None"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def g7(x):
    """a float32 value as the short decimal it was given as (0.35, not 0.3499999940395355)"""
    return float(f"{float(x):.7g}")


def cfg_view(error=None):
    """the reply to a cfg request: the parameters in force, the profile, the strategy's memory report"""
    changed = sorted(k for k in P if float(P[k]) != float(P0[k]))
    rep = {}
    if isinstance(mem, dict):
        for k, v in mem.items():                                   # the per-batch scalars (shape (1,)): the bot's state
            if hasattr(v, "shape") and tuple(v.shape) == (1,):
                rep[k] = round(float(v[0]), 3)
    out = dict(bot=args.bot, profile=PROFILE, now={k: g7(P[k]) for k in sorted(set(PROFILE) | set(changed))},
               changed=changed, default={k: g7(mod.PARAMS[k]) for k in changed if k not in PROFILE}, report=rep,
               keys=sorted(P), compiles=sum(f._cache_size() for f in (act, act_team, stop_pose, stop_pose_team)))
    if error:
        out["error"] = error
    return out


def cfg_request(q):
    """a cfg request ("get", "set", "profile"): checks it whole, changes P only when it is all valid"""
    global P
    op = q.get("cfg")
    if op == "get":
        return cfg_view()
    if op == "set":
        new = q.get("set")
        if not isinstance(new, dict) or not new:
            return cfg_view("set needs {key: number, ...}")
        bad = [k for k in new if k not in P]
        if bad:
            return cfg_view(f"not parameters of {args.bot}: {sorted(map(str, bad))}")
        bad = [k for k in new if k in FIXED]
        if bad:
            return cfg_view(f"fixed while the server runs (start it again to change): {sorted(bad)}")
        vals = {k: finite(v) for k, v in new.items()}
        bad = [k for k, v in vals.items() if v is None]
        if bad:
            return cfg_view(f"not finite numbers: {sorted(bad)}")
        upd = vals
    elif op == "profile":
        keys = q.get("keys", list(P))
        if not isinstance(keys, list) or any(k not in P for k in keys):
            return cfg_view("profile: keys must be a list of parameters")
        upd = {k: float(P0[k]) for k in keys}
    else:
        return cfg_view(f"unknown cfg request {op!r} (get, set, profile)")
    was = {k: g7(P[k]) for k in upd}
    P = {**P, **{k: jnp.float32(v) for k, v in upd.items()}}           # float32 scalars, as compiled: no recompile
    diff = [f"{k} {was[k]:g} -> {g7(P[k]):g}" for k in sorted(upd) if was[k] != g7(P[k])]
    print(time.strftime("%H:%M:%S") + " params " + (", ".join(diff) if diff else "unchanged") + f" ({op})", flush=True)
    return cfg_view()


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
WARM = (Q0, {**Q0, "state": int(GS.STOP), "next": [int(Next.FREE_KICK), 0, -1.0, 1.0]},
        {**Q0, "state": int(GS.STOP)},                                                    # a stop with no restart named
        {**Q0, "state": int(GS.BALL_PLACEMENT), "place": [-0.5, -0.4]}, {**Q0, "state": int(GS.FREE_KICK), "ours": True})
for wq in WARM:                         # each call also with the start memory: the loop starts from mem0, and a first stop
    STEP(wq, mem0)                      # before any play step would otherwise compile for it then (about 20 s)
for _ in range(3):                                              # the same calls the loop makes, until nothing recompiles
    for wq in WARM:
        _, m = STEP(wq, m)
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
        if isinstance(q, dict) and "cfg" in q:                  # the live settings: no robot command in the reply
            out = cfg_request(q)
            if "seq" in q:
                out["seq"] = q["seq"]
            sock.sendto(json.dumps(out).encode(), addr)
            continue
        if q.get("reset"):
            mem = reset_mem(mem)
        out, mem = (step_team if is_team(q) else step)(q, mem)
        if is_team(q):
            out["team_size"] = TEAM_SIZE                        # the client checks that the bot plays its number of robots
            if isinstance(mem, dict) and "vo_m" in mem:         # duo_att: the fastest opponent speed seen in play so far,
                out["their_v"] = round(float(mem["vo_m"][0]), 2)    # and whether it made the team cautious (a_nf 2)
                out["cautious"] = bool(float(P["a_nf"]) > 1.5 and float(mem["vo_m"][0]) >= float(P["a_nfv"]))
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
