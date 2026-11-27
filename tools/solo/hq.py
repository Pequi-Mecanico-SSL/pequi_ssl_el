#!/usr/bin/env python3
"""Field page: a top-down view of the field in the browser; click a point and the robot drives there.
    source tools/solo/env.sh; python3 tools/solo/hq.py <team> [--port 8700] [--bind 127.0.0.1] [--speed 0.6]
Then open http://localhost:8700 . Needs the vision connector running (/vision/poses/...).
    click            drive to that point, keeping the current heading
    click and drag   drive to the point and turn to face along the drag
    STOP / space / Esc   stop at once
The page is the dead man's switch: when it stops asking for the state for 1.5 s (tab closed, computer asleep), the
target is dropped and the robot stops. Only the page's own requests (GET /state?page=1) count; a plain GET /state, as
other readers send, does not keep a target or a placement alive. Without a fresh pose from vision (0.2 s) the robot is not driven.
It refuses to drive while another program of ours that commands the robot is running (the strategy node, a test):
checked at start, on each click and once a second; a running target or placement is dropped when one appears. It looks
only at this computer's processes.
The target is dropped, and a placement ends, when our robot has not been seen for 2 s (neither resumes when vision
sees it again).
Requests: every POST must carry Content-Type: application/json, else 415 and nothing changes (no plain cross-site form
posts). A request whose Host header is not 127.0.0.1:<port>, localhost:<port> or the --bind address with the port gets
403 (GET too). Malformed JSON or a body over 4096 bytes gets 400; an unknown POST path 404. Numbers (x, y, th, v, tol)
must be finite JSON numbers and flags (on, precise) JSON true/false, else 400 and nothing changes. While locked, every
POST but /stop and /lock gets 423.
LOCK (on at start): while locked the page only shows; clicks on the field, the strategy button and the speed slider do
nothing, here and in the server. Unlocking takes two clicks within 3 s. STOP works whether locked or not.
The top lines show the referee: the derived state from the referee connector, and, read straight from the game
controller's multicast packets (--referee 224.5.23.1:10003 on the interface $SOLO_IFACE), the stage and time left, the
command, the score, fouls, cards, timeouts, goalkeepers and the last game events.
EVENTS opens a scrolling log (newest first): referee commands, stage changes, goals, fouls and cards, game events,
the strategy's mode changes, our robot lost from and back in vision, and what was done on this page. The log is also
appended to $SOLO_DATA/hq_events_<date>.jsonl (default <repo>/log/solo).
CAMERA MAP switches an underlay on and off: the coverage map of tools/solo/vision_map.py, each 0.1 m cell coloured by
how often the overhead camera detected a robot standing there (green at 90 % and more, through yellow, to red at 30 %
and less; grey: no evidence yet). It is read from the map file every 5 s, so a map being updated live shows as it grows.
The STRATEGY button stops the strategy node (the robot can then be placed by clicking) and starts it again with the
arguments it had; the policy server keeps running in between, so the restart takes a second or two. It starts the
node only from a saved line of exactly the node's form (procs.check_node_cmd: an absolute python3, an absolute
solo_node, --ros-args, -p pairs of the node's own parameters), and not while another commanding program runs (409);
a refusal and its reason go to the note and the event log.
Two robots (tools/solo/run_duo.sh, strategy node duo_node): the button stops and restarts duo_node the same way (both
robots get zeros when it stops). The page's own robot is SOLO_ROBOT_INDEX (run_duo.sh starts the connector with our
colour's count 16, so set it to the keeper's vision id); the other robot is drawn as <colour><vision id>. A click
drives only the robot that listens on the page's command topic in the page's ROS domain.
Speed: the slider (at most --max-speed, default 1.5 m/s); the commanded velocity is ramped at 1.5 m/s2.
"click = place ball" (the mode button; "click = drive robot" is the default): a click sets the spot where the ball
should go and the robot pushes it there with BallPlacer from place_ball.py (rough 0.15 m or precise 0.05 m; the slider
speed is its transit speed, at most 1.0 m/s; its carry speed 0.35 m/s, at most --max-speed); its body-frame command is published as it is. The page shows its status,
the spot with its tolerance, the line-up point, and the result (POST /place {x, y, precise | tol}; /state "place").
STOP, LOCK, the dead man's switch, the 0.2 s pose rule and the busy check apply as for a target; a placement also ends
when the robot leaves the target box widened by 0.25 m. Without place_ball.py the mode is off and the rest works.
"""
import json, math, os, signal, socket, subprocess, sys, threading, time, warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D, Twist
from std_msgs.msg import String
try:
    from pequi_msgs.msg import GameState                 # the referee state, when the referee connector runs
except ImportError:
    GameState = None
if os.path.dirname(os.path.abspath(__file__)) not in sys.path: sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import procs                                             # the commanding programs, the node's command line check
try:                                                     # the ball placer; without it the page works, ball mode off
    from place_ball import BallPlacer
    PLACE_ERR = ""
except Exception as e:
    BallPlacer = None; PLACE_ERR = f"place_ball.py not loaded ({type(e).__name__}: {e})"

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
PORT, BIND = int(opt("--port", "8700")), opt("--bind", "127.0.0.1")
REFEREE = opt("--referee", "224.5.23.1:10003")
VMAX = float(opt("--max-speed", "1.5"))
if not (math.isfinite(VMAX) and VMAX > 0):
    sys.exit("--max-speed must be a positive number")
SPEED0 = float(opt("--speed", "0.6"))
SPEED0 = max(0.1, min(VMAX, SPEED0 if math.isfinite(SPEED0) else 0.1))
if not args or args[0] not in ("blue", "yellow"):
    sys.exit(__doc__)
TEAM = args[0]; IDX = int(os.environ.get("SOLO_ROBOT_INDEX", "0"))
LX, LY = 2.25, 1.5                                  # half field (m)
BOX = (-LX - 0.10, LX + 0.10, -LY - 0.10, LY + 0.10)   # targets are kept inside this
ACC = 1.5                                           # m/s2 ramp of the commanded velocity
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi

S = dict(us=None, us_t=0.0, ball=None, ball_t=0.0, others={}, target=None, speed=SPEED0, poll=0.0,
         note="", cmd=(0.0, 0.0, 0.0), ref=None, ref_t=0.0, gc=None, gc_t=0.0, lock=True, place=None)
LOCK = threading.Lock()
DATA = os.environ.get("SOLO_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "log", "solo")
EVENTS = []; ELOCK = threading.Lock()
def event(kind, text):
    """one line of the event log (kept in memory for the page, appended to a file)"""
    e = dict(n=0, t=round(time.time(), 2), clock=time.strftime("%H:%M:%S"), kind=kind, text=text)
    with ELOCK:
        e["n"] = len(EVENTS); EVENTS.append(e)
        try:
            os.makedirs(DATA, exist_ok=True)
            with open(os.path.join(DATA, time.strftime("hq_events_%Y%m%d.jsonl")), "a") as f: f.write(json.dumps(e) + "\n")
        except OSError:
            pass
def note(text):
    """the status note of the page; a new one also goes to the event log (call with LOCK held)"""
    if text and text != S["note"]: event("page", text)
    S["note"] = text
now = time.monotonic

# ---- request checks (pure functions) --------------------------------------------------------------------------------
def num(v):
    """a finite JSON number (not a bool, not a string) as float, else None"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None

def check_goto(b):
    """/goto body -> (ok, x, y, th or None); th None: keep the current heading"""
    if not isinstance(b, dict):
        return False, None, None, None
    x, y = num(b.get("x")), num(b.get("y"))
    th = num(b["th"]) if "th" in b else None
    if x is None or y is None or ("th" in b and th is None):
        return False, None, None, None
    return True, x, y, th

def check_speed(b):
    """/speed body -> (ok, v)"""
    v = num(b.get("v")) if isinstance(b, dict) else None
    return v is not None, v

def check_flag(b, key, default=None):
    """a JSON true/false under key -> (ok, value); when absent: (True, default), or not ok when there is no default"""
    if not isinstance(b, dict):
        return False, None
    if key not in b:
        return default is not None, default
    return (True, b[key]) if isinstance(b[key], bool) else (False, None)

def host_ok(host, port, bind):
    ok = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if bind != "127.0.0.1":
        ok.add(f"{bind}:{port}")
    return (host or "").strip().lower() in ok

# ---- ball placement (click = place ball): small pure functions, the placer itself is place_ball.BallPlacer ----------
TOL_ROUGH, TOL_PRECISE = 0.15, 0.05
PLACE_MARGIN = 0.25                                 # the placer's margin beyond the lines; the robot box grows by it
PLACE_CARRY = 0.35                                  # the placer's carry speed (m/s), at most --max-speed
def place_request(b):
    """the body of POST /place -> ((x, y, tol), "") or (None, why). precise: true -> 0.05 m, else tol (0.02..0.5), 0.15.
    x, y and tol must be finite JSON numbers, precise true or false"""
    if not isinstance(b, dict) or "x" not in b or "y" not in b: return None, "bad request"
    ok, precise = check_flag(b, "precise", False)
    if not ok: return None, "bad request: precise must be true or false"
    x, y = num(b["x"]), num(b["y"])
    tol = TOL_PRECISE if precise else (num(b["tol"]) if "tol" in b else TOL_ROUGH)
    if x is None or y is None or tol is None: return None, "bad request: not a finite number"
    if not 0.02 <= tol <= 0.5: return None, "bad request: tol outside 0.02..0.5 m"
    return (x, y, tol), ""
def place_speed(v):
    """the slider speed as the placer's transit speed (place_ball.py accepts 0.1..1.0)"""
    return max(0.1, min(1.0, v))
def in_place_box(x, y, margin=PLACE_MARGIN):
    """is the robot inside the target box of the go-to targets, extended by the placer's margin?"""
    return BOX[0] - margin <= x <= BOX[1] + margin and BOX[2] - margin <= y <= BOX[3] + margin
def loop_mode(target, place):
    """what the 50 Hz loop runs: "place" (an active placement), "goto" (a drive target) or "idle" """
    if place is not None and not place["pl"].finished: return "place"
    return "goto" if target is not None else "idle"
def new_ball(ball, ball_t, used_t):
    """the ball only when a new detection arrived since the last step -> (ball or None, the new used_t)"""
    return (ball, ball_t) if ball is not None and ball_t > used_t else (None, used_t)
def place_tick(pl, t, us, age, ball, poll, busy, margin=PLACE_MARGIN):
    """one 50 Hz step of an active placement -> the body-frame command to publish as it is (zeros when it must not
    drive). Ends it on another commanding program, the page no longer polling, our robot unseen for 2 s, or the robot
    outside the target box."""
    if busy: return pl.stop_with(t, "failed: " + ", ".join(busy) + " is running")[:3]
    if poll > 1.5: return pl.stop_with(t, "failed: page stopped asking")[:3]
    if us is None or age > 2.0: return pl.stop_with(t, "failed: our robot not seen")[:3]
    if us is not None and not in_place_box(us[0], us[1], margin):
        return pl.stop_with(t, f"failed: robot left the area at ({us[0]:+.2f}, {us[1]:+.2f})")[:3]
    fresh = us if us is not None and age <= 0.2 else None          # never drive on a pose older than 0.2 s
    vx, vy, w, _ = pl.step(t, fresh, ball)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (vx, vy, w)):
        return pl.stop_with(t, "failed: the placer gave a bad command")[:3]
    if fresh is None or pl.finished: return (0.0, 0.0, 0.0)
    return (float(vx), float(vy), float(w))
def phase_line(said, status):
    """the status to log when the placer enters a new phase (the waiting states flicker and are only shown), or None"""
    return status if status and status != said and not status.startswith("waiting") else None
def clean(o):
    """JSON-safe: tuples as lists, non-finite numbers as None"""
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    return o if o is None or isinstance(o, (bool, int, str)) else str(o)
def place_view(P):
    """the place object of /state: target, tol, status, the points of debug() (x, y pairs), the other debug values"""
    if P is None: return None
    pl = P["pl"]
    try: d = clean(pl.debug())
    except Exception: d = {}
    pts = {k: v for k, v in d.items() if isinstance(v, list) and len(v) == 2 and all(isinstance(q, (int, float)) and not isinstance(q, bool) for q in v)}
    info = {k: v for k, v in d.items() if k not in pts and not isinstance(v, (list, dict))}
    return dict(active=not pl.finished, target=list(P["target"]), tol=P["tol"], speed=P["speed"], status=str(pl.status),
                points=pts, info=info, result=clean(P["result"]), text=result_text(P["result"]) if P["result"] else "")
def result_text(r):
    """one line for the note and the event log from BallPlacer.result()"""
    s = "ball placed" if r.get("ok") else "placement ended: " + str(r.get("status"))
    b, err = r.get("ball"), r.get("error")
    if b: s += f"; ball at ({b[0]:+.2f}, {b[1]:+.2f})"
    if err is not None: s += f", {err * 100:.0f} cm from the spot"
    return s + f"; {r.get('time') or 0.0:.1f} s, {r.get('pushes') or 0} push" + ("" if r.get("pushes") == 1 else "es")
def place_close(t, why=None):
    """end the active placement (stop_with why), or take note of how it ended: once, a "ball" line with the result
    and the result as the note (call with LOCK held)"""
    P = S["place"]
    if P is None or P["result"] is not None: return
    if why and not P["pl"].finished: P["pl"].stop_with(t, why)
    if not P["pl"].finished: return
    try: P["result"] = clean(P["pl"].result())
    except Exception as e: P["result"] = dict(ok=False, status=f"{P['pl'].status} (no result: {e})")
    P["result"].setdefault("ok", False); txt = result_text(P["result"])
    event("ball", txt); S["note"] = txt

COMMANDERS = procs.COMMANDERS
def others_running():
    """our programs that command the robot (procs.COMMANDERS, from the process table), other than this process and
    its ancestors"""
    return procs.running(COMMANDERS)

RUN = os.environ.get("SOLO_RUN_DIR", "/tmp/solo_run"); CMDF = os.path.join(RUN, "solo.cmd")
NODES = procs.STRATEGY_NODES                             # solo_node (one robot) or duo_node (two robots)
def strategy_procs():
    """pids of the strategy node (and of the launcher around it) on this computer"""
    return [pid for pid, a in procs.process_table() if any(procs.is_instance(a, n) for n in NODES) or procs.is_launcher(a, NODES)]
def strategy_off():
    """stop the node (SIGINT, then SIGTERM); its command line is saved first. A failure to save does not keep it
    from stopping the node -> (stopped, note)"""
    why = ""
    for pid, argv in procs.process_table():               # remember how the node itself was started
        if len(argv) > 1 and os.path.basename(argv[1]) in NODES and procs.is_instance(argv, os.path.basename(argv[1])):
            try:
                os.makedirs(RUN, exist_ok=True)
                with open(CMDF, "w") as f: json.dump(argv, f)
            except OSError as e: why = f"command line not saved ({e})"
    for sig, wait in ((signal.SIGINT, 2.0), (signal.SIGTERM, 1.0)):
        for pid in strategy_procs():
            try: os.kill(pid, sig)
            except OSError: pass
        t0 = now()
        while strategy_procs() and now() - t0 < wait: time.sleep(0.1)
    return not strategy_procs(), why
def strategy_on():
    """start the node again from the saved command line, only when procs.check_node_cmd accepts it -> (ok, note)"""
    if strategy_procs(): return True, ""
    try:
        with open(CMDF) as f: argv = json.load(f)
    except (OSError, ValueError) as e: return False, f"no saved command line ({type(e).__name__})"
    why = procs.check_node_cmd(argv, NODES)
    if why: return False, "saved command line refused: " + why
    try:
        log = open(os.path.join(RUN, "solo.log"), "a")
        p = subprocess.Popen(argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as e: return False, f"cannot start: {e}"
    try:
        with open(os.path.join(RUN, "solo.pid"), "w") as f: f.write(str(p.pid))
    except OSError: pass
    return True, ""
STRAT = ["on" if strategy_procs() else ("off" if os.path.exists(CMDF) else "not started")]

rclpy.init(); node = Node("solo_hq")
pub = node.create_publisher(Twist, os.environ.get("SOLO_CMD_TOPIC", "/cmd_vel"), qos_profile_sensor_data)
def on_us(m):
    with LOCK: S["us"], S["us_t"] = (m.x, m.y, m.theta), now()
def on_ball(m):
    with LOCK: S["ball"], S["ball_t"] = (m.x, m.y), now()
def on_other(key):
    def f(m):
        with LOCK: S["others"][key] = (m.x, m.y, m.theta, now())
    return f
node.create_subscription(Pose2D, f"/vision/poses/{TEAM}/robot{IDX}", on_us, 20)
node.create_subscription(Pose2D, "/vision/poses/ball", on_ball, 20)
SN = {0: "UNKNOWN", 1: "HALT", 2: "STOP", 3: "RUNNING", 4: "FREE_KICK", 5: "KICKOFF", 6: "PENALTY", 7: "TIMEOUT", 8: "BALL_PLACEMENT", 9: "BREAK", 10: "POST_GAME"}
TN = {0: "", 1: "yellow", 2: "blue"}
def on_ref(m):
    spot = [m.placement_x, m.placement_y] if m.has_placement_position else ([0.0, 0.0] if 5 in (m.state, m.next_state if m.has_next_state else -1) else None)
    with LOCK:
        S["ref"] = dict(state=SN.get(m.state, str(m.state)), team=TN.get(m.for_team, ""), next=SN.get(m.next_state, "") if m.has_next_state else "",
                        next_team=TN.get(m.next_for_team, "") if m.has_next_state else "", spot=spot, blue_positive=bool(m.blue_team_on_positive_half))
        S["ref_t"] = now()
MODE = [None]
def on_debug(m):
    try: mode = json.loads(m.data).get("mode")
    except ValueError: return
    if mode != MODE[0]: MODE[0] = mode; event("strategy", str(mode))
node.create_subscription(String, "/solo/debug", on_debug, 10)
if GameState is not None:
    node.create_subscription(GameState, "/referee/game_state", on_ref, 10)
for tm in ("blue", "yellow"):
    for i in range(16):                                  # every topic number the connector can have (with a count of 16 it is the vision id)
        if (tm, i) != (TEAM, IDX):
            node.create_subscription(Pose2D, f"/vision/poses/{tm}/robot{i}", on_other(f"{tm}{i}"), 20)

def gc_listen():
    """the game controller's own packets: stage, command, score, fouls, cards (receive only)"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from referee_connector.proto import ssl_gc_referee_message_pb2 as R
    except Exception:
        return
    g, p = REFEREE.rsplit(":", 1); iface = os.environ.get("SOLO_IFACE", "0.0.0.0")
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP); sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sk.bind((g, int(p))); sk.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(g) + socket.inet_aton(iface)); sk.settimeout(1.0)
    def team(t):
        return dict(name=t.name, score=t.score, fouls=t.foul_counter, yellow=t.yellow_cards, yellow_left=[round(u / 1e6) for u in t.yellow_card_times],
                    red=t.red_cards, timeouts=t.timeouts, keeper=t.goalkeeper, max_bots=t.max_allowed_bots)
    prev = None; seen_ev = set()
    while True:
        try: d, src = sk.recvfrom(65535)
        except socket.timeout: continue
        except OSError: return
        try:
            r = R.Referee(); r.ParseFromString(d)
            ev = [e.WhichOneof("event") or "" for e in r.game_events][-4:]
            info = dict(src=src[0], stage=R.Referee.Stage.Name(r.stage), stage_left=round(r.stage_time_left / 1e6) if r.HasField("stage_time_left") else None,
                        command=R.Referee.Command.Name(r.command), next=R.Referee.Command.Name(r.next_command) if r.HasField("next_command") else "",
                        action_left=round(r.current_action_time_remaining / 1e6, 1) if r.HasField("current_action_time_remaining") else None,
                        blue=team(r.blue), yellow=team(r.yellow), blue_positive=bool(r.blue_team_on_positive_half), events=ev)
        except Exception:
            continue
        try:
            if prev is None: event("referee", f"game controller at {src[0]}: {info['stage']}, {info['command']}, {info['blue']['name']} (blue) {info['blue']['score']} : {info['yellow']['score']} {info['yellow']['name']} (yellow)")
            else:
                if info["stage"] != prev["stage"]: event("referee", "stage " + info["stage"])
                if info["command"] != prev["command"]: event("referee", info["command"] + (f"  (next {info['next']})" if info["next"] else ""))
                if info["blue_positive"] != prev["blue_positive"]: event("referee", "sides: blue defends " + ("+x" if info["blue_positive"] else "-x"))
                for tm in ("blue", "yellow"):
                    a, b0 = info[tm], prev[tm]
                    if a["score"] != b0["score"]: event("goal", f"GOAL for {tm} ({a['name']}): {info['blue']['score']} : {info['yellow']['score']}")
                    for k, lab in (("fouls", "fouls"), ("yellow", "yellow cards"), ("red", "red cards"), ("timeouts", "timeouts left"), ("keeper", "goalkeeper"), ("max_bots", "robots allowed")):
                        if a[k] != b0[k]: event("card" if k in ("yellow", "red") else "referee", f"{tm}: {lab} {b0[k]} -> {a[k]}")
            for e in r.game_events:
                key = e.id if e.HasField("id") else e.SerializeToString()
                if key in seen_ev: continue
                seen_ev.add(key); kind = e.WhichOneof("event") or "event"
                det = " ".join(str(getattr(e, kind)).split()) if kind != "event" else ""
                event("game event", kind + (": " + det[:160] if det else ""))
            prev = info
        except Exception:
            pass
        with LOCK: S["gc"], S["gc_t"] = info, now()
threading.Thread(target=gc_listen, daemon=True).start()

def send(vx, vy, w):
    m = Twist(); m.linear.x, m.linear.y, m.angular.z = float(vx), float(vy), float(w); pub.publish(m)
    S["cmd"] = (round(vx, 3), round(vy, 3), round(w, 3))

def control():
    """50 Hz: field-frame velocity toward the target, ramped, turned into the body frame"""
    vf = [0.0, 0.0]; quiet = 0; dt = 0.02; t_ps = 0.0; lost = None; b_used = 0.0
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=dt)
        with LOCK: u0, a0 = S["us"], now() - S["us_t"]
        if u0 is not None and a0 > 1.0 and lost is None:
            lost = now() - a0; event("vision", f"our robot lost from vision at ({u0[0]:+.2f}, {u0[1]:+.2f})")
        elif lost is not None and u0 is not None and a0 < 0.2:
            event("vision", f"our robot seen again at ({u0[0]:+.2f}, {u0[1]:+.2f}) after {now() - lost:.1f} s"); lost = None
        if now() - t_ps > 1.0:                                        # who else commands the robot, once a second
            t_ps = now(); BUSY[0] = others_running()
            STRAT[0] = "on" if set(NODES) & set(BUSY[0]) else ("off" if os.path.exists(CMDF) else "not started")
            if BUSY[0]:
                with LOCK:
                    if S["target"] is not None: S["target"] = None; note("target dropped: " + ", ".join(BUSY[0]) + " is running")
        with LOCK:
            tgt, us, age, sp, poll = S["target"], S["us"], now() - S["us_t"], S["speed"], now() - S["poll"]
        if tgt is not None and poll > 1.5:
            with LOCK: S["target"] = tgt = None; note("page stopped asking: target dropped")
        if tgt is not None and (us is None or age > 2.0):
            with LOCK: S["target"] = tgt = None; note("target dropped: our robot not seen")
        with LOCK:                                                    # a placement: the placer's body-frame command as it is
            if loop_mode(None, S["place"]) == "place":
                P = S["place"]; t = now()
                ball, b_used = new_ball(S["ball"], S["ball_t"], b_used)
                try: cmd = place_tick(P["pl"], t, S["us"], t - S["us_t"], ball, t - S["poll"], BUSY[0])
                except Exception as e:                                # a placer error ends the placement, not the page
                    cmd = (0.0, 0.0, 0.0); why = f"failed: placer error ({type(e).__name__}: {e})"
                    try: P["pl"].stop_with(t, why)
                    except Exception: P["pl"].finished = True; P["pl"].status = why
                send(*cmd)
                if P["pl"].finished: place_close(t)
                else:
                    ln = phase_line(P["said"], P["pl"].status)
                    if ln: P["said"] = ln; event("ball", ln)
                vf = [0.0, 0.0]; quiet = 0
                continue
            b_used = S["ball_t"]
            if S["place"] is not None: place_close(now())             # ended by /stop, /lock, ... : the result line
        if tgt is None or us is None or age > 0.2:
            vf = [0.0, 0.0]
            if tgt is not None or quiet < 25: send(0, 0, 0)          # zeros while blind, and for 0.5 s after a stop
            quiet = 0 if tgt is not None else quiet + 1
            continue
        quiet = 0
        x, y, th = us; tx, ty, tth = tgt
        d = math.hypot(tx - x, ty - y); eth = wrap(tth - th)
        if d < 0.03 and abs(eth) < 0.05:
            with LOCK: S["target"] = None; note("arrived")
            continue
        v = min(sp, 1.8 * d + 0.03) * (0.3 if abs(eth) > 0.8 else 1.0)   # turn first when far off the heading
        want = ((tx - x) / d * v, (ty - y) / d * v) if d > 1e-3 else (0.0, 0.0)
        for k in (0, 1):                                              # ramp up, never delay slowing down
            if abs(want[k]) > abs(vf[k]) and want[k] * vf[k] >= 0: vf[k] += max(-ACC * dt, min(ACC * dt, want[k] - vf[k]))
            else: vf[k] = want[k]
        w = max(-1.5, min(1.5, 2.5 * eth))
        thp = th + w * 0.1                                            # the heading when the command acts
        c, s = math.cos(thp), math.sin(thp)
        send(c * vf[0] + s * vf[1], -s * vf[0] + c * vf[1], w)

PAGE = r"""<!doctype html><meta charset="utf-8"><title>field</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif}#bar{padding:6px 10px;display:flex;gap:14px;align-items:center}
#stop{background:#c22;color:#fff;border:0;font-size:20px;font-weight:bold;padding:8px 26px;border-radius:6px;cursor:pointer}
canvas{display:block;background:#0b3d1c;cursor:crosshair}#note{color:#fc6}
#main{margin-right:0}#main.wide{margin-right:480px}
#evp{position:fixed;right:0;top:0;bottom:0;width:480px;box-sizing:border-box;background:#161616;border-left:1px solid #3a3a3a;overflow-y:auto;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace}
#evh{position:sticky;top:0;background:#222;border-bottom:1px solid #3a3a3a;padding:9px 12px;font:bold 14px sans-serif;display:flex;justify-content:space-between;cursor:pointer}
.ev{display:grid;grid-template-columns:64px 92px 1fr;gap:8px;padding:3px 12px;border-bottom:1px solid #222}.ev .t{color:#888}.ev .x{color:#ddd;overflow-wrap:anywhere}
.ev.goal{background:#14301a}.ev.card{background:#33240f}.ev.ball{background:#2a1c0c}#pl{display:none;padding:0 10px 4px;font-size:15px;color:#fb6}</style>
<div id="evp"><div id="evh"><span id="evt">Events</span><span>hide</span></div><div id="evl"></div></div><div id="main" class="wide"><div id="bar"><button id="stop">STOP</button><button id="cov" style="font-size:16px;padding:8px 14px;border-radius:6px;border:0;cursor:pointer;background:#345;color:#fff">camera map: off</button><button id="evb" style="font-size:16px;padding:8px 14px;border-radius:6px;border:0;cursor:pointer;background:#345;color:#fff">events</button><button id="lock" style="font-size:16px;padding:8px 14px;border-radius:6px;border:0;cursor:pointer"></button><button id="strat" style="font-size:16px;padding:8px 14px;border-radius:6px;border:0;cursor:pointer"></button><button id="pmode" style="font-size:16px;padding:8px 14px;border-radius:6px;border:0;cursor:pointer;color:#fff"></button><select id="ptol" style="font-size:15px"><option value="rough">rough 0.15 m</option><option value="precise">precise 0.05 m</option></select><label>speed <input id="sp" type="range" min="0.1" max="VMAXV" step="0.05"> <span id="spv"></span> m/s</label>
<span id="st"></span><span id="note"></span></div><div id="pl"></div><div id="gc" style="padding:0 10px 4px;font-size:17px;font-weight:bold"></div><div id="gc2" style="padding:0 10px 4px;font-size:14px;color:#bbb"></div><div id="ref" style="padding:0 10px 6px;font-size:15px"></div><canvas id="c"></canvas></div>
<script>
let evopen=true,bmode=false,cov=null,covOn=false;
async function loadCov(){try{cov=await(await fetch('/map')).json()}catch(e){cov=null}}
function drawCov(){if(!covOn)return;g.fillStyle='#2a2a2a';g.fillRect(0,0,c.width,c.height);if(!cov||!cov.ok)return;const w=cov.cell*k+0.6;
 for(let i=0;i<cov.nx;i++)for(let j=0;j<cov.ny;j++){const v=cov.rate[i][j];if(v==null)continue;const h=120*Math.max(0,Math.min(1,(v-0.3)/0.6));
  g.fillStyle='hsl('+h+',80%,'+(28+12*Math.min(1,v))+'%)';g.fillRect(X(cov.x0+i*cov.cell),Y(cov.y0+(j+1)*cov.cell),w,w)}}
setInterval(()=>{if(covOn)loadCov()},5000);
const plEl=document.getElementById('pl'),pm=document.getElementById('pmode'),ptol=document.getElementById('ptol');const LX=2.25,LY=1.5,M=0.6,c=document.getElementById('c'),g=c.getContext('2d');let st=null,drag=null,k=1;
function fit(){c.width=innerWidth-(evopen?480:0);c.height=innerHeight-140-(plEl.offsetHeight||0);k=Math.min(c.width/(2*(LX+M)),c.height/(2*(LY+M)))}
const X=x=>c.width/2+x*k,Y=y=>c.height/2-y*k,FX=p=>(p-c.width/2)/k,FY=p=>(c.height/2-p)/k;
function post(u,b){fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})})}
function robot(x,y,th,col,lab){g.fillStyle=col;g.beginPath();g.arc(X(x),Y(y),0.09*k,0,7);g.fill();g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();
 g.moveTo(X(x),Y(y));g.lineTo(X(x+0.16*Math.cos(th)),Y(y+0.16*Math.sin(th)));g.stroke();g.fillStyle='#fff';g.fillText(lab,X(x)+8,Y(y)-10)}
function draw(){fit();g.clearRect(0,0,c.width,c.height);drawCov();g.strokeStyle='#fff';g.lineWidth=2;g.strokeRect(X(-LX),Y(LY),2*LX*k,2*LY*k);
 g.beginPath();g.moveTo(X(0),Y(LY));g.lineTo(X(0),Y(-LY));g.stroke();g.beginPath();g.arc(X(0),Y(0),0.5*k,0,7);g.stroke();
 for(const s of[-1,1]){g.strokeRect(X(s>0?LX-0.5:-LX),Y(0.675),0.5*k,1.35*k);g.strokeStyle='#c6f';g.strokeRect(X(s>0?LX:-LX-0.18),Y(0.4),0.18*k,0.8*k);g.strokeStyle='#fff'}
 g.fillStyle='#9c9';g.fillText('-x',X(-LX)+4,Y(-LY)-6);g.fillText('+x',X(LX)-20,Y(-LY)-6);g.fillText('+y',X(-LX)+4,Y(LY)+14);
 if(!st)return;for(const [n,o] of Object.entries(st.others))robot(o[0],o[1],o[2],n.startsWith('blue')?'#36c':'#cc3',n);
 let rt='no referee';if(st.ref){const r=st.ref;rt='referee: '+r.state+(r.team?' for '+r.team:'')+(r.next?'   next: '+r.next+(r.next_team?' for '+r.next_team:''):'')+'   blue defends '+(r.blue_positive?'+x':'-x');
  if(r.spot){const q=r.spot;g.strokeStyle='#fff';g.setLineDash([6,4]);g.beginPath();g.arc(X(q[0]),Y(q[1]),0.1*k,0,7);g.stroke();g.setLineDash([]);g.beginPath();g.moveTo(X(q[0])-12,Y(q[1]));g.lineTo(X(q[0])+12,Y(q[1]));g.moveTo(X(q[0]),Y(q[1])-12);g.lineTo(X(q[0]),Y(q[1])+12);g.stroke();
   rt+='   BALL SPOT ('+q[0].toFixed(2)+', '+q[1].toFixed(2)+')';if(st.ball){const d=Math.hypot(st.ball[0]-q[0],st.ball[1]-q[1]);rt+=d<0.05?'  ball is on it':'  ball is '+(d*100).toFixed(0)+' cm off: move it '+(Math.abs(q[0]-st.ball[0])*100).toFixed(0)+' cm toward '+(q[0]>st.ball[0]?'+x':'-x')+', '+(Math.abs(q[1]-st.ball[1])*100).toFixed(0)+' cm toward '+(q[1]>st.ball[1]?'+y':'-y')}}}
 rt+=st.ball?'   |  ball at ('+st.ball[0].toFixed(2)+', '+st.ball[1].toFixed(2)+')':'   |  ball not seen';document.getElementById('ref').textContent=rt;
 if(st.ball){g.fillStyle='#f80';g.beginPath();g.arc(X(st.ball[0]),Y(st.ball[1]),0.035*k,0,7);g.fill()}
 if(st.place){const P=st.place,q=P.target,col=P.active?'#f80':'#a86';g.strokeStyle=col;g.lineWidth=2;g.setLineDash(P.active?[]:[5,4]);g.beginPath();g.arc(X(q[0]),Y(q[1]),Math.max(P.tol*k,3),0,7);g.stroke();g.setLineDash([]);
  g.beginPath();g.moveTo(X(q[0])-8,Y(q[1]));g.lineTo(X(q[0])+8,Y(q[1]));g.moveTo(X(q[0]),Y(q[1])-8);g.lineTo(X(q[0]),Y(q[1])+8);g.stroke();
  if(P.active){const pp=P.points;if(pp.stand&&pp.aim){g.strokeStyle='#0cf';g.setLineDash([3,4]);g.beginPath();g.moveTo(X(pp.stand[0]),Y(pp.stand[1]));g.lineTo(X(pp.aim[0]),Y(pp.aim[1]));g.stroke();g.setLineDash([])}
   for(const [n,p] of Object.entries(pp)){if(n=='target')continue;g.fillStyle=n=='stand'?'#0cf':'#fff';g.beginPath();g.arc(X(p[0]),Y(p[1]),4,0,7);g.fill();g.fillText(n=='stand'?'line-up':n,X(p[0])+6,Y(p[1])+12)}}
  const R=P.result;if(R&&R.ball){g.strokeStyle=R.ok?'#6f6':'#f66';g.beginPath();g.arc(X(R.ball[0]),Y(R.ball[1]),0.05*k,0,7);g.stroke();g.fillStyle=g.strokeStyle;if(R.error!=null)g.fillText((R.error*100).toFixed(0)+' cm',X(R.ball[0])+8,Y(R.ball[1])-8)}}
 if(st.target){const t=st.target;g.strokeStyle='#f44';g.lineWidth=2;g.beginPath();g.arc(X(t[0]),Y(t[1]),0.1*k,0,7);g.moveTo(X(t[0]),Y(t[1]));
  g.lineTo(X(t[0]+0.2*Math.cos(t[2])),Y(t[1]+0.2*Math.sin(t[2])));g.stroke()}
 if(st.us)robot(st.us[0],st.us[1],st.us[2],st.us_age<0.3?(st.team=='blue'?'#4af':'#ff4'):'#888','us');
 if(drag&&!bmode){g.strokeStyle='#f44';g.beginPath();g.moveTo(drag.x0,drag.y0);g.lineTo(drag.x1,drag.y1);g.stroke()}
 document.getElementById('st').textContent=(st.us?`us (${st.us[0].toFixed(2)}, ${st.us[1].toFixed(2)}) ${(st.us[2]*57.3).toFixed(0)} deg, seen ${st.us_age.toFixed(1)} s ago`:'our robot not seen')+
  (st.target?`  ->  (${st.target[0].toFixed(2)}, ${st.target[1].toFixed(2)})`:'')+`  cmd ${st.cmd}`+(st.busy.length?'  BLOCKED by '+st.busy:'');
 document.getElementById('note').textContent=st.note}
c.onmousedown=e=>{drag={x0:e.offsetX,y0:e.offsetY,x1:e.offsetX,y1:e.offsetY}};c.onmousemove=e=>{if(drag){drag.x1=e.offsetX;drag.y1=e.offsetY}};
c.onmouseup=e=>{if(!drag)return;if(st&&st.lock){drag=null;return}const dx=drag.x1-drag.x0,dy=drag.y1-drag.y0,b={x:FX(drag.x0),y:FY(drag.y0)};if(bmode){post('/place',{x:b.x,y:b.y,precise:ptol.value=='precise'});drag=null;return}if(Math.hypot(dx,dy)>15)b.th=Math.atan2(-dy,dx);post('/goto',b);drag=null};
document.getElementById('stop').onclick=()=>post('/stop');
const cb=document.getElementById('cov');cb.onclick=()=>{covOn=!covOn;cb.textContent='camera map: '+(covOn?'on':'off');cb.style.background=covOn?'#274':'#345';if(covOn)loadCov().then(()=>{if(cov&&!cov.ok)cb.textContent='camera map: none ('+cov.why+')'})};
const evp=document.getElementById('evp'),evb=document.getElementById('evb'),evl=document.getElementById('evl'),mainEl=document.getElementById('main');let evn=0;
const COL={goal:'#6f6',card:'#fa4','game event':'#fc6',referee:'#9cf',strategy:'#ccc',vision:'#f88',page:'#aaa',ball:'#f80'};
function evset(o){evopen=o;evp.style.display=o?'block':'none';mainEl.className=o?'wide':'';evb.style.display=o?'none':''}
evb.onclick=()=>evset(true);document.getElementById('evh').onclick=()=>evset(false);evset(true);
async function pollev(){try{const ev=await(await fetch('/events?since='+evn)).json();for(const e of ev){const d=document.createElement('div');d.className='ev '+e.kind.replace(' ','');for(const [cl,tx,co] of [['t',e.clock],['k',e.kind,COL[e.kind]||'#ddd'],['x',e.text]]){const q=document.createElement('span');q.className=cl;q.textContent=tx;if(co)q.style.color=co;d.appendChild(q)}evl.prepend(d);evn=e.n+1}
 evb.textContent='events ('+evn+')';document.getElementById('evt').textContent='Events ('+evn+')'}catch(e){}}
setInterval(pollev,1000);
const lb=document.getElementById('lock');let arm=0;lb.onclick=()=>{if(!st)return;if(st.lock){if(Date.now()-arm<3000){post('/lock',{on:false});arm=0}else{arm=Date.now();lb.textContent='click again to unlock'}}else post('/lock',{on:true})};
const sb=document.getElementById('strat');sb.onclick=()=>{if(st){sb.textContent='strategy: switching...';post('/strategy',{on:st.strategy!='on'})}};addEventListener('keydown',e=>{if(e.key==' '||e.key=='Escape'){e.preventDefault();post('/stop')}});
function pmset(o){bmode=o;pm.textContent=o?'click = place ball':'click = drive robot';pm.style.background=o?'#c60':'#345';ptol.style.display=o?'':'none'}
pm.onclick=()=>{if(st&&!st.place_ok)return;pmset(!bmode)};pmset(false);
function plshow(){let t='';const P=st.place;
 if(!st.place_ok){if(bmode)pmset(false);pm.textContent='ball mode off';pm.style.background='#555';pm.style.opacity=0.5;pm.title=st.place_err;t='ball mode off: '+st.place_err}
 else if(P&&P.active)t='placing the ball at ('+P.target[0].toFixed(2)+', '+P.target[1].toFixed(2)+'), tol '+(P.tol*100).toFixed(0)+' cm, '+P.speed.toFixed(2)+' m/s:  '+P.status+(P.info.kind?'   ['+P.info.kind+' push]':'')+'   pushes '+(P.info.pushes==null?0:P.info.pushes);
 else if(P&&P.text)t='last placement: '+P.text+(bmode?'   |   click where the ball should go':'');
 else if(bmode)t='ball mode: click where the ball should go';
 plEl.textContent=t;plEl.style.display=t?'block':'none'}
const sp=document.getElementById('sp');sp.oninput=()=>{post('/speed',{v:+sp.value});document.getElementById('spv').textContent=sp.value};
async function poll(){try{st=await(await fetch('/state?page=1')).json();if(document.activeElement!=sp){sp.value=st.speed;document.getElementById('spv').textContent=st.speed}}catch(e){st=null}
 if(st&&!sb.textContent.includes('switching')||st&&st.strategy!=sb.dataset.s){sb.dataset.s=st.strategy;sb.textContent='strategy: '+st.strategy.toUpperCase()+(st.strategy=='on'?' (click: off, place by hand)':st.strategy=='off'?' (click: on)':'');sb.style.background=st.strategy=='on'?'#2a7':'#888';sb.style.color='#fff'}
 if(st){if(!(st.lock&&Date.now()-arm<3000)){lb.textContent=st.lock?'LOCKED (view only)':'unlocked: click to lock';lb.style.background=st.lock?'#b70':'#555';lb.style.color='#fff'}
  c.style.cursor=st.lock?'not-allowed':'crosshair';sb.style.opacity=st.lock?0.4:1;sp.disabled=st.lock;
  const G=st.gc,g1=document.getElementById('gc'),g2=document.getElementById('gc2');
  if(G){const tm=x=>x==null?'':Math.floor(x/60)+':'+String(Math.max(0,x)%60).padStart(2,'0'),T=(n,t)=>n+' '+(t.name||'?')+': fouls '+t.fouls+', yellow '+t.yellow+(t.yellow_left.length?' (active '+t.yellow_left.map(tm).join(' ')+')':'')+', red '+t.red+', timeouts '+t.timeouts+', keeper '+t.keeper+', robots allowed '+t.max_bots;
   g1.textContent=(G.blue.name||'blue')+' '+G.blue.score+' : '+G.yellow.score+' '+(G.yellow.name||'yellow')+'    '+G.stage+(G.stage_left!=null?' '+tm(G.stage_left):'')+'    '+G.command+(G.next?'  (next '+G.next+')':'')+(G.action_left!=null&&G.action_left>0?'  '+G.action_left+' s':'');
   g2.textContent=T('BLUE',G.blue)+'   |   '+T('YELLOW',G.yellow)+'   |   blue defends '+(G.blue_positive?'+x':'-x')+(G.events.length?'   |   events: '+G.events.join(', '):'')+'   |   from '+G.src}
  else{g1.textContent='no game controller packets';g2.textContent=''}
  plshow()}
 draw()}
setInterval(poll,100);</script>""".replace("VMAXV", str(VMAX))

MAPV = [None, -1e9, None]                             # the coverage map object, when it was last read, the payload
def coverage_rows(r):
    """the rate grid as nx lists of ny values: two decimals, None where there is no evidence"""
    return [[None if v != v else round(float(v), 2) for v in col] for col in r]
def coverage():
    """the camera coverage map for the page (re-read from its file at most every 5 s); ok False with the reason without one"""
    t = now()
    if MAPV[2] is not None and t - MAPV[1] < 5.0: return MAPV[2]
    MAPV[1] = t
    try:
        if MAPV[0] is None:
            import vision_map
            MAPV[0] = vision_map.load()
        else:
            MAPV[0].reload()
        m = MAPV[0]
        if not m.ok: raise OSError("no coverage map file yet")
        MAPV[2] = dict(ok=True, x0=m.x0, y0=m.y0, cell=m.cell, nx=m.nx, ny=m.ny, rate=coverage_rows(m.rate_grid(min_evidence=0.5)))
    except Exception as e:
        MAPV[2] = dict(ok=False, why=str(e)[:120])
    return MAPV[2]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def out(self, body, ctype="application/json", code=200):
        b = body.encode(); self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def refuse(self, code, why):
        self.out(json.dumps(dict(error=why)), code=code)
    def do_GET(self):
        if not host_ok(self.headers.get("Host"), PORT, BIND):
            return self.refuse(403, "bad Host header")
        if self.path.startswith("/events"):
            try: since = max(0, int(self.path.split("since=")[1]))
            except (IndexError, ValueError): since = 0
            with ELOCK: ev = EVENTS[since:][-500:]
            return self.out(json.dumps(ev))
        path, _, query = self.path.partition("?")
        if path == "/map":
            return self.out(json.dumps(coverage()))
        if path == "/state":
            with LOCK:
                if "page=1" in query.split("&"): S["poll"] = now()      # the dead man's switch: the page's own polls only
                t = now()
                d = dict(team=TEAM, us=S["us"], us_age=round(t - S["us_t"], 2) if S["us"] else None,
                         ball=S["ball"] if S["ball"] and t - S["ball_t"] < 1.0 else None,
                         others={k: v[:3] for k, v in S["others"].items() if t - v[3] < 1.0},
                         target=S["target"], speed=S["speed"], note=S["note"], cmd=S["cmd"], busy=BUSY[0], strategy=STRAT[0],
                         ref=S["ref"] if S["ref"] and t - S["ref_t"] < 1.0 else None,
                         gc=S["gc"] if S["gc"] and t - S["gc_t"] < 2.0 else None, lock=S["lock"],
                         place=place_view(S["place"]), place_ok=BallPlacer is not None, place_err=PLACE_ERR)
            self.out(json.dumps(d))
        else:
            self.out(PAGE, "text/html; charset=utf-8")
    def do_POST(self):
        if not host_ok(self.headers.get("Host"), PORT, BIND):
            return self.refuse(403, "bad Host header")
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            return self.refuse(415, "Content-Type must be application/json")
        if self.path not in ("/stop", "/speed", "/goto", "/place", "/lock", "/strategy"):
            return self.refuse(404, "no such request")
        try:
            n = int(self.headers.get("Content-Length", 0))
            if not 0 <= n <= 4096: raise ValueError("body size")
            b = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, UnicodeDecodeError):
            return self.refuse(400, "malformed JSON")
        if self.path == "/lock":
            ok, on = check_flag(b, "on", True)
            if not ok: return self.refuse(400, "on must be true or false")
            with LOCK:
                S["lock"] = on
                if on: S["target"] = None
                note("locked" if on else "unlocked")
                if on: place_close(now(), "stopped: locked")
            return self.out("{}")
        if S["lock"] and self.path != "/stop":
            with LOCK: note("locked: unlock first")
            return self.refuse(423, "locked: unlock first")
        if self.path == "/strategy":
            ok, on = check_flag(b, "on")
            if not ok: return self.refuse(400, "on must be true or false")
            busy = [n for n in others_running() if n not in NODES]
            if on and busy:                                           # the node would command the robot with them
                why = "strategy not started: " + ", ".join(busy) + " is running"
                with LOCK: event("page", why); S["note"] = why
                return self.refuse(409, why)
            with LOCK: S["target"] = None; place_close(now(), "stopped: strategy switched")
            try: ok, why = strategy_on() if on else strategy_off()
            except Exception as e: ok, why = False, f"{type(e).__name__}: {e}"
            BUSY[0] = others_running(); STRAT[0] = "on" if set(NODES) & set(BUSY[0]) else ("off" if os.path.exists(CMDF) else "not started")
            text = ("strategy " + STRAT[0]) if ok else "strategy switch failed"
            if why: text += ": " + why
            with LOCK: event("page", text); S["note"] = text
            return self.out("{}")
        if self.path == "/stop":
            with LOCK: S["target"] = None; note("stopped"); place_close(now(), "stopped")
        elif self.path == "/speed":
            ok, v = check_speed(b)
            if not ok: return self.refuse(400, "v must be a finite number")
            with LOCK: S["speed"] = max(0.1, min(VMAX, v))
        elif self.path == "/goto":
            ok, gx, gy, gth = check_goto(b)
            if not ok: return self.refuse(400, "x, y (and th if given) must be finite numbers")
            busy = others_running()                                   # the process list is read outside the lock
            with LOCK:
                BUSY[0] = busy
                if busy: note("not driving: " + ", ".join(busy) + " is running")
                elif S["us"] is None or now() - S["us_t"] > 0.5: note("not driving: our robot is not seen")
                else:
                    x = max(BOX[0], min(BOX[1], gx)); y = max(BOX[2], min(BOX[3], gy))
                    place_close(now(), "stopped: replaced by a drive target")
                    S["target"] = (x, y, gth if gth is not None else S["us"][2]); note(""); event("page", f"target ({x:+.2f}, {y:+.2f})")
        elif self.path == "/place":
            req, why = place_request(b)
            if BallPlacer is None:
                with LOCK: note("ball mode off: " + PLACE_ERR)
                return self.refuse(409, "ball mode off: " + PLACE_ERR)
            if req is None:
                with LOCK: note("not placing: " + why)
                return self.refuse(400, why)
            busy = others_running()                                   # the process list is read outside the lock
            with LOCK:
                BUSY[0] = busy
                if busy: note("not placing: " + ", ".join(busy) + " is running")
                elif S["us"] is None or now() - S["us_t"] > 0.5: note("not placing: our robot is not seen")
                else:
                    x, y, tol = req; sp = place_speed(S["speed"]); t = now()
                    try: pl = BallPlacer((x, y), tol=tol, speed=sp, carry=min(PLACE_CARRY, VMAX), margin=PLACE_MARGIN)
                    except Exception as e: note(f"not placing: {type(e).__name__}: {e}"); pl = None
                    if pl is not None:
                        S["target"] = None; place_close(t, "stopped: replaced by a new placement")
                        S["place"] = dict(pl=pl, target=(x, y), tol=tol, speed=sp, said="", result=None)
                        note(""); event("ball", f"placement to ({x:+.2f}, {y:+.2f}), tol {tol * 100:.0f} cm, speed {sp:.2f} m/s")
        self.out("{}")

BUSY = [others_running()]
def main():
    if BUSY[0]: print("note: running now:", ", ".join(BUSY[0]), "(driving is refused while they run)", flush=True)
    if BallPlacer is None: print("note: ball mode off:", PLACE_ERR, flush=True)
    srv = ThreadingHTTPServer((BIND, PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"field page on http://{BIND}:{PORT}  team {TEAM} robot {IDX}", flush=True)
    try:
        control()
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(10): send(0, 0, 0); time.sleep(0.02)

if __name__ == "__main__":
    main()
