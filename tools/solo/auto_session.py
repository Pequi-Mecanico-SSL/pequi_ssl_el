#!/usr/bin/env python3
"""Run a list of field tests with our robot and the ball: alone on the field, or (with a home area) in one part of a
field shared with other robots, which the supervisor watches in the raw vision.

OPERATOR NOTE
  Before: vision connector, recorders and the field page (hq.py, locked) running; the strategy stack started with
  run_real.sh and switched OFF (so $SOLO_RUN_DIR/solo.cmd holds the node's command line); the ball anywhere inside
  the field; without a home area no other robot on the field (none is watched; see HOME AREA); SOLO_NOTIFY_CMD set to an executable called as `CMD "message" "title"`.
  Start (after `source tools/solo/env.sh`):
      python3 tools/solo/auto_session.py <team> tools/solo/sessions/short_10.jsonl --budget 10
      python3 tools/solo/auto_session.py <team> SESSION --dry-run       # prints every command and the decision table
  Options: --budget MIN (default: the session's budget_min), --var NAME=VALUE (e.g. P, the folder of the series plans;
  OUT, the folder of the logs),
  --hq URL (field page, default http://127.0.0.1:8700), --ball-lost S (5), --robot-lost S (2), --robot-wait S (10),
  --busy-wait S (20), --realert S (180).
      python3 tools/solo/auto_session.py <team> SESSION --sim corner     # offline: fakes for every step, simulated clock
                                    (scenarios: ok, corner, goal, stuck, timeout; log in $SOLO_DATA/auto/*_sim_*.jsonl)
  Stop: Ctrl-C once (or SIGTERM) = finish the current step, strategy off, end marker, exit. Twice = also stop the
  running step now: its process group gets SIGINT, after 1.5 s SIGTERM, after 1.5 s more SIGKILL. place_ball.py
  sends zeros for 0.3 s on SIGINT and SIGTERM; after SIGKILL nothing is sent, and the robot's own command timeout
  (0.3 s) stops it. The budget ends the session the same way as one Ctrl-C.
  What the notification says (title "field session"), and what the person does:
    "ball in the +x goal at (x, y): put it anywhere inside the field"            -> put the ball back inside the lines
    "ball in the (+x, -y) corner at ..." / "ball against the +y wall at ..."      -> the same
    "ball not seen since 12:01:02 (last at (x, y)): put a ball anywhere inside the field"
    "robot stuck against the +y wall at (x, y): free it and put it inside the field" -> lift it 0.5 m into the field
    "robot not seen since 12:01:02, last at (x, y): ..."                          -> check the robot / its tag
    "another program commands the robot (solo_node): the session waits"          -> stop that program
    "step 7 (ram 1.5 >) failed: ABORT: ...; the session goes on"                  -> nothing (steps with on_fail alert)
    "session ended: budget used (12 steps done, 2 skipped)"                       -> collect the robot
  The session goes on by itself as soon as the ball is seen resting inside the field again / the robot is seen
  0.25 m clear of every wall. Reminders every --realert s, at most 3 messages per incident.
  Log: $SOLO_DATA/auto/<YYYYmmdd_HHMMSS>_session.jsonl (every step's command, times, exit status, result line, the
  world after it and every decision); markers go to $SOLO_DATA/events_<YYYYmmdd>.jsonl (one line per marker: its
  text, and "data": "use" when the recordings from there on are test data, "skip" when they are not).

SESSION FILE: JSON lines, one step per line ('#' lines are comments). An optional first line
    {"kind": "session", "name": "...", "vars": {"P": "${SESSION_DIR}/plans"}, "env": {"SOLO_BRAKE": "auto"}}
sets variables (${NAME} in any string; TEAM, T (this folder), SOLO_DATA, SESSION_DIR (the session file's folder) and
the environment are always there; --var overrides) and the environment of every step.
Steps (fields common to all: label, needs_ball, on_fail = skip | retry | stop | alert (default skip), timeout s,
group (a failed step skips the rest of its group), env {}, final (run at the end even when the budget is used;
marker / cmd / wait only), keep_ball_out "x0,x1,y0,y1" (the ball must be 0.15 m clear of this box before the step;
with park [x, y] the ball is placed there first)):
    {"kind": "place", "x": -0.85, "y": 0.65, "opts": ["--carry", "0.35"]}               place_ball.py; needs_ball
    {"kind": "test", "test": "ram", "args": [1.2, 0, 1.0, 0.25], "out": "${OUT}/logs", "tee": "${OUT}/rams.log",
     "env": {"SOLO_RAMP": "1.7", "SOLO_BOX": "-2.05,2.05,0.25,1.05"}}                   characterize.py
    {"kind": "series", "plan": "${P}/steps_ramped.jsonl", "from": 1, "max": 8, "args": ["--out", "${OUT}/logs"]}  series.py
    {"kind": "strategy", "on": true, "seconds": 10}       start the strategy node, wait, stop it ("on": false = stop)
    {"kind": "goto", "x": 1.2, "y": -0.8, "v": 0.4}                                      goto.py
    {"kind": "marker", "text": "short session start", "data": "use"}                    a line in the events file
    {"kind": "wait", "seconds": 5}
    {"kind": "cmd", "argv": ["${T}/robot_stack.sh", "save"], "final": true}              any command, no motion
needs_ball defaults to true for place, false otherwise.

DECISIONS (after every step the world is read from the field page's /state: robot pose, ball, their ages):
  ball inside the field (or in a defense area, or past a line within the placer's margin) -> ball steps run
  ball in a goal / against a goal frame / in a corner / against a wall / not seen > --ball-lost s:
      the placer is asked (offline, BallPlacer) whether it can push the ball toward the nearest point 0.3 m inside the
      lines; if so, one place_ball.py run; otherwise (or if that fails) one alert; the steps with needs_ball false go
      on; ball steps resume when a ball is seen resting inside the field; nothing ball-free left: wait, re-alert
  place_ball "unreachable: ball ..." -> the same as above, the place step stays at the head of the ball steps
  place_ball "failed: robot stuck ..." or a step that aborted / timed out with the robot within 0.10 m of a wall or
      a goal frame -> one slow goto 0.5 m inside the field, straight away from it; moved < 0.1 m: motion stops,
      alert, wait until the robot is seen clear
  place_ball failed otherwise (timeout, left the area) -> once more with SOLO_BOX 0.15 m wider and --carry 0.25;
      a second failure follows on_fail
  a test's ABORT line or a non-zero exit -> on_fail
  robot not seen > --robot-lost s before a motion step -> wait --robot-wait s, then alert and wait
  another commanding program running (procs.COMMANDERS: a python process whose script is solo_node, characterize.py,
      series.py, ram_series.py, place_ball.py, goto.py, nudge.py or frame_check.py; this process, its ancestors and
      the step it runs itself excluded; a command line that only mentions a name does not count), or the field page
      driving (a click target, a placement) -> no motion step starts, and no placement the supervisor adds (parking,
      rescue); after --busy-wait s an alert, and the session waits
  budget used / Ctrl-C -> the current step ends, strategy off, final steps, end marker, exit

HOME AREA (tests confined to one half of a shared field; tools/solo/sessions/left_*.jsonl)
  --home "x0,x1,y0,y1" (or "home" in the session header; default: the whole field, no home checks) is where our robot
  works; its edges inside the field (more than 0.3 m from the lines) are "open": the side other teams use. Options:
  --corridor M (0.7), --fetch-wait S (60), --park X,Y (or "park" in the header), --intrude-dist M (1.0), --clear S (5),
  --hold-limit S (120), --ball-hold S (30), --fetch-limit M (1.0), --info-alert S (10), --alert-dead S (300), --settle S
  (1.5), --max-speed M/S (2.0), --reappear M (0.15), --probes N (6; 0 = no identity probe), --probe-rate RAD/S (1.5),
  --probe-time S (0.7), --clear-own S (1.5); --no-ball (or "ball": "ignore")
  for a session that never touches a ball.
  Required with a home area: --our-id N (or SOLO_OUR_ID=N), our robot's vision id; the field's cameras, --cameras N
  (ids 0..N-1, or SOLO_CAMERAS=N) or --camera-ids 0,1,... (or SOLO_CAMERA_IDS): a listed camera that has not sent a
  detection frame for 0.5 s, or never, is no information (a camera that sends without being listed is listed from
  its first frame on, and logged, its hold alerts saying it is not on the list; frames from one during the start check refuse the start, naming its id);
  SOLO_NOTIFY_CMD (a person must be alerted; a failing command is tried once more, both results in the session log):
  the start check sends "session starting: <name>, the -x half, budget N min" and the session starts only when the
  command exits 0 (a dry run only prints it). Every other alert goes through a queue a background thread sends, so
  no stop ever waits for the alert command (a stop comes first, its alert after); a failed send is retried after 15,
  30, 60 s, then every 60 s, in order; an alert older than 10 min is dropped and counted ("N earlier alerts were not
  delivered since HH:MM: ..." starts the next one delivered); every failure is in the session log. When no alert has
  been delivered for --alert-dead s (counted from the first failed send or from the oldest alert still waiting, an
  alert command that never returns included), no motion step runs (the robot stands, the hold of no information)
  until one is; the first alert delivered then says so. The sending thread is watched: ended, it is started again
  once (logged); ended again, the alert channel is dead for the rest of the session. At the end the session waits up to 20 s for the queue;
  with the sending thread ended, "session ended" is sent once directly (the result logged). The supervisor listens to the
  vision multicast itself (receive only: --vision GROUP:PORT, default 224.5.23.2:10006; --vision-iface ADDR, default
  $SOLO_IFACE or 0.0.0.0; --vision-mirror when the connector runs with mirror:=true) and sees every robot of both
  colours with its vision id; our robot is the detection of our colour and id within 0.25 m of the robot the tools
  drive. A robot of another colour or id counts from its second sighting by one camera within 0.25 s (any two frames:
  a robot detected in every other frame counts), or within 1.0 s when it lies within 0.2 m + 2 m/s x the gap of the
  first (a robot detected only now and then), and then until its sighting expires (0.5 s); a sighting within 0.2 m
  of one of another colour or id of the last 0.25 s counts at once (an id read differently from frame to frame; our
  robot's sightings do not confirm another), and so does one whose camera sends no newer frame within 0.1 s: a single
  isolated false detection stops nothing. Two
  detections of one camera frame are always two robots; two cameras' detections of one colour + id are
  one robot when closer than 0.15 m plus 2 m/s per second of frame age difference (at most 0.35 m), or, for one its
  camera no longer sees, 3 m/s (at most 0.8 m). Every other detection is another robot. Our robot's last unambiguous
  position (the anchor) is kept with the time of its detection. A single robot of our colour and id seen after it is
  ours when ours was unseen for less than 0.5 s and it is within 0.15 m + --max-speed x the time since the anchor's
  detection (0.15 m when no motion child or strategy node ran then or stopped within --settle s before: it stood; after
  two copies were seen, only the copy the anchor followed until 0.15 s before), or when ours was unseen for 0.5 s or
  more, it is within --reappear m of the anchor (0.15 m when it stood) and no second copy was seen since the anchor.
  Anything else is "which robot is ours was lost: ..." (no information, alerted at once, "...: put the robot back at
  (x, y) or restart the session; the session will try to recognise it by a short turn on the spot", the last part
  while probes are left, in the hold that runs the probe, with the candidate (none in view: the anchor) where a probe
  is accepted; a new loss is alerted whatever the backoff); when the lost message went out in any alert, its recovery
  is alerted too: "our robot recognised by its turn at (x, y): the session resumes" or "our robot seen again at its
  place (x, y): the session resumes" ("...; the session still holds: <cause>" while another cause keeps the hold);
  a probe that fails leaves it lost at the same anchor; it clears when one robot of our colour and id has been seen within 0.15 m of that
  position for --clear s, or by the identity probe, never by time alone.
  IDENTITY PROBE: while which robot is ours is lost, the supervisor may prove whether the one robot of our colour and
  id in view (the candidate) is ours: when for --clear s the vision has been fresh, exactly one robot of our colour
  and id has been seen, standing (moved less than 0.03 m, turned less than 5 deg), no other robot has been inside our
  watched area or within --intrude-dist of the candidate, and it stands inside our half or within --fetch-limit of its open
  edges, two turns on the spot in a random order of signs run, each its own child `characterize.py <team> spin
  <+-rate> <time>` (--probe-rate, --probe-time; SOLO_BOX 0.25 m round the candidate), 0.5 s apart, watched (no
  information, another robot inside or near, the candidate moving 0.10 m stop it). The candidate's heading must change
  in the commanded direction by 0.4 to 1.6 x rate x time in each turn (read before the child and 0.3 s after it ended)
  and it must move less than 0.10 m: then identity is anchored on it (the alert above; a quiet notice with the same
  text when no lost alert went out); else the alert "the robot seen as blue 3 at (x, y) did not follow a commanded turn: it is not
  ours, or ours does not respond; the session holds", and no probe of a candidate at that place for 60 s. At most
  --probes per session (the last alert says none is left, how many gave no verdict and the last reason). A probe
  child that did not complete its turn (a non-zero exit, a result line marked interrupted or aborted) gives no
  verdict: no alert, no ban, the reason logged (probe_verdict), the next probe not before 10 s. Safe while it is unknown which robot is ours: a pure turn
  on the spot moves nobody across the field; if the candidate is another team's robot, ours only turns where it
  stands. characterize.py spin sends only an angular rate (zeros before and after), never drives to a start pose and
  uses its pose only to stop (stale, or outside SOLO_BOX). Two or more robots seen as our colour and
  id is no information (the connector may follow either): the session holds and alerts at once, "two robots are
  seen as blue 3: at (x, y) and (x, y), the tools cannot tell which is ours; the session holds". If the
  listener cannot be opened or the generated vision messages cannot be imported, the session does not start (without
  a home area: a warning that other robots are not watched). The robot the tools drive (the page's, through the
  connector's index) must stand within 0.25 m of ours in the raw vision (ours detected within the last 0.1 s; an
  older detection kept after its camera stopped showing it is not compared): at start, else the session does not start;
  during the session, else all motion stops, "the robot the tools are driving is not ours: ..." and the session ends.
  At start the page's heading must also agree with ours in the raw view within 0.5 rad, else the session does not
  start and the message names --vision-mirror (a wrong mirror flips y and the heading; skipped, logged, when either
  heading is missing).
  Start, left half: the same as above, e.g.
      python3 tools/solo/auto_session.py <team> tools/solo/sessions/left_noball.jsonl --budget 10 --our-id N --cameras C
      python3 tools/solo/auto_session.py <team> tools/solo/sessions/left_ball_slot_10.jsonl --budget 10 --our-id N --cameras C   (with the ball)
  Tell the other teams: our robot stays in its half; in a session with the ball it may cross briefly to fetch our ball
  when it rolled over (at most --fetch-limit m beyond our half), only while no robot is within 0.7 m of its way, and it
  backs off when one comes near; any robot
  that enters our half (or comes within 1 m of ours) stops our robot until our half has been clear for 5 s.
  At load: a step whose area (SOLO_BOX, each plan line's box, a goto's point, the placer's area, a parking spot) leaves
  the home area, its edges along the walls moved out to the placer's limit, is refused; so is a strategy step without
  "side" or with a node parameter the node does not declare, a series step with --resume, and a cmd step that is not
  on KNOWN_CMDS. The boxes of the placements the supervisor adds (parking, rescue) are clipped to that area, a fetch's
  to it plus --fetch-limit beyond the open edges. Dry run lists them.
  Every motion child (place, test, series, goto; the supervisor's own retry, fetch, retreat, drive back, freeing move,
  parking and rescue) runs watched: the raw view and /state are read 10 times a second; another robot inside the home
  area (its edges along the field's boundary moved out 0.15 m past the walls: our goal and the strip along the walls
  count) or within --intrude-dist of ours, or no information (no vision frame for 0.5 s, the page silent for 1 s, our
  robot not seen for 0.5 s), stops the step at once (the stop of the time limit: SIGINT to its group; characterize.py,
  place_ball.py and the strategy node send zeros), holds all motion, and the step is repeated from its start once the
  home area has been known and clear for --clear s (--clear-own s when the only cause of the hold was our robot not
  seen: the vision fresh throughout, no other robot inside or within --intrude-dist of the watched area; counted
  from our robot seen again, accepted as ours). A robot seen again within that time restarts the count. Alerts
  of a hold: every robot at once when it is first seen in it; no information without a break for --info-alert s,
  "no vision for N s: the session holds"; a hold that has lasted --info-alert s with nothing sent (information that
  comes and goes), "the session has held for N s without 5 s known and clear: <the causes seen>"; then a reminder
  every --hold-limit s (at most 3 in all). A fetch starts only once the home area,
  the surroundings of our robot and its corridor have been known and clear for --clear s, and runs with the same
  check plus the corridor. A drive back into the home area (after a fetch, a stop, a switch of halves) starts only
  while no other robot is inside the home area and its surroundings or its way back have been clear for --clear s;
  a robot entering the home area or the way stops it. While it waits the alert names the reason (no information,
  which robot is ours, a robot inside our half: after --info-alert s; robots on the way back: after --fetch-wait s)
  and where our robot stands ("...; our robot at (x, y), outside our half: the session waits"); every hold alert
  sent while our robot is outside the home area says where it stands. The alert rate: the robots first seen in one
  poll go in one alert; a repeat of one cause (kind and robot) goes out at once the first time, then not before 60 s,
  then not before 300 s, with the occurrences in between counted in its text ("(3 more times since the alert at
  HH:MM:SS)") and in "session ended"; at most one alert every 20 s, except the first of each kind, two robots seen
  as ours and which robot is ours lost: one held by this is merged into the next one sent (" | earlier: ...") or sent
  once the 20 s have passed (the start, the end of the session and "not ours" always go out at once).
  Switching halves: a session is written for the -x half (its "home"); every coordinate in it is mirrored at run time
  (x -> -x; boxes, plan lines, goto points, placement and parking spots, ram and step directions d -> 180 - d, the
  strategy side negative <-> positive; spin rates are kept; mirrored plans go to OUT/mirrored, the shipped ones are
  not changed). While the session holds for another robot inside our half: when that half has held a robot for
  --switch-after s (20; 0 = never; gaps under 1 s do not count), the other half has been known and empty of every robot
  but ours for that whole time, and the way over has been clear (--corridor) for --clear s, the home area becomes the
  other half; our robot drives to 0.3 m inside it (watched), a ball outside it is fetched as usual, the interrupted
  step is repeated there, steps done are not repeated. Not within --switch-cooldown s (60) of the last switch; after
  --max-switches (4) one alert and the session holds. Each switch is one quiet notice (SOLO_NOTICE_CMD, called as
  `CMD "message" "title"`; unset: a log line). --home auto: at start the half our robot stands in when no other robot
  is in it, else the other half when that one is empty (either for --clear s), else hold: no half chosen within
  --clear + --info-alert s, one alert naming the reason (no information, our robot not seen, both halves held), then
  reminders every --hold-limit s. The dry run
  lists both halves (its mirrored plans go to a temporary folder). A step test without its explicit start (fwd|side
  v x0 y0 dir length) is refused in a home session.
  Strategy steps: {"kind": "strategy", "on": true, "seconds": 10, "side": "negative", "params": {"force_state":
  "running", "foe_all": false}, "end_ball_out": "x0,x1,y0,y1", "edge_guard": 1.0}: the saved node line with those -p
  parameters replaced or added (checked by procs.check_node_cmd); nothing is restored afterwards. The node is switched
  off when the robot comes within edge_guard m (0.9) of an open edge (its fresh position only), on the motion rule
  above, or 0.5 s after the ball leaves end_ball_out.
  What the notification says in the new cases, and what the person does:
    "another robot (yellow 2) entered our half at (x, y); our robot stopped at (x, y)"      -> nothing; it resumes
    "another robot (yellow 2) still in our half at (x, y) after 120 s: the session keeps holding (reminder 2/3)"
    "ball at (x, y) in the other half, other robots near it (yellow 1 at (x, y)): fetch it when you can"
                                                                                -> bring our ball back into our half
    "ball at (x, y) is 1.3 m beyond our half (fetch limit 1.0 m): fetch it when you can" -> the same
    "no vision for 10.2 s: the session holds"                                     -> check the vision computer / network
    "two robots are seen as blue 3: at (x, y) and (x, y), the tools cannot tell which is ours; the session holds"
                                                         -> ask the other team to change its id, or change --our-id
    "which robot is ours was lost: ...; the session holds: put the robot back at (x, y) or restart the session"
    "the robot seen as blue 3 at (x, y) did not follow a commanded turn: it is not ours, or ours does not respond; the
     session holds"                         -> check our robot (powered, its radio), the other robot away from our half
                                                           -> put our robot back there, the other robot away from it
    "our robot recognised by its turn at (x, y): the session resumes"          -> keep clear of it: it moves again
    "the session has held for 10 s without 5 s known and clear: <causes>; the session holds"
                                                         -> check the vision of our robot (tag, lights, camera)
    "the robot the tools are driving is not ours: ..."      -> check --our-id and the connector (SOLO_OWN_COUNT, ids)
    "another robot (yellow 1 at (x, y)) blocks the way back; our robot at (x, y), outside our half: the session waits"
    "no alert could be delivered from 12:01:02 for 300 s: the session held all motion ..."  -> check the alert service
  Nothing is sent for a foreign ball in a no-ball session: a ball within 0.15 m of the next step's area holds it up to
  --ball-hold s, then the step moves to the end of the list; a ball inside the running step's area within 0.6 m of our
  robot stops the step (zeros), which moves to the end. series.py itself only delays a test's start while a ball is
  in its box ("REMOVE THE BALL"); characterize.py does not look at the ball during a step test.

Tests (no robot, no ROS): uv run --with pytest pytest -q tools/solo/tests/test_auto_session.py
"""
import json, math, os, random, re, signal, subprocess, sys, threading, time, urllib.request
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import procs as solo_procs                                 # noqa: E402  the commanding programs, the node's command line
from procs import is_instance, is_launcher                # noqa: E402,F401
REPO = os.path.dirname(os.path.dirname(HERE))
FX, FY = 2.25, 1.5                  # half field to the lines (m)
WALL = 0.30                         # boundary wall beyond the lines
GOAL_HW, GOAL_D = 0.40, 0.18        # goal half width, depth behind the goal line
DA_D, DA_HW = 0.5, 0.675            # defense area depth, half width
ROBOT_R = 0.09
PLACER_MARGIN = 0.25                # place_ball.py --margin default: how far past the lines it may push from
NEAR_WALL = 0.10                    # robot body within this of a wall / frame = "against it"
CLEAR_WALL = 0.25                   # body clearance that ends a stuck incident
FREE_IN = 0.5                       # the freeing goto ends this far inside the lines
COMMANDERS = solo_procs.COMMANDERS
MOTION = ("place", "test", "series", "goto", "strategy")
is_motion = lambda st: st["kind"] in MOTION and not (st["kind"] == "strategy" and not st.get("on", True))
TIMEOUTS = dict(place=90, test=60, series=900, goto=35, cmd=120, marker=5, wait=0)
clip = lambda v, lo, hi: lo if v < lo else hi if v > hi else v
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
sgn = lambda v, a, b: a if v >= 0 else b


# ------------------------------------------------------------------ geometry
def _seg(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0; L2 = dx * dx + dy * dy
    k = clip(((px - x0) * dx + (py - y0) * dy) / L2, 0.0, 1.0) if L2 else 0.0
    return math.hypot(px - x0 - k * dx, py - y0 - k * dy)

def obstacles(x, y):
    """[(clearance of the robot's body in m, name)] to the four boundary walls and the two goal frames, nearest first"""
    out = [(FX + WALL - x - ROBOT_R, "+x wall"), (x + FX + WALL - ROBOT_R, "-x wall"),
           (FY + WALL - y - ROBOT_R, "+y wall"), (y + FY + WALL - ROBOT_R, "-y wall")]
    for s, name in ((1, "+x goal frame"), (-1, "-x goal frame")):
        segs = [(s * FX, GOAL_HW, s * (FX + GOAL_D), GOAL_HW), (s * FX, -GOAL_HW, s * (FX + GOAL_D), -GOAL_HW),
                (s * (FX + GOAL_D), -GOAL_HW, s * (FX + GOAL_D), GOAL_HW)]
        out.append((min(_seg(x, y, *g) for g in segs) - ROBOT_R, name))
    return sorted(out)

def free_point(x, y):
    """the nearest point FREE_IN inside the lines: straight away from the wall or frame the robot is against"""
    return clip(x, -(FX - FREE_IN), FX - FREE_IN), clip(y, -(FY - FREE_IN), FY - FREE_IN)

def ball_class(b, margin=PLACER_MARGIN):
    """-> (class, where): field / defense / outside (placeable) or goal / frame / corner / wall / lost"""
    if b is None: return "lost", "ball not seen"
    x, y = b; ax, ay = abs(x), abs(y); sx, sy = sgn(x, "+x", "-x"), sgn(y, "+y", "-y"); at = f"at ({x:+.2f}, {y:+.2f})"
    if ax > FX and ay < GOAL_HW + 0.02: return "goal", f"ball in the {sx} goal {at}"
    if ax > FX - 0.03 and ay < GOAL_HW + 0.07: return "frame", f"ball against the {sx} goal frame {at}"
    if ax > FX and ay > FY: return "corner", f"ball in the ({sx}, {sy}) corner {at}"
    if ax > FX + margin: return "wall", f"ball against the {sx} wall {at}"
    if ay > FY + margin: return "wall", f"ball against the {sy} wall {at}"
    if ax > FX or ay > FY: return "outside", f"ball past the lines {at}"
    if ax > FX - DA_D and ay < DA_HW: return "defense", f"ball in the {sx} defense area {at}"
    return "field", f"ball {at}"

PLACEABLE = ("field", "defense", "outside")
IN_FIELD = ("field", "defense")

def reach_point(b):
    """the nearest point 0.3 m inside the lines, outside the defense areas"""
    x, y = clip(b[0], -(FX - 0.3), FX - 0.3), clip(b[1], -(FY - 0.3), FY - 0.3)
    if abs(x) > FX - DA_D - 0.1 and abs(y) < DA_HW + 0.1: x = math.copysign(FX - DA_D - 0.1, x)
    return round(x, 2), round(y, 2)

def placer_can(b, target):
    """ask BallPlacer (no ROS in it) whether it can push the ball from b toward target; None = cannot tell"""
    try:
        if HERE not in sys.path: sys.path.insert(0, HERE)
        from place_ball import BallPlacer
        p = BallPlacer(target)
        why = p.check_target() or p.check_ball(b)
        if why: return False
        plan = p.make_plan(b)
        return not isinstance(plan, str)
    except Exception:
        return None

def rects(spec):
    return [[float(v) for v in r.split(",")] for r in spec.split(";") if r.strip()] if spec else []

# ------------------------------------------------------------------ the home area (tests confined to part of a shared field)
CHAR_AREA = [-2.0, 2.0, -1.35, 1.35]                    # characterize.py's area without SOLO_BOX
PLACER_AREA = [-FX - PLACER_MARGIN, FX + PLACER_MARGIN, -FY - PLACER_MARGIN, FY + PLACER_MARGIN]
PLACER_LIM = min(PLACER_MARGIN, WALL - ROBOT_R - 0.03)  # how far past the lines the placer lets the robot's centre go

def home_rect(spec):
    """"x0,x1,y0,y1" -> [x0, x1, y0, y1]; None / "" = the whole field (no home checks)"""
    if not spec: return None
    r = [float(v) for v in spec.split(",")]
    if len(r) != 4 or r[0] >= r[1] or r[2] >= r[3]: raise ValueError(f"home area {spec!r}: want x0,x1,y0,y1 with x0 < x1, y0 < y1")
    return r

def in_rect(p, r, pad=0.0):
    return p is not None and r[0] - pad <= p[0] <= r[1] + pad and r[2] - pad <= p[1] <= r[3] + pad

def rect_in(r, home, tol=0.005):
    return r[0] >= home[0] - tol and r[1] <= home[1] + tol and r[2] >= home[2] - tol and r[3] <= home[3] + tol

def into(r, p, d=0.3):
    """the nearest point to p at least d inside the rectangle r"""
    return (round(clip(p[0], r[0] + d, r[1] - d), 2), round(clip(p[1], r[2] + d, r[3] - d), 2))

def rect_dist(p, r):
    return math.hypot(max(r[0] - p[0], 0.0, p[0] - r[1]), max(r[2] - p[1], 0.0, p[1] - r[3]))

def open_edges(home):
    """the edges of the home area that lie inside the field (more than 0.3 m from the lines): the side other teams use"""
    return dict(x0=home[0] > -FX + 0.3, x1=home[1] < FX - 0.3, y0=home[2] > -FY + 0.3, y1=home[3] < FY - 0.3)

def ball_area(home):
    """where a ball counts as ours: the home area, its edges along the field's boundary moved out to the walls"""
    e = open_edges(home); wx, wy = FX + WALL, FY + WALL
    return [home[0] if e["x0"] else -wx, home[1] if e["x1"] else wx, home[2] if e["y0"] else -wy, home[3] if e["y1"] else wy]

WATCH_PAST_WALL = 0.15              # how far beyond the walls the watched area reaches on the home area's closed edges

def watch_area(home):
    """where another robot counts as inside our half: the home area, its edges along the field's boundary moved out
    past the walls (our goal, the strip between the home area and the walls, the robot area our robot may use)"""
    e = open_edges(home); wx, wy = FX + WALL + WATCH_PAST_WALL, FY + WALL + WATCH_PAST_WALL
    return [home[0] if e["x0"] else min(home[0], -wx), home[1] if e["x1"] else max(home[1], wx),
            home[2] if e["y0"] else min(home[2], -wy), home[3] if e["y1"] else max(home[3], wy)]

def robot_area(home):
    """where our robot's centre may be: the home area, its edges along the field's boundary moved out to the placer's
    limit beyond the lines (a wall is not another team's side)"""
    e = open_edges(home); lx, ly = FX + PLACER_LIM, FY + PLACER_LIM
    return [home[0] if e["x0"] else min(home[0], -lx), home[1] if e["x1"] else max(home[1], lx),
            home[2] if e["y0"] else min(home[2], -ly), home[3] if e["y1"] else max(home[3], ly)]

def clip_rects(spec, area):
    return ";".join(",".join(f"{v:.2f}" for v in (max(r[0], area[0]), min(r[1], area[1]), max(r[2], area[2]), min(r[3], area[3]))) for r in rects(spec))

def near_open_edge(p, home, guard):
    """p is outside the home area, or within `guard` of one of its open edges"""
    if not in_rect(p, home): return True
    e = open_edges(home)
    return ((e["x0"] and p[0] < home[0] + guard) or (e["x1"] and p[0] > home[1] - guard) or
            (e["y0"] and p[1] < home[2] + guard) or (e["y1"] and p[1] > home[3] - guard))

def path_box(pts, pad=0.35):
    """a SOLO_BOX spec around the points, inside the placer's own area"""
    lx, ly = FX + PLACER_LIM, FY + PLACER_LIM
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    r = (max(-lx, min(xs) - pad), min(lx, max(xs) + pad), max(-ly, min(ys) - pad), min(ly, max(ys) + pad))
    return ",".join(f"{v:.2f}" for v in r)

def blockers(others, segs, d):
    """[(name, x, y, distance)] of the other robots within d of any of the segments [((x0, y0), (x1, y1))]"""
    out = []
    for name, x, y in others or ():
        m = min(_seg(x, y, a[0], a[1], b[0], b[1]) for a, b in segs)
        if m < d: out.append((name, round(x, 2), round(y, 2), round(m, 2)))
    return out

def intruders(others, home):
    """the other robots inside the watched area of `home` (watch_area)"""
    a = watch_area(home)
    return [(name, round(x, 2), round(y, 2)) for name, x, y in others or () if in_rect((x, y), a)]

def with_params(argv, params):
    """the node's command line with `-p name:=value` replaced (or added after --ros-args) for each of params"""
    a = list(argv)
    for k, v in params.items():
        v = str(v).lower() if isinstance(v, bool) else str(v); done = False
        for i in range(len(a) - 1):
            if a[i] in ("-p", "--param") and a[i + 1].split(":=", 1)[0] == k:
                a[i + 1] = f"{k}:={v}"; done = True
        if not done:
            if "--ros-args" not in a: a.append("--ros-args")
            a += ["-p", f"{k}:={v}"]
    return a

def widen(spec, d=0.15):
    """a SOLO_BOX spec d wider on every side, kept 0.12 m off the walls"""
    lx, ly = FX + WALL - 0.12, FY + WALL - 0.12
    rs = rects(spec) or [[-FX - PLACER_MARGIN, FX + PLACER_MARGIN, -FY - PLACER_MARGIN, FY + PLACER_MARGIN]]
    return ";".join(",".join(f"{v:.2f}" for v in (max(-lx, r[0] - d), min(lx, r[1] + d), max(-ly, r[2] - d), min(ly, r[3] + d))) for r in rs)

KNOWN_CMDS = (("robot_stack.sh", "save"),)              # cmd steps known not to move the robot: (program, first argument)
known_cmd = lambda argv: bool(argv) and (os.path.basename(str(argv[0])), str(argv[1]) if len(argv) > 1 else "") in KNOWN_CMDS

FETCH_ROOM = 0.3                    # room behind a ball the placer needs: a ball is fetched up to fetch_limit - this beyond our half

def fetch_area(home, lim):
    """where a fetch may take our robot: the robot area, its open edges moved out by lim"""
    e = open_edges(home); a = robot_area(home)
    return [a[0] - lim if e["x0"] else a[0], a[1] + lim if e["x1"] else a[1], a[2] - lim if e["y0"] else a[2], a[3] + lim if e["y1"] else a[3]]

def ball_in_box(b, spec, pad=0.15):
    return b is not None and any(r[0] - pad <= b[0] <= r[1] + pad and r[2] - pad <= b[1] <= r[3] + pad for r in rects(spec))

# ------------------------------------------------------------------ the other half: mirroring a session (x -> -x)
SIDE_SWAP = {"negative": "positive", "positive": "negative"}
_r4 = lambda v: round(float(v), 4) + 0.0

def mirror_dir(d):
    """a direction (deg) mirrored across the y axis: 180 - d, in (-180, 180]"""
    v = (180.0 - float(d) + 180.0) % 360.0 - 180.0
    return _r4(180.0 if v == -180.0 else v)

def mirror_rect(r):
    return [_r4(-r[1]), _r4(-r[0]), _r4(r[2]), _r4(r[3])]

def mirror_spec(spec):
    """a SOLO_BOX spec "x0,x1,y0,y1;..." mirrored ("" stays "")"""
    return ";".join(",".join(f"{v:g}" for v in mirror_rect(r)) for r in rects(spec)) if spec else spec

def mirror_args(test, args):
    """a characterize.py test's arguments mirrored: positions x -> -x, directions d -> 180 - d (spin rates kept)"""
    a = list(args)
    if test == "step" and len(a) >= 5: a[2] = _r4(-float(a[2])); a[4] = mirror_dir(a[4])
    elif test == "ram" and len(a) >= 2: a[1] = mirror_dir(a[1])
    elif test == "goto" and len(a) >= 3: a[0] = _r4(-float(a[0])); a[2] = mirror_dir(a[2])
    elif test == "goto" and len(a) >= 2: a[0] = _r4(-float(a[0]))
    elif test == "sweep" and len(a) >= 2: a[0], a[1] = _r4(-float(a[1])), _r4(-float(a[0]))
    return a

def mirror_pose(p):
    """[x, y] or [x, y, dir_deg] mirrored"""
    q = [_r4(-float(p[0])), _r4(p[1])]
    return q + [mirror_dir(p[2])] if len(p) > 2 else q

def mirror_plan_line(e):
    """one series.py plan line mirrored (box, brake_box, at, spot, back, args)"""
    e = dict(e)
    for k in ("box", "brake_box"):
        if e.get(k): e[k] = mirror_spec(e[k])
    for k in ("at", "spot", "back"):
        if e.get(k): e[k] = mirror_pose(e[k])
    if "args" in e: e["args"] = mirror_args(e.get("test"), e["args"])
    return e

MIRROR_ENV = ("SOLO_BOX", "SOLO_BRAKE_BOX")

def mirror_env(env):
    return {k: (mirror_spec(v) if k in MIRROR_ENV and isinstance(v, str) and "$" not in v else v) for k, v in (env or {}).items()}

def mirror_step(st, plan_dir=None, vs=None):
    """a session step mirrored into the other half; a series step's plan is written mirrored to plan_dir (the shipped
    plan is never changed). Internal marks of the load-time check (_boxes, _refused, _unchecked) are dropped"""
    m = {k: v for k, v in st.items() if k not in ("_boxes", "_refused", "_unchecked")}
    if "env" in m: m["env"] = mirror_env(m["env"])
    if m.get("keep_ball_out"): m["keep_ball_out"] = mirror_spec(m["keep_ball_out"])
    if m.get("end_ball_out"): m["end_ball_out"] = mirror_spec(m["end_ball_out"])
    if m.get("park"): m["park"] = mirror_pose(m["park"])
    k = m["kind"]
    if k in ("place", "goto"): m["x"] = _r4(-float(m["x"]))
    if k == "test": m["args"] = mirror_args(m.get("test"), m.get("args", []))
    if k == "strategy" and m.get("side") in SIDE_SWAP: m["side"] = SIDE_SWAP[m["side"]]
    m["_half"] = -st.get("_half", 1)
    if k == "series":
        src = os.path.abspath(expand(m["plan"], vs or {})); canon = st.get("plan_canonical", src); m["plan_canonical"] = canon
        if m["_half"] == 1:                              # mirrored back into the half it was written for: the shipped plan
            m["plan"] = canon; return m
        if plan_dir is None: raise ValueError("a series step needs a folder for its mirrored plan")
        os.makedirs(plan_dir, exist_ok=True)
        import hashlib
        stem = os.path.basename(canon); stem = stem[:-6] if stem.endswith(".jsonl") else stem
        dst = os.path.join(plan_dir, f"{stem}_{hashlib.sha1(canon.encode()).hexdigest()[:8]}_mx.jsonl"); out = []
        for l in open(canon):
            s = l.strip()
            if not s or s.startswith("#"): out.append(l.rstrip("\n")); continue
            out.append(json.dumps(mirror_plan_line(json.loads(s))))
        with open(dst, "w") as f: f.write("\n".join(out) + "\n")
        m["plan"] = dst
    return m

def mirror_head(head):
    h = dict(head)
    if h.get("home"): h["home"] = ",".join(f"{v:g}" for v in mirror_rect(home_rect(h["home"])))
    if h.get("park"): h["park"] = mirror_pose(h["park"])
    if "env" in h: h["env"] = mirror_env(h["env"])
    return h

def half_name(home):
    return "+x" if home[0] + home[1] > 0 else "-x"


# ------------------------------------------------------------------ processes (the checks are in procs.py)
class Procs:
    """the process table from /proc"""
    def table(self):
        return solo_procs.process_table()
    def mine(self):
        """this process and its ancestors"""
        return solo_procs.ancestors()
    def group(self, pgid):
        """the pids of the process group pgid (a step's process and what it started)"""
        out = set()
        for pid, _ in self.table():
            try:
                with open(f"/proc/{pid}/stat") as f:
                    if int(f.read().rsplit(")", 1)[1].split()[3]) == pgid: out.add(pid)
            except (OSError, ValueError, IndexError): pass
        return out

def busy(procs, names=COMMANDERS, exclude=()):
    """the commanding programs that run now, other than this process, its ancestors and the pids in exclude"""
    return solo_procs.running(names, procs.table(), set(procs.mine()) | set(exclude))


class Strategy:
    """switch the strategy node off and on the way hq.py does: SIGINT it (its command line is kept in
    $SOLO_RUN_DIR/solo.cmd) and start it again from that line; the policy server keeps running"""
    WAITS = (2.5, 1.5, 1.0)             # s after SIGINT, after SIGTERM, after SIGKILL
    def __init__(self, procs, run_dir=None, waits=None):
        self.procs = procs; self.run = run_dir or os.environ.get("SOLO_RUN_DIR", "/tmp/solo_run")
        self.cmdf = os.path.join(self.run, "solo.cmd"); self.child = None; self.waits = tuple(waits or self.WAITS)
    def pids(self):
        return [pid for pid, a in self.procs.table() if is_instance(a, "solo_node") or is_launcher(a)]
    def child_alive(self):
        return self.child is not None and self.child.poll() is None
    def running(self):
        return self.child_alive() or any(is_instance(a, "solo_node") for _, a in self.procs.table())
    def can_start(self):
        return self.running() or os.path.exists(self.cmdf)
    def off(self):
        """SIGINT to every node and launcher (the node sends its zeros), then SIGTERM, then SIGKILL (the process group
        of a node this process started, every other node pid by itself); -> True when no node is left"""
        for pid, a in self.procs.table():
            if is_instance(a, "solo_node") and len(a) > 1 and os.path.basename(a[1]) == "solo_node":
                try:
                    os.makedirs(self.run, exist_ok=True)
                    with open(self.cmdf, "w") as f: json.dump(a, f)
                except OSError: pass
        for sig, wait in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), self.waits):
            if not self.pids() and not self.child_alive(): break
            for pid in set(self.pids()) | ({self.child.pid} if self.child_alive() else set()):
                try: os.kill(pid, sig)
                except OSError: pass
            if self.child is not None and sig == signal.SIGKILL:  # the whole group this process created for it
                try: os.killpg(self.child.pid, sig)
                except OSError: pass
                self.child.poll()
            t0 = time.time()
            while (self.pids() or self.child_alive()) and time.time() - t0 < wait:
                time.sleep(0.05)
        return not self.running()
    def on(self, params=None):
        """start the node from the saved line, only when procs.check_node_cmd accepts it; else self.why says why.
        params {name: value} replace or add its `-p name:=value` (a running node is restarted with them); the changed
        line becomes the saved line when the node is next switched off. Never while an old node is still alive"""
        self.why = ""
        if self.running():
            if not params: return True
            if not self.off(): self.why = "the old node is still running"; return False
        if self.child_alive() or self.pids(): self.why = "the old node is still running"; return False
        try:
            with open(self.cmdf) as f: argv = json.load(f)
        except (OSError, ValueError) as e: self.why = f"no saved command line ({type(e).__name__})"; return False
        if params: argv = with_params(argv, params)
        self.last_argv = argv
        why = solo_procs.check_node_cmd(argv)
        if why: self.why = "saved command line refused: " + why; return False
        try:
            log = open(os.path.join(self.run, "solo.log"), "a")
            self.child = subprocess.Popen(argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e: self.why = f"cannot start: {e}"; return False
        try:
            with open(os.path.join(self.run, "solo.pid"), "w") as f: f.write(str(self.child.pid))
        except OSError: pass
        return True


# ------------------------------------------------------------------ the world
VISION_GROUP, VISION_PORT = "224.5.23.2", 10006    # the vision multicast (the other tools' defaults)
VISION_FRESH = 0.5                  # the raw view and our robot's position: older = no information
PAGE_FRESH = 1.0                    # the field page's /state: older = no information
MATCH_D = 0.25                      # the page's robot and ours in the raw vision: farther apart = not the same robot
MATCH_AGE = 0.1                     # s: an older detection of ours (kept after its camera stopped showing it) is not compared

def frame_camera(w):
    """the camera id of a detection frame"""
    return int(w.detection.camera_id)

def frame_robots(w, mirror=False, heading=False):
    """a parsed SSL_WrapperPacket -> [(colour, vision id, x, y)] of its detection frame (metres; y flipped when the
    connector runs with mirror:=true), None for a packet without a detection frame. heading: (colour, id, x, y, th),
    th the detection's orientation (rad; negated with mirror) or None when the frame carries none"""
    if not w.HasField("detection"): return None
    s = -1.0 if mirror else 1.0
    def th(r):
        try: ok = r.HasField("orientation")
        except Exception: ok = hasattr(r, "orientation")
        return s * float(r.orientation) if ok else None
    return [(col, int(r.robot_id), r.x / 1000.0, s * r.y / 1000.0) + ((th(r),) if heading else ())
            for col, rs in (("blue", w.detection.robots_blue), ("yellow", w.detection.robots_yellow)) for r in rs]

class McastSource:
    """the vision multicast, receive only: joins the group on the interface, never sends; decodes with the generated
    messages the repository ships (vision_map._pb). Raises when the socket or the modules are not available"""
    def __init__(self, group=VISION_GROUP, port=VISION_PORT, iface="0.0.0.0", mirror=False):
        import socket
        import vision_map
        self.pb = vision_map._pb(); self.mirror = mirror
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try: s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError): pass
        s.bind((group, int(port)))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(group) + socket.inet_aton(iface))
        self.s = s
    def read(self, timeout):
        import select
        r, _, _ = select.select([self.s], [], [], timeout)
        if not r: return None
        w = self.pb.SSL_WrapperPacket()
        try: w.ParseFromString(self.s.recv(65535))
        except Exception: return None
        rs = frame_robots(w, self.mirror, heading=True)
        return None if rs is None else (frame_camera(w), rs)

MERGE_D = 0.15                      # m: two cameras' sightings of one colour + id closer than this are one robot (a robot is 0.18 m across)
MERGE_SPEED = 2.0                   # m/s: allowed for the age difference of the two sightings ...
MERGE_CAP = 0.35                    # m: ... up to this distance
SPEED_MAX = 3.0                     # m/s: a sighting its own camera no longer confirms (newer frames without it): the robot moved on
LEFT_CAP = 0.8                      # m: ... up to this distance

def merge_dets(dets, flags=False):
    """[(colour, id, x, y, age, camera, left[, confirmed])] -> [(colour, id, x, y, age)] with the sightings of one robot merged
    (flags: [((colour, id, x, y, age), confirmed)], confirmed when one of its sightings is).
    Two sightings of one camera (one frame) are never merged: they are two robots. Sightings of two cameras with the
    same colour + id are merged when closer than MERGE_D plus MERGE_SPEED per second of age difference, at most
    MERGE_CAP; for an older sighting its camera no longer confirms (left: that camera's newer frames do not hold this
    colour + id) the allowance is SPEED_MAX per second (the robot left that camera's view), at most LEFT_CAP. The freshest
    sighting stands for the group. Tuples without camera count as camera 0"""
    groups = []                                        # [first det, set of cameras]
    for q in sorted(dets, key=lambda q: q[4]):
        c, i, x, y, a = q[:5]; cam = q[5] if len(q) > 5 else 0; left = bool(q[6]) if len(q) > 6 else False
        best = None
        for g in groups:
            o, cams = g[0], g[1]
            if c != o[0] or i != o[1] or cam in cams: continue
            dt = abs(a - o[4]); d = math.hypot(x - o[2], y - o[3])
            lim = min(MERGE_D + SPEED_MAX * dt, LEFT_CAP) if left else min(MERGE_D + MERGE_SPEED * dt, MERGE_CAP)
            if d < lim and (best is None or d < best[0]): best = (d, g)
        ok = bool(q[7]) if len(q) > 7 else True
        if best is not None: best[1][1].add(cam); best[1].append(ok)
        else: groups.append([(c, i, x, y, a), {cam}, ok])
    return [(g[0], any(g[2:])) for g in groups] if flags else [g[0] for g in groups]

class RawVision:
    """the supervisor's own view of the robots: every robot of both colours with its vision id, from the packets of a
    source (McastSource, or a fake in tests), independent of the vision connector. A source yields (camera id,
    robots) or robots (camera 0) (a fake: (camera id, robots, frame time)), a robot (colour, id, x, y) or (colour, id, x, y, heading). Every detection is kept,
    two robots with the same colour + id included: per camera and colour + id the sightings of the last frame that
    held that colour + id, kept `keep` s (a robot missed in a frame or two is not gone). snapshot: cam_age {camera:
    age of its last detection frame}; vision_age: the age of the stalest camera (listed in `cameras`, a camera never
    seen counting as never; without the list, the cameras seen so far; a camera that sends without being listed is
    listed from then on, `unlisted` names it); vision_dets: [(colour, id, x, y, age)] merged across cameras
    (merge_dets); vision_unconfirmed: those of vision_dets not confirmed yet; vision_poses: [(colour, id, x, y,
    heading, age)] of every kept sighting that carries a heading (not merged).
    Confirmed: a colour + id of one camera seen twice within PAIR_S (in any two frames, not necessarily consecutive;
    a robot detected in every other frame counts), or twice within SLOW_S with the second sighting within NEAR_D +
    SLOW_V x the gap of the first, and from then on until its sighting expires; or a sighting within
    NEAR_D of a sighting of another colour + id of the last PAIR_S (a robot whose id is read differently from frame
    to frame; the colour + id in `own`, our robot, does not confirm another); or one whose camera sends no newer
    frame within CONFIRM_S. A single isolated sighting is not confirmed.
    own_track: [(age, x, y)] of every sighting of `own` (any camera) of the last OWN_HIST s, oldest first: the
    supervisor follows our robot on these frames, also across the time it took no snapshot.
    thread=False: the source is read when a snapshot is taken (tests, simulated clock)"""
    def __init__(self, source, clock=time.time, keep=VISION_FRESH, thread=True, cameras=None):
        self.src, self.clock, self.keep = source, clock, keep; self.L = threading.Lock(); self.cam_t = {}; self.seen = {}
        self.cameras = None if cameras is None else sorted(int(c) for c in cameras); self.unlisted = []; self.own = None
        self.halt = threading.Event(); self.threaded = thread; self.own_hist = deque()
        if thread: threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while not self.halt.is_set():
            try: f = self.src.read(0.2)
            except Exception: f = None; self.halt.wait(0.2)
            if f is not None: self.feed(f)
    CONFIRM_S = 0.1                     # s: a sighting of one frame counts when its camera sends no newer frame this long
    PAIR_S = 0.25                       # s: two sightings of one colour + id within this confirm it
    NEAR_D = 0.2                        # m: a sighting this close to a recent one of another colour + id is confirmed
    SLOW_S, SLOW_V = 1.0, 2.0           # s, m/s: a sighting within NEAR_D + SLOW_V x the gap of the last one of its colour + id
                                        # (same camera) of the last SLOW_S s is confirmed (a robot detected only now and then)
    OWN_HIST = 10.0                     # s: the sightings of our colour + id kept for following our robot frame by frame
    def feed(self, robots):
        t = self.clock(); cam = 0
        if isinstance(robots, tuple) and len(robots) == 3 and isinstance(robots[1], list): cam, robots, t = robots   # a fake's frame time
        if isinstance(robots, tuple) and len(robots) == 2 and isinstance(robots[1], list): cam, robots = robots
        cam = int(cam); fr = {}
        for r in robots:
            col, i, x, y = r[:4]; th = r[4] if len(r) > 4 and r[4] is not None else None
            fr.setdefault((col, int(i)), []).append((float(x), float(y), None if th is None else float(th)))
        with self.L:
            self.cam_t[cam] = t
            if self.cameras is not None and cam not in self.cameras:
                self.cameras = sorted(self.cameras + [cam]); self.unlisted.append(cam)
            recent = [(k2, ps2) for (c2, k2), (t2, ps2, _) in self.seen.items() if k2 != self.own and t - t2 <= self.PAIR_S]
            for k, ps in fr.items():
                old = self.seen.get((cam, k))
                ok = old is not None and (t - old[0] <= self.PAIR_S or (old[2] and t - old[0] <= self.keep) or
                                          (t - old[0] <= self.SLOW_S and any(math.hypot(x - x0, y - y0) <= self.NEAR_D + self.SLOW_V * (t - old[0])
                                                                             for x0, y0, _ in old[1] for x, y, _ in ps)))
                if not ok and k != self.own:
                    ok = any(k2 != k and math.hypot(x - x2, y - y2) <= self.NEAR_D for k2, ps2 in recent for x2, y2, _ in ps2 for x, y, _ in ps)
                self.seen[(cam, k)] = (t, ps, ok)
                if k == self.own: self.own_hist.extend((t, x, y) for x, y, _ in ps)
            while self.own_hist and self.own_hist[0][0] < t - self.OWN_HIST: self.own_hist.popleft()
    def snapshot(self):
        if not self.threaded:
            for _ in range(1000):
                f = self.src.read(0)
                if f is None: break
                self.feed(f)
        t = self.clock()
        with self.L:
            cam_age = {c: round(t - tt, 2) for c, tt in self.cam_t.items()}
            cams = self.cameras if self.cameras is not None else sorted(cam_age)
            live = [(c, k, tt, ps, ok) for (c, k), (tt, ps, ok) in self.seen.items() if t - tt <= self.keep]
            dets = [(k[0], k[1], x, y, round(t - tt, 2), c, self.cam_t[c] > tt,
                     ok or (self.cam_t[c] == tt and t - tt >= self.CONFIRM_S - 1e-9)) for c, k, tt, ps, ok in live for x, y, _ in ps]
            poses = [(k[0], k[1], x, y, th, round(t - tt, 3)) for c, k, tt, ps, ok in live for x, y, th in ps if th is not None]
            unlisted = list(self.unlisted)
            track = [(round(t - tt, 3), x, y) for tt, x, y in sorted(self.own_hist)]
        ages = [cam_age.get(c, 1e9) for c in cams]; merged = merge_dets(dets, flags=True)
        return dict(vision_age=max(ages) if ages else 1e9, cam_age=cam_age, vision_dets=[d for d, ok in merged],
                    vision_unconfirmed=[d for d, ok in merged if not ok], vision_poses=poses, unlisted=unlisted, own_track=track)

def open_vision(group=VISION_GROUP, port=VISION_PORT, iface="0.0.0.0", mirror=False, cameras=None):
    """-> (RawVision, None) or (None, why it cannot be opened)"""
    try: return RawVision(McastSource(group, port, iface, mirror), cameras=cameras), None
    except ImportError as e: return None, f"the generated vision messages cannot be imported ({e}); see vision_map.py (SOLO_VISION_PB_DIR)"
    except OSError as e: return None, f"cannot listen to the vision multicast {group}:{port} on {iface} ({e})"


class PageWorld:
    """robot and ball from the field page's /state, polled in the background (10 Hz); ages kept here; with `vision`
    (RawVision) its view is part of every snapshot. A plain /state read (without ?page=1) does not refresh the page's
    dead man's switch: only the browser page keeps a target alive"""
    def __init__(self, url, clock=time.time, period=0.1, vision=None):
        self.vision = vision
        self.url = url.rstrip("/") + "/state"; self.clock = clock; self.L = threading.Lock()
        self.robot = None; self.robot_t = -1e9; self.ball = None; self.ball_t = -1e9; self.page_t = -1e9
        self.hist = deque(maxlen=60); self.strategy = None; self.page_driving = False; self.halt = threading.Event()
        self.others = []; self.others_t = -1e9
        self.period = period                # 10 polls a second: the other robots are watched during every motion step
        self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()
    def stop(self, wait=2.0):
        """end the polling thread"""
        self.halt.set(); self.thread.join(wait)
    def _loop(self):
        while not self.halt.is_set():
            try:
                d = json.loads(urllib.request.urlopen(self.url, timeout=1.0).read())
                t = self.clock()
                with self.L:
                    self.page_t = t; self.strategy = d.get("strategy")
                    self.page_driving = d.get("target") is not None or bool((d.get("place") or {}).get("active"))
                    if d.get("us") is not None and d.get("us_age") is not None:
                        self.robot = tuple(d["us"]); self.robot_t = t - float(d["us_age"])
                    if d.get("ball") is not None:
                        self.ball = tuple(d["ball"]); self.ball_t = t; self.hist.append((t, *self.ball))
                    if isinstance(d.get("others"), dict):        # robots seen within the last second (hq.py)
                        self.others = [(k, float(v[0]), float(v[1])) for k, v in d["others"].items()]; self.others_t = t
            except Exception:
                pass
            self.halt.wait(self.period)
    def snapshot(self):
        t = self.clock()
        with self.L:
            h = [p for p in self.hist if t - p[0] < 1.5]
            still = len(h) >= 4 and h[-1][0] - h[0][0] > 1.0 and max(math.hypot(p[1] - h[-1][1], p[2] - h[-1][2]) for p in h) < 0.02
            d = dict(robot=self.robot, robot_age=round(t - self.robot_t, 2), ball=self.ball, ball_age=round(t - self.ball_t, 2),
                     ball_still=still, page_age=round(t - self.page_t, 2), page_driving=self.page_driving,
                     others=list(self.others) if t - self.others_t < 1.0 else [])
        if self.vision is not None: d.update(self.vision.snapshot())
        return d


class ScriptedWorld:
    """for tests: robot and ball set by hand or by a timeline [(t, {"robot": .., "ball": ..})] on a simulated clock;
    None hides an object (its age then grows). The raw vision view: from `vision` (a RawVision on a fake packet
    source) when given, else made from the robot (colour+id `our`) and `others` ("yellow2" = yellow id 2);
    vision=False / page=False: that source is silent from then on"""
    def __init__(self, clock, robot=(0.0, 0.6, 0.0), ball=(1.2, -0.8), timeline=(), others=(), vision=None, our=("blue", 0)):
        self.clock = clock; self.robot = robot; self.ball = ball; self.robot_t = self.ball_t = clock(); self.others = list(others)
        self.robot_on = robot is not None; self.ball_on = ball is not None; self.timeline = sorted(timeline, key=lambda e: e[0])
        self.vision = vision; self.our = our; self.vision_on = True; self.page_on = True; self.vision_t = self.page_t = clock()
        self.raw_robot = None                                       # set: our robot in the raw view elsewhere than on the page
        self.coasting = None
    def coast(self, v, react, decel):
        """the robot keeps its velocity v (vx, vy) for react s, then brakes at decel m/s^2 to a standstill (a stopped
        motion child: the robot's reaction delay and braking); set(robot=...) ends it"""
        self.coasting = dict(t0=self.clock(), p0=tuple(self.robot), v=tuple(v), react=react, decel=decel)
    def _coast(self):
        c = self.coasting
        if c is None or not self.robot_on: return
        sp = math.hypot(*c["v"]); dt = self.clock() - c["t0"]
        if sp < 1e-9: self.coasting = None; return
        tb = min(max(0.0, dt - c["react"]), sp / c["decel"])
        d = sp * min(dt, c["react"]) + sp * tb - 0.5 * c["decel"] * tb * tb
        p0 = c["p0"]; self.robot = (p0[0] + c["v"][0] / sp * d, p0[1] + c["v"][1] / sp * d) + tuple(p0[2:]); self.robot_t = self.clock()
        if dt - c["react"] >= sp / c["decel"]: self.coasting = None
    def set(self, **kw):
        t = self.clock()
        if "robot" in kw: self.coasting = None
        if "vision" in kw: self.vision_on = bool(kw["vision"])
        if "page" in kw: self.page_on = bool(kw["page"])
        if "raw_robot" in kw: self.raw_robot = kw["raw_robot"]
        if "robot" in kw:
            self.robot_on = kw["robot"] is not None
            if self.robot_on: self.robot = tuple(kw["robot"]); self.robot_t = t
        if "ball" in kw:
            self.ball_on = kw["ball"] is not None
            if self.ball_on: self.ball = tuple(kw["ball"]); self.ball_t = t
        if "others" in kw: self.others = list(kw["others"])          # [(name, x, y)], e.g. ("yellow2", 0.4, 0.3)
    def snapshot(self):
        self._coast()
        while self.timeline and self.timeline[0][0] <= self.clock(): self.set(**self.timeline.pop(0)[1])
        t = self.clock()
        if self.page_on:
            self.page_t = t
            if self.robot_on: self.robot_t = t
            if self.ball_on: self.ball_t = t
        d = dict(robot=self.robot, robot_age=round(t - self.robot_t, 2), ball=self.ball, ball_age=round(t - self.ball_t, 2),
                 ball_still=self.ball_on and self.page_on, page_age=round(t - self.page_t, 2), others=list(self.others))
        if self.vision is not None: d.update(self.vision.snapshot()); return d
        if self.vision_on:
            self.vision_t = t; vr = []
            rr = self.raw_robot if self.raw_robot is not None else self.robot
            if rr is not None and (self.robot_on or t - self.robot_t <= VISION_FRESH):     # the raw view keeps a robot 0.5 s
                vr.append((self.our[0], int(self.our[1]), rr[0], rr[1], round(t - self.robot_t, 2) if not self.robot_on else 0.0))
            for name, x, y in self.others:
                m = re.match(r"([a-z]+)(\d+)$", name)
                if m: vr.append((m.group(1), int(m.group(2)), x, y, 0.0))
            self.vision_dets = vr
        d.update(vision_age=round(t - self.vision_t, 2), cam_age={0: round(t - self.vision_t, 2)},
                 vision_dets=list(getattr(self, "vision_dets", [])) if self.vision_on else [])
        return d


class FakePacketSource:
    """for tests: the packets of the vision multicast without a socket. frames(t) -> [(colour, id, x, y)] of the
    detection frame at simulated time t, or None (no camera frame); one frame per `period` s"""
    def __init__(self, clock, frames, period=1 / 60):
        self.clock, self.frames, self.period, self.last = clock, frames, period, -1e9
    def read(self, timeout):
        t = self.clock()
        if t - self.last < self.period - 1e-9: return None
        self.last = t; return self.frames(t)


# ------------------------------------------------------------------ running a step
def result_of(lines):
    """the result line: the last JSON line, else the last ABORT / failure line, else the last line"""
    for l in reversed(lines):
        if l.startswith("{"):
            try: return json.loads(l), l
            except ValueError: pass
    keys = ("ABORT", "refusing", "failed", "not in vision", "no ball", "arrived:", "timeout:", "vision lost", "series done", "stopped after", "Error")
    for l in reversed(lines):
        if any(k in l for k in keys): return None, l
    return None, (lines[-1] if lines else "")

class Runner:
    """each step a subprocess in its own process group with a hard time limit"""
    def __init__(self, echo=True):
        self.echo = echo; self.p = None; self.kill_now = False
    def run(self, argv, env, timeout, tee=None, watch=None):
        """watch: called every 0.1 s while the step runs; a non-empty string it returns stops the step at once (the
        same stop as the time limit) and is returned as 'stopped'"""
        t0 = time.time(); lines = []; self.kill_now = False; stopped = None
        try:
            p = self.p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                          text=True, bufsize=1, start_new_session=True)
        except OSError as e:
            return dict(rc=127, timed_out=False, line=f"cannot start: {e}", data=None, t0=t0, t1=time.time(), tail=[])
        def read():
            for l in p.stdout:
                l = l.rstrip("\n"); lines.append(l)
                if self.echo: print("    | " + l, flush=True)
        th = threading.Thread(target=read, daemon=True); th.start()
        timed_out = False
        while p.poll() is None:
            if time.time() - t0 > timeout or self.kill_now:
                timed_out = True; self.stop(p); break
            if watch:
                try: stopped = watch()
                except Exception as e: stopped = f"no information: the watch failed ({e!r})"     # fail safe: stop
                if stopped: self.stop(p); break
            time.sleep(0.1)
        th.join(2.0); self.p = None
        if tee:
            try:
                with open(tee, "a") as f: f.write("\n".join(lines) + "\n")
            except OSError: pass
        data, line = result_of(lines)
        if stopped: line = f"stopped by the supervisor: {stopped}"
        return dict(rc=p.returncode, timed_out=timed_out, line=line, data=data, t0=t0, t1=time.time(), tail=lines[-5:], stopped=stopped)
    def stop(self, p):
        """the step's process group: SIGINT, 1.5 s to end; SIGTERM, 1.5 s; then SIGKILL. place_ball.py sends zeros for
        0.3 s on SIGINT and on SIGTERM; after SIGKILL nothing is sent, and the robot's own command timeout stops it.
        Once the step's own process has ended, SIGKILL to what is left of its group (a child it started that is
        still running). Then 0.5 s for the robot's command timeout"""
        for sig, wait in ((signal.SIGINT, 1.5), (signal.SIGTERM, 1.5), (signal.SIGKILL, 1.0)):
            try: os.killpg(p.pid, sig)
            except OSError: break
            try: p.wait(wait); break
            except subprocess.TimeoutExpired: pass
        try: os.killpg(p.pid, signal.SIGKILL)
        except OSError: pass
        time.sleep(0.5)
    def abort(self):
        self.kill_now = True
    def child_pid(self):
        """the pid (and process group) of the running step, or None"""
        p = self.p
        return p.pid if p is not None and p.poll() is None else None


class Notifier:
    """alerts through SOLO_NOTIFY_CMD; quiet notices through SOLO_NOTICE_CMD (optional: unset = a log line only)"""
    def __init__(self, cmd=None, notice_cmd=None):
        self.cmd = cmd if cmd is not None else os.environ.get("SOLO_NOTIFY_CMD", "")
        self.notice_cmd = notice_cmd if notice_cmd is not None else os.environ.get("SOLO_NOTICE_CMD", "")
    def notice(self, msg, title):
        if not self.notice_cmd: print(f"notice: {title}: {msg}", flush=True); return "log only (SOLO_NOTICE_CMD not set)"
        try: return f"rc {subprocess.run([self.notice_cmd, msg, title], capture_output=True, text=True, timeout=30).returncode}"
        except Exception as e: return f"error {e}"
    def send(self, msg, title):
        """-> "rc 0" when sent; a failing command (non-zero exit, an error, no answer in 30 s) is tried once more,
        and the result of both tries is returned (the session logs it)"""
        if not self.cmd:
            print(f"\n*** ALERT (SOLO_NOTIFY_CMD not set): {title}: {msg}\n", flush=True); return "not set"
        out = []
        for _ in range(2):
            try: r = f"rc {subprocess.run([self.cmd, msg, title], capture_output=True, text=True, timeout=30).returncode}"
            except Exception as e: r = f"error {e}"
            if r == "rc 0": return r if not out else f"rc 0 (second try; first: {out[0]})"
            out.append(r)
        print(f"\n*** ALERT (SOLO_NOTIFY_CMD failed: {'; '.join(out)}): {title}: {msg}\n", flush=True)
        return "failed: " + "; ".join(out)

DELIVERED = ("rc 0", "ok", "not set", "no notifier")

class AlertSender:
    """delivers the session's alerts in order from a queue so that the supervisor never waits for the alert command:
    a background thread (thread=False: pump() is called from the supervisor's polls and after every put, for a
    simulated clock). A failed send is retried after RETRY s (then every RETRY[-1] s), the order kept; an alert older
    than max_age s is dropped (unless kept) and counted, and the next one delivered starts with "N earlier alerts were
    not delivered since HH:MM". Quiet notices go through the same queue, tried once. failing_for(): how long alerts
    have been waiting without a delivery: since a send failed, or the age of the oldest alert in the queue, whichever
    is longer (0 = none waiting). alive(): the thread runs; start(): a new thread"""
    RETRY = (15.0, 30.0, 60.0)
    def __init__(self, notifier, clock=time.time, log=None, thread=True, max_age=600.0):
        self.notifier = notifier; self.clock = clock; self.logf = log or (lambda kind, **kw: None); self.max_age = max_age
        self.q = deque(); self.L = threading.Lock(); self.wake = threading.Event(); self.halt = threading.Event()
        self.fails = 0; self.next_t = -1e18; self.fail_since = None; self.dropped = 0; self.drop_since = None
        self.delivered = 0; self.busy = False; self.threaded = thread; self.thread = None
        if thread: self.start()
    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()
    def alive(self):
        return not self.threaded or (self.thread is not None and self.thread.is_alive())
    def put(self, msg, title, keep=False, front=False, notice=False):
        it = dict(msg=msg, title=title, t=self.clock(), keep=keep, notice=notice)
        with self.L:
            if front: self.q.appendleft(it)
            else: self.q.append(it)
        if self.threaded: self.wake.set()
        else: self.pump()
    def pending(self):
        with self.L: return len(self.q)
    def failing_for(self, now=None):
        now = self.clock() if now is None else now
        with self.L:
            ts = [it["t"] for it in self.q if not it["notice"]]
            if self.fail_since is not None and self.q: ts.append(self.fail_since)
        return max(0.0, now - min(ts)) if ts else 0.0
    def _loop(self):
        while not self.halt.is_set():
            try: self.pump()
            except Exception as e: self.logf("alert_sender_error", error=repr(e))
            self.wake.wait(0.2); self.wake.clear()
    def pump(self):
        """deliver what is due, head of the queue first; stops at the first failure (retried later)"""
        while True:
            now = self.clock()
            with self.L:
                if not self.q or self.busy: return
                it = self.q[0]
                if not it["notice"] and not it["keep"] and now - it["t"] > self.max_age:
                    self.q.popleft(); self.dropped += 1; self.drop_since = it["t"] if self.drop_since is None else self.drop_since
                    self.logf("alert_dropped", message=it["msg"], age=round(now - it["t"]), dropped=self.dropped); continue
                if not it["notice"] and now < self.next_t: return
                self.busy = True
            try:
                if it["notice"]:
                    n = self.notifier
                    r = n.notice(it["msg"], it["title"]) if n is not None and hasattr(n, "notice") else "log only"
                    self.logf("notice_sent", message=it["msg"], sent=r)
                    with self.L: self.q.remove(it)
                    continue
                msg = it["msg"]
                if self.dropped:
                    since = time.strftime("%H:%M", time.localtime(self.drop_since))
                    msg = f"{self.dropped} earlier alert{'s were' if self.dropped > 1 else ' was'} not delivered since {since}: {msg}"
                t_try = self.clock()
                try: r = self.notifier.send(msg, it["title"]) if self.notifier is not None else "no notifier"
                except Exception as e: r = f"error {e!r}"
                ok = str(r).startswith(DELIVERED)
                with self.L:
                    if ok:
                        self.q.remove(it); self.fails = 0; self.next_t = -1e18; self.fail_since = None
                        self.dropped = 0; self.drop_since = None; self.delivered += 1
                    else:
                        self.fail_since = t_try if self.fail_since is None else self.fail_since
                        self.next_t = self.clock() + self.RETRY[min(self.fails, len(self.RETRY) - 1)]; self.fails += 1
                if ok: self.logf("alert_delivered", message=msg, sent=r, waited=round(t_try - it["t"], 1))
                else:
                    self.logf("notify_failed", message=msg, result=r, tries=self.fails, retry_in=round(self.next_t - self.clock(), 1))
                    return
            finally:
                with self.L: self.busy = False
    def drain(self, limit, sleep=time.sleep):
        """wait up to limit s (on the sender's clock) for the queue to empty; -> True when it did"""
        t0 = self.clock()
        while self.pending() and self.clock() - t0 < limit:
            if not self.threaded: self.pump()
            if not self.pending(): break
            sleep(0.5 if self.threaded else min(1.0, max(0.01, limit - (self.clock() - t0))))
        return not self.pending()


# ------------------------------------------------------------------ the session
def load_session(path, team, extra_vars=None):
    head, steps = {}, []
    for k, l in enumerate(open(path), 1):
        l = l.strip()
        if not l or l.startswith("#"): continue
        d = json.loads(l)
        if d.get("kind") == "session": head = d; continue
        d.setdefault("n", len(steps) + 1); d["line"] = k; steps.append(d)
    vs = dict(os.environ); vs.update(TEAM=team, T=HERE, SOLO_DATA=data_dir(), SESSION_DIR=os.path.dirname(os.path.abspath(path)))
    for k, v in head.get("vars", {}).items(): vs[k] = expand(v, vs)
    vs.update(extra_vars or {})
    return head, steps, vs

def data_dir():
    return os.environ.get("SOLO_DATA") or os.path.join(REPO, "log", "solo")

def expand(v, vs):
    if isinstance(v, str): return re.sub(r"\$\{(\w+)\}", lambda m: str(vs.get(m.group(1), m.group(0))), v)
    if isinstance(v, list): return [expand(x, vs) for x in v]
    if isinstance(v, dict): return {k: expand(x, vs) for k, x in v.items()}
    return v


class Session:
    def __init__(self, team, steps, head=None, vs=None, world=None, runner=None, notifier=None, procs=None, strategy=None,
                 log_path=None, events_path=None, budget_s=600.0, clock=time.time, sleep=time.sleep, python="python3",
                 ball_lost=5.0, robot_lost=2.0, robot_wait=10.0, busy_wait=20.0, realert=180.0, max_alerts=3, echo=True,
                 home=None, no_ball=None, corridor=0.7, fetch_wait=60.0, fetch_tries=3, park=None, intrude_dist=1.0,
                 clear_s=5.0, hold_limit=120.0, ball_hold=30.0, ball_guard=0.6, edge_guard=0.9, our_id=None, vision_error=None,
                 robot_fresh=VISION_FRESH, info_alert=10.0, fetch_limit=1.0, switch_after=20.0, switch_cooldown=60.0, max_switches=4,
                 home_auto=False, cameras=None, alert_dead=300.0, alert_thread=None, settle=1.5, max_speed=2.0,
                 reappear=0.15, probes=6, probe_rate=1.5, probe_time=0.7, clear_own=1.5):
        self.cameras = None if cameras is None else sorted({int(c) for c in cameras})
        self.log_lock = threading.Lock(); self.alert_keys = {}; self.kinds_sent = set(); self.floor_buf = []; self.t_out = -1e18
        self.last_send = None; self.alert_dead = alert_dead; self.dead = None; self.sender_restarts = 0
        self.us_last = None; self.ours_lost = None; self.twins_seen = False; self.found_due = None    # telling our robot apart (snap)
        self.moving = False; self.moves = []; self.settle = settle; self.max_speed = max_speed   # motion intervals [start, end]
        self.holding = None                                                     # the current hold: since, cause
        self.reappear, self.clear_own = reappear, clear_own
        self.probes, self.probe_rate, self.probe_time = int(probes), float(probe_rate), float(probe_time)
        self.probes_run = 0; self.probe_bad = []; self.probe_cand = None; self.probes_out_sent = False   # the identity probe
        self.probe_none = 0; self.probe_none_why = None; self.probe_wait = -1e18      # probes without a verdict
        self.cameras_given = None if self.cameras is None else list(self.cameras); self.cams_logged = set()
        self.switch_after = switch_after if switch_after else None; self.switch_cooldown, self.max_switches = switch_cooldown, max_switches
        self.home_auto = home_auto; self.half_sign = 1; self.switches = []; self.notices = []; self.pending = None; self.current = None
        self.our_id = None if our_id is None else int(our_id); self.vision_error = vision_error
        self.robot_fresh, self.info_alert, self.fetch_limit = robot_fresh, info_alert, fetch_limit; self.not_ours = None
        self.team = team; self.steps = steps; self.head = head or {}; self.vs = vs or dict(TEAM=team, T=HERE, SOLO_DATA=data_dir())
        self.world = world; self.runner = runner; self.notifier = notifier; self.procs = procs; self.strat = strategy
        v = getattr(world, "vision", None)
        if isinstance(v, RawVision) and self.our_id is not None: v.own = (team, self.our_id)   # ours does not confirm another robot
        self.clock, self.sleep = clock, sleep; self.py = python; self.echo = echo
        self.ball_lost, self.robot_lost, self.robot_wait, self.busy_wait = ball_lost, robot_lost, robot_wait, busy_wait
        self.realert, self.max_alerts = realert, max_alerts
        self.t_start = clock(); self.t_end = self.t_start + budget_s
        self.log_path = log_path; self.events_path = events_path
        self.stop_req = 0; self.stop_why = None; self.motion_halted = False
        self.ball_incident = None; self.skipped_groups = set(); self.alerts = []; self.done = []; self.skipped = []
        self.retried_place = set(); self.bad_ball = None    # where the placer said "unreachable: ball"
        self.home = home_rect(home) if isinstance(home, str) else home
        self.no_ball = bool(no_ball if no_ball is not None else self.head.get("ball") == "ignore")
        self.corridor, self.fetch_wait, self.fetch_tries, self.intrude_dist = corridor, fetch_wait, fetch_tries, intrude_dist
        self.clear_s, self.hold_limit, self.ball_hold, self.ball_guard, self.edge_guard = clear_s, hold_limit, ball_hold, ball_guard, edge_guard
        self.park = tuple(park) if park else (tuple(self.head["park"]) if self.head.get("park") else (into(self.home, (-1.0, -1.0)) if self.home else None))
        self.fetch_inc = None; self.intrusions = []
        self.sender = AlertSender(None, clock, lambda kind, **kw: self.log(kind, **kw), thread=(clock is time.time) if alert_thread is None else alert_thread)
        self.prepare()

    # ---- records
    def log(self, kind, **kw):
        d = dict(t=round(self.clock(), 3), clock=time.strftime("%H:%M:%S", time.localtime(self.clock())), kind=kind, **kw)
        with self.log_lock:
            if self.echo: print(f"[{d['clock']}] {kind}: " + json.dumps(kw)[:300], flush=True)
            if self.log_path:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                with open(self.log_path, "a") as f: f.write(json.dumps(d) + "\n")
        return d
    ALERT_BACKOFF = (60.0, 300.0)       # s: a repeat of one cause (kind, robot) not before this after its 1st, its 2nd and later alerts
    ALERT_FLOOR = 20.0                  # s: at most one alert this often, except the kinds in ALERT_NOW and the first of each kind
    ALERT_NOW = ("twins", "ours_lost", "probe_fail", "ours_found")   # a person must get these at once (and the urgent ones: start, end, not ours)
    def alert_due(self, kind, robot=None, since=None):
        """the backoff of one cause: -> True when an alert with key (kind, robot) may go out now; else the occurrence
        (since: when the cause began; one count per since, however often the caller repeats it) is counted"""
        st = self.alert_keys.get((kind, robot)); now = self.clock(); since = now if since is None else since
        if st is None or now - st["t"] >= self.ALERT_BACKOFF[min(st["n"], len(self.ALERT_BACKOFF)) - 1]: return True
        if st["occ"] != since:
            st["occ"] = since; st["between"] += 1
            self.log("alert_held", of=kind, robot=robot, held=st["between"], next_in=round(st["t"] + self.ALERT_BACKOFF[min(st["n"], len(self.ALERT_BACKOFF)) - 1] - now, 1))
        return False
    LOST = "which robot is ours was lost"
    ALERT_LASTING = 20.0                # s: a cause that has lasted this long goes out whatever its backoff (force)
    def alert(self, msg, *, kind, robot=None, since=None, title="field session", urgent=False, force=False):
        """an alert -> True when it was accepted (queued for the sender, or merged into the next one sent), False when
        the backoff of its cause holds it (alert_due: per (kind, robot), at once the first time, then not before 60 s,
        then not before 300 s; the occurrences in between are counted in the text of the next one). robot may be a
        list (one alert naming several robots, one key each). The rate floor: within ALERT_FLOOR s of the last alert
        sent, an alert of a kind already sent this session (except ALERT_NOW) waits and is merged into the next one
        sent (at the latest when the floor has passed). The caller repeats an alert held by the backoff while its cause
        lasts. force: the backoff is skipped (a cause that has lasted ALERT_LASTING s; the rate floor still applies).
        urgent (the start, the end, not ours, a node that does not stop): always sent at once. While which robot is
        ours is lost: an ours_lost alert of a loss not yet alerted skips the backoff (a new loss goes out at once), and
        any alert accepted with the lost message in its text marks the loss alerted (its recovery is alerted too)"""
        now = self.clock(); robots = robot if isinstance(robot, list) else [robot]
        L = self.ours_lost; lost = L is not None and (kind == "ours_lost" or self.LOST in msg)
        if kind == "ours_lost" and L is not None and not L.get("alerted"): force = True
        if not urgent:
            if not force and not all([self.alert_due(kind, r, since) for r in robots]): return False
            between, t_prev = 0, None
            for r in robots:
                st = self.alert_keys.setdefault((kind, r), dict(n=0, t=now, between=0, occ=None))
                if st["n"]:
                    if st["occ"] == since and st["between"]: st["between"] -= 1      # the one counted is this one
                    between += st["between"]; t_prev = st["t"] if t_prev is None else min(t_prev, st["t"])
                st.update(n=st["n"] + 1, t=now, between=0, occ=None)
            if between:
                msg += f" ({between} more time{'s' if between > 1 else ''} since the alert at {time.strftime('%H:%M:%S', time.localtime(t_prev))})"
            first = kind not in self.kinds_sent
            if not first and kind not in self.ALERT_NOW and now - self.t_out < self.ALERT_FLOOR:
                self.floor_buf.append(msg); self.kinds_sent.add(kind)
                self.log("alert_merged", message=msg, of=kind, robot=robot, note=f"sent with the next alert (at most one every {self.ALERT_FLOOR:.0f} s)")
                if lost: L["alerted"] = True
                return True
        self.kinds_sent.add(kind); self.dispatch(msg, title, kind, robot)
        if lost: L["alerted"] = True
        return True
    def dispatch(self, msg, title="field session", kind=None, robot=None):
        """queue one alert for the sender, the alerts the rate floor held merged into it"""
        if self.floor_buf:
            msg += " | earlier: " + " | ".join(self.floor_buf); self.floor_buf = []
        self.t_out = self.clock(); self.alerts.append(msg)
        self.sender.notifier = self.notifier
        self.log("alert", message=msg, title=title, of=kind, robot=robot, sent="queued")
        self.sender.put(msg, title)
    def alert_poll(self):
        """called from every poll of the world: the alerts the rate floor held once it has passed; the sender (a
        simulated clock); a dead alert channel (no alert delivered for alert_dead s while one fails or waits); the
        sender's thread alive (restarted once; if it ends again the channel is dead)"""
        now = self.clock()
        if self.floor_buf and now - self.t_out >= self.ALERT_FLOOR:
            m = self.floor_buf.pop(0); self.dispatch(m, kind="merged")
        self.sender.notifier = self.notifier
        if not self.sender.threaded: self.sender.pump()
        f = self.sender.failing_for(now)
        if not self.sender.alive():
            if not self.sender_restarts:
                self.sender_restarts += 1; self.log("alert_sender_restarted", note="the alert sender's thread had ended: started again")
                self.sender.start()
            elif self.dead is None:
                self.dead = dict(since=now, fail_since=now - f, thread=True)
                self.log("alert_channel_dead", failing_for=round(f, 1), note="the alert sender's thread ended again: no motion step runs")
            return
        if self.dead is None and self.alert_dead and f >= self.alert_dead:
            since = time.strftime("%H:%M:%S", time.localtime(now - f))
            self.dead = dict(since=now, fail_since=now - f)
            self.log("alert_channel_dead", failing_for=round(f, 1), note="no alert delivered: no motion step runs until one is")
            self.sender.put(f"no alert could be delivered from {since} for {f:.0f} s: the session held all motion (robot standing) "
                            f"from {time.strftime('%H:%M:%S', time.localtime(now))} until alerts were delivered again", "field session", keep=True, front=True)
        elif self.dead is not None and not self.dead.get("thread") and f == 0.0 and not self.sender.pending():
            self.log("alert_channel_back", dead_for=round(now - self.dead["since"], 1)); self.dead = None
    def dead_text(self):
        """with a home area: why motion holds for the alert channel, or None"""
        if self.dead is None: return None
        why = "the alert sender stopped" if self.dead.get("thread") else "the alert command fails"
        return f"no alert delivered since {time.strftime('%H:%M:%S', time.localtime(self.dead['fail_since']))}: {why}"
    def send_now(self, msg, title="field session", kind="start"):
        """the start alert: sent synchronously (the session starts only when it is delivered)"""
        r = self.notifier.send(msg, title) if self.notifier else "no notifier"
        self.last_send = r; self.alerts.append(msg); self.t_out = self.clock(); self.kinds_sent.add(kind)
        self.log("alert", message=msg, title=title, of=kind, sent=r)
        if not str(r).startswith(DELIVERED): self.log("notify_failed", message=msg, result=r)
        return r
    def notice(self, msg, title="field session"):
        """a quiet notice (SOLO_NOTICE_CMD; unset: the log line only), through the sender's queue"""
        self.notices.append(msg); self.log("notice", message=msg, title=title, sent="queued")
        self.sender.notifier = self.notifier; self.sender.put(msg, title, notice=True)
    def marker(self, text, data):
        """a line in the events file: data "use" (the recordings from here on are test data) or "skip" (they are not)"""
        line = dict(to=None, state=text, data=data, source="auto_session")
        line = {"from": time.strftime("%H:%M:%S", time.localtime(self.clock())), **line}
        if self.events_path:
            with open(self.events_path, "a") as f: f.write(json.dumps(line) + "\n")
        self.log("marker", text=text, data=data)

    # ---- time and stop
    def out_of_time(self):
        return self.stop_req > 0 or self.clock() >= self.t_end
    def request_stop(self, *_):
        self.stop_req += 1
        if self.stop_req == 1: self.stop_why = self.stop_why or "Ctrl-C"; print("\nstopping after the current step (Ctrl-C again: now)", flush=True)
        elif self.runner: self.runner.abort()

    # ---- the world
    def snap(self):
        """the world's snapshot; with a home area every detection of the raw vision view is another robot except ours:
        the one of our colour + id within MATCH_D of the robot the tools drive (the page's), or, while the page's robot
        is not fresh, the only one of our colour + id. A robot of another colour or id seen in a single frame so far
        (vision_unconfirmed: RawVision's confirmation rule) is left out of others. Any other copy of our colour + id is another robot; when it
        cannot be told which one is ours, us_ambiguous says why (no information). raw_us: ours, (x, y, age) or None.
        twins: [(x, y)] of every robot seen as our colour + id when there are two or more (no information: the
        connector may follow either). A camera that sends without being listed is listed from then on (logged once).
        Our robot's last unambiguous position is kept (track_us): a position taken as ours that it cannot be known to
        have reached from there is "which robot is ours was lost" (us_lost, no information), which our robot seen
        again at that position or the identity probe clears. mine: [(x, y, age)] of every robot of our colour + id"""
        self.alert_poll()
        s = self.world.snapshot()
        if self.cameras is not None:
            for c in sorted(set(s.get("cam_age") or {}) | set(s.get("unlisted") or ())):
                if c not in self.cameras:
                    self.cameras = sorted(self.cameras + [c])
                    if c not in self.cams_logged:
                        self.cams_logged.add(c); self.log("camera_unlisted", camera=c, listed=self.cameras_given,
                                                          note="frames from a camera not on the list: watched from now on")
        if self.home is not None and "vision_dets" in s:
            me = (self.team, self.our_id); dets = list(s["vision_dets"] or [])
            mine = [d for d in dets if (d[0], int(d[1])) == me]
            r = s["robot"] if s.get("robot") is not None and s.get("robot_age", 9) <= VISION_FRESH else None
            us = None; amb = None; who = f"{self.team} {self.our_id}"
            if r is not None and mine:
                near = [d for d in mine if math.hypot(d[2] - r[0], d[3] - r[1]) <= MATCH_D]
                if len(near) == 1: us = near[0]
                elif near: amb = f"{len(near)} robots seen as {who} within {MATCH_D} m of the tools' robot"
                elif len(mine) == 1: us = mine[0]                  # the mismatch check tells
                else: amb = f"{len(mine)} robots seen as {who}, none within {MATCH_D} m of the tools' robot"
            elif len(mine) == 1: us = mine[0]
            elif mine: amb = f"{len(mine)} robots seen as {who} and the tools' robot not known"
            self.follow_us(s.get("own_track"))
            us, s["us_lost"] = self.track_us(mine, us, amb)
            s["raw_us"] = (us[2], us[3], us[4]) if us is not None else None; s["us_ambiguous"] = amb
            s["twins"] = [(round(d[2], 2), round(d[3], 2)) for d in sorted(mine, key=lambda d: (d[2], d[3]))] if len(mine) > 1 else []
            s["mine"] = [(d[2], d[3], d[4]) for d in mine]
            single = s.get("vision_unconfirmed") or []             # another robot counts once confirmed (RawVision)
            s["others"] = [(f"{c}{i}", x, y) for c, i, x, y, a in sorted(d for d in dets if d is not us and ((d[0], int(d[1])) == me or d not in single))]
            if self.found_due: L, msg = self.found_due; self.found_due = None; self.found_text(L, msg, s)
        return s
    ID_TIGHT = 0.15                     # m: how far our robot may seem to move while it stands (detection noise)
    def motion(self, on):
        """a motion child (or the strategy node) starts (on) or has stopped: the time it ran, plus settle s after
        each stop (reaction and braking), bounds our robot's moves"""
        now = self.clock()
        if on and not self.moving: self.moving = True; self.moves.append([now, None])
        elif not on and self.moving: self.moving = False; self.moves[-1][1] = now
    REAPPEAR_GAP = 0.5                  # s: our robot unseen this long (by detection times) is a reappearance, not continuity
    FOLLOW_GAP = 0.15                   # s: after two copies were seen, the one left continues ours only if the anchor followed it this recently
    def may_move(self, t0, t1):
        """our robot may have been moving at some time within [t0, t1]: a motion child or the strategy node ran then,
        or stopped less than settle s before (reaction and braking). It says whether, not how far"""
        return any(a <= t1 and (b is None or b + self.settle >= t0) for a, b in self.moves)
    def id_bound(self, now):
        """continuity: how far our robot can be from its last unambiguous position at time now (a detection's time):
        ID_TIGHT plus max_speed x the time since that position's detection when it may have been moving, else
        ID_TIGHT (it stood). The settle window after a stop only says that it may move, it adds no distance"""
        p = self.us_last
        return self.ID_TIGHT + (self.max_speed * max(0.0, now - p["t"]) if self.may_move(p["t"], now) else 0.0)
    def track_us(self, mine, us, amb):
        """keep our robot's last unambiguous position (the anchor: one robot of our colour + id, taken as ours); ->
        (us, why lost). The anchor is kept with its detection's time. A single robot of our colour + id seen after the
        anchor's detection is ours when
          continuity: ours was unseen for less than REAPPEAR_GAP s and it is within id_bound of the anchor (after two
            copies were seen: and the anchor followed one of them within the last FOLLOW_GAP s);
          reappearance: ours was unseen for REAPPEAR_GAP s or more, it is within `reappear` m of the anchor (ID_TIGHT
            when our robot cannot have moved since) and no second robot of our colour + id was seen since the anchor;
        otherwise which robot is ours is lost (no information): the position taken as ours is not one ours can be
        known to have reached (another robot with our colour + id, the connector following it), also after two copies
        were seen and one disappeared. While two or more are seen, the anchor follows the one copy within id_bound of
        it (when exactly one is). Lost holds until one robot of our colour + id has been seen within ID_TIGHT of the
        anchor for clear_s without a break, or the identity probe recognises ours (probe); time alone never clears it"""
        now = self.clock(); who = f"{self.team} {self.our_id}"
        if len(mine) > 1: self.twins_seen = True
        if self.ours_lost is None and us is not None and len(mine) == 1 and amb is None and self.us_last is not None:
            p = self.us_last; d = math.hypot(us[2] - p["x"], us[3] - p["y"]); t_det = now - us[4]; gap = t_det - p["t"]
            if gap < self.REAPPEAR_GAP:
                lim = self.id_bound(t_det); extra = ""
                bad = d > lim or (self.twins_seen and gap > self.FOLLOW_GAP)   # after two copies: only the one the anchor followed
            else:
                lim = self.reappear if self.may_move(p["t"], t_det) else self.ID_TIGHT
                bad = d > lim or self.twins_seen; extra = f", ours unseen for {gap:.1f} s"
            if bad:
                how = "after two robots were seen as ours, the one left is" if self.twins_seen else "the robot seen as ours is"
                self.ours_lost = dict(since=now, x=p["x"], y=p["y"], t_ok=None,
                                      why=f"{how} {who} at ({us[2]:+.2f}, {us[3]:+.2f}), {d:.2f} m from where ours last stood at "
                                          f"({p['x']:+.2f}, {p['y']:+.2f}) {now - p['t']:.0f} s ago (at most {lim:.2f} m{extra})")
                self.log("ours_lost", why=self.ours_lost["why"], gap=round(gap, 2))
        L = self.ours_lost
        if L is not None:
            back = len(mine) == 1 and math.hypot(mine[0][2] - L["x"], mine[0][3] - L["y"]) <= self.ID_TIGHT
            if not back: L["t_ok"] = None; return None, f"{self.LOST}: {L['why']}"
            L["t_ok"] = now if L["t_ok"] is None else L["t_ok"]
            if now - L["t_ok"] < self.clear_s: return None, f"{self.LOST}: {L['why']}"
            self.log("ours_found", at=[round(mine[0][2], 3), round(mine[0][3], 3)], lost_for=round(now - L["since"], 1))
            self.ours_lost = None; us = mine[0] if us is None or us is mine[0] else us
            self.found_due = (L, f"our robot seen again at its place ({mine[0][2]:+.2f}, {mine[0][3]:+.2f})")   # sent by snap
        if us is not None and len(mine) == 1 and amb is None: self.keep_us(us, now, alone=True)
        elif len(mine) > 1 and self.us_last is not None:              # two copies: the one ours can have become
            p = self.us_last
            near = [d for d in mine if math.hypot(d[2] - p["x"], d[3] - p["y"]) <= self.id_bound(now - d[4])]
            if len(near) == 1: self.keep_us(near[0], now, alone=False)
        return us, None
    def follow_us(self, track):
        """follow our robot on the camera frames between snapshots (RawVision's own_track: every sighting of our
        colour + id, oldest first), so that the time the supervisor took no snapshot (a child being stopped, a sleep)
        is not counted as our robot unseen: from the anchor forward, each sighting within ID_TIGHT plus max_speed x the
        time since the anchor's (when it may have been moving) moves the anchor to it; a sighting REAPPEAR_GAP s or
        more after the anchor's is a real gap and is left to track_us (reappearance), as is any sighting after two
        copies were seen that comes more than FOLLOW_GAP s after the anchor's. Sightings farther away (another copy)
        are skipped; while identity is lost the anchor stays"""
        p = self.us_last
        if not track or p is None or self.ours_lost is not None: return
        now = self.clock(); x, y, t = p["x"], p["y"], p["t"]
        for age, qx, qy in track:
            tq = now - age; dt = tq - t
            if dt <= 0: continue
            if dt >= self.REAPPEAR_GAP or (self.twins_seen and dt > self.FOLLOW_GAP): break
            lim = self.ID_TIGHT + (self.max_speed * dt if self.may_move(t, tq) else 0.0)
            if math.hypot(qx - x, qy - y) <= lim: x, y, t = qx, qy, tq
        if t > p["t"]: self.keep_us((None, None, x, y, now - t), now, alone=False)
    def keep_us(self, d, now, alone):
        """our robot's last unambiguous position from detection d (kept with the detection's time)"""
        t_det = now - d[4]
        if self.us_last is None or t_det >= self.us_last["t"]:
            self.us_last = dict(x=d[2], y=d[3], t=t_det)
            if alone: self.twins_seen = False
            self.moves = [m for m in self.moves if m[1] is None or m[1] + self.settle > t_det]
    def lost_tail(self, s=None, probe=False):
        """what a person can do when which robot is ours was lost; and that a probe may run when probes are left, the
        hold alerting is the one that runs it (probe) and the candidate in snapshot s (with none in view: the anchor)
        stands where a probe is accepted (our half or within fetch_limit beyond its open edges)"""
        L = self.ours_lost
        if not L: return "restart the session"
        mine = (s or {}).get("mine") or []; at = mine[0][:2] if len(mine) == 1 else (L["x"], L["y"])
        can = probe and self.probes and self.probes_run < self.probes and in_rect(at, fetch_area(self.home, self.fetch_limit))
        more = "; the session will try to recognise it by a short turn on the spot" if can else ""
        return f"put the robot back at ({L['x']:+.2f}, {L['y']:+.2f}) or restart the session{more}"
    def found_text(self, L, msg, s):
        """identity recovered after L (the lost record): msg + ": the session resumes", or "; the session still holds:
        <cause>" while another cause keeps the hold (no information, another robot in our half or near ours: snapshot
        s); an alert when the lost alert went out (a person may be on the way to the robot and must know it moves
        again), else a quiet notice"""
        why = self.no_info(s) or (self.robots_text(self.intrusion(s)) if self.intrusion(s) else None)
        msg += f"; the session still holds: {why}" if why else ": the session resumes"
        if L.get("alerted"): self.alert(msg, kind="ours_found", force=True)
        else: self.notice(msg)
    # ---- the identity probe: which robot is ours, proved by a commanded turn on the spot
    PROBE_STILL_D, PROBE_STILL_TH = 0.03, math.radians(5.0)   # the candidate stands: moved less, turned less
    PROBE_MOVE = 0.10                   # m: the candidate translating more stops the probe; a pass needs less overall
    PROBE_GAP = 0.5                     # s between the two turns
    PROBE_AFTER = 0.3                   # s after a turn's child ended before its heading is read
    PROBE_BAND = (0.4, 1.6)             # the heading change that passes, times rate x time
    PROBE_BAD_S, PROBE_BAD_D = 60.0, 0.3   # no new probe of a candidate within this of a failed one for this long
    PROBE_RETRY = 10.0                  # s: no new probe this long after one whose child did not complete its turn
    def candidate(self, s):
        """the one robot of our colour + id in view with its heading: (x, y, heading, age), or None"""
        mine = s.get("mine") or []
        if len(mine) != 1: return None
        x, y, _ = mine[0]; me = (self.team, self.our_id)
        hs = [q for q in s.get("vision_poses") or () if (q[0], int(q[1])) == me and math.hypot(q[2] - x, q[3] - y) <= MERGE_CAP]
        if not hs: return None
        q = min(hs, key=lambda q: q[5]); return (x, y, q[4], q[5])
    def probe_others(self, s, c):
        """the other robots (not of our colour + id: those are counted in "mine") inside our watch area or within
        intrude_dist of the candidate c"""
        wa = watch_area(self.home); me = f"{self.team}{self.our_id}"
        return [(n, x, y) for n, x, y in s.get("others") or ()
                if n != me and (in_rect((x, y), wa) or math.hypot(x - c[0], y - c[1]) < self.intrude_dist)]
    def probe_ready(self, s, now):
        """called from every poll of a hold while which robot is ours is lost: -> the candidate (x, y, heading) once the
        preconditions have held for clear_s without a break, else None. Preconditions: vision fresh; exactly one robot of
        our colour + id seen, with a heading, standing (moved less than PROBE_STILL_D, turned less than PROBE_STILL_TH);
        no second copy; no other robot inside our watch area or within intrude_dist of it; it stands inside our half or
        within fetch_limit beyond its open edges; no failed probe at that place within PROBE_BAD_S s; probes left; not
        within PROBE_RETRY s of a probe whose child did not complete its turn"""
        if not self.probes or self.probes_run >= self.probes or self.ours_lost is None: self.probe_cand = None; return None
        c = None if self.vision_gap(s) else self.candidate(s)
        ok = c is not None and not self.probe_others(s, c) and in_rect(c[:2], fetch_area(self.home, self.fetch_limit))
        ok = ok and not any(now - t < self.PROBE_BAD_S and math.hypot(c[0] - x, c[1] - y) < self.PROBE_BAD_D for x, y, t in self.probe_bad)
        pc = self.probe_cand
        if ok and pc is not None and (math.hypot(c[0] - pc["x"], c[1] - pc["y"]) >= self.PROBE_STILL_D
                                      or abs(wrap(c[2] - pc["th"])) >= self.PROBE_STILL_TH): ok = False
        if not ok:
            self.probe_cand = None if c is None else dict(x=c[0], y=c[1], th=c[2], since=now)
            return None
        if pc is None: self.probe_cand = pc = dict(x=c[0], y=c[1], th=c[2], since=now)
        return c[:3] if now - pc["since"] >= self.clear_s and now >= self.probe_wait else None
    def probe_watch(self, c0, acc):
        """the watch of a probe child: no information (vision, the alert channel, the candidate unseen or a second
        copy), another robot inside our watch area or near the candidate, the candidate translating more than
        PROBE_MOVE -> why to stop. acc: the candidate's unwrapped heading change, accumulated poll by poll"""
        def watch():
            s = self.snap(); g = self.vision_gap(s)
            if g: return "no information: " + g
            mine = s.get("mine") or []
            if len(mine) != 1: return f"no information: {len(mine)} robots seen as {self.team} {self.our_id}"
            c = self.candidate(s)
            if c is None: return "no information: the candidate's heading not seen"
            w = self.probe_others(s, c)
            if w: return "another robot: " + ", ".join(f"{n} at ({x:+.2f}, {y:+.2f})" for n, x, y in w)
            d = math.hypot(c[0] - c0[0], c[1] - c0[1])
            if d > self.PROBE_MOVE: acc["moved"] = d; return f"the candidate moved {d:.2f} m"
            acc["dth"] += wrap(c[2] - acc["th"]); acc["th"] = c[2]; acc["last"] = c
            return None
        return watch
    def probe_step(self, rate, box):
        """the probe child as a step: characterize.py spin <rate> <time> (open loop: only an angular rate, zeros before
        and after; it stops on its own when the pose it reads is stale or leaves SOLO_BOX)"""
        return dict(kind="test", test="spin", args=[f"{rate:+.2f}", f"{self.probe_time:.2f}"], env={"SOLO_BOX": box},
                    label="identity probe", timeout=15.0)
    def probe(self, c):
        """prove whether the candidate c (x, y, heading) is ours: two turns on the spot in a random order of signs,
        probe_rate rad/s for probe_time s each, PROBE_GAP s apart, each its own characterize.py spin child, watched
        (probe_watch). Safe while it is not known which robot is ours: a pure turn on the spot moves nobody across the
        field; if the candidate is a twin, our real robot only turns where it stands. Pass: during each turn the
        candidate's heading changed in the commanded direction by PROBE_BAND x rate x time (from before the turn to
        PROBE_AFTER s after its child ended) and it moved less than PROBE_MOVE overall -> identity anchored on it, an
        alert when the lost alert went out (else a quiet notice). Fail (no turn seen, the wrong direction, moved):
        identity stays lost (lost again, at the same anchor, when the tracking cleared it during the turns), one
        alert, no new probe at that place for PROBE_BAD_S s. Stopped by no information or
        another robot, or a child that did not complete its turn (a non-zero exit, a result line marked interrupted or
        aborted, none at all): no verdict, no alert, the hold goes on (after a child's abort, the next probe not before
        PROBE_RETRY s); it still counts against `probes`.
        -> True on a pass"""
        self.probes_run += 1; self.probe_cand = None; L0 = self.ours_lost
        signs = random.sample((1.0, -1.0), 2); x0, y0 = c[0], c[1]
        box = f"{x0 - 0.25:.2f},{x0 + 0.25:.2f},{y0 - 0.25:.2f},{y0 + 0.25:.2f}"
        self.log("probe_start", n=self.probes_run, at=[round(x0, 3), round(y0, 3)], signs=signs, rate=self.probe_rate, seconds=self.probe_time)
        turns = []; why = None; last = c; child = None
        for k, sg in enumerate(signs):
            if k: self.sleep(self.PROBE_GAP)
            q = self.candidate(self.snap())
            if q is None: why = "no information: the candidate not seen"; break
            acc = dict(dth=0.0, th=q[2], last=q, moved=None)
            r = self.run_step(self.probe_step(sg * self.probe_rate, box), watch=self.probe_watch((x0, y0), acc))
            if r.get("stopped") and acc["moved"] is None: why = r["stopped"]; break
            d = r.get("data") if isinstance(r.get("data"), dict) else {}
            if acc["moved"] is None and (r.get("rc") != 0 or not d or d.get("interrupted") or d.get("aborted")):
                why = child = f"the probe child did not complete its turn (rc {r.get('rc')}): {r.get('line') or 'no result line'}"; break
            if acc["moved"] is None:
                self.sleep(self.PROBE_AFTER); stop = self.probe_watch((x0, y0), acc)()
                if stop and acc["moved"] is None: why = stop; break
            last = acc["last"]; turns.append((sg, acc["dth"], acc["moved"]))
            if acc["moved"] is not None: break
        at = f"({x0:+.2f}, {y0:+.2f})"; want = self.probe_rate * self.probe_time
        if why is not None:
            self.probe_none += 1; self.probe_none_why = why
            if child:                                  # the child aborted: no verdict, the next probe not before PROBE_RETRY s
                self.probe_wait = self.clock() + self.PROBE_RETRY
                self.log("probe_verdict", n=self.probes_run, ok=None, at=[round(x0, 3), round(y0, 3)], no_verdict=why)
            else: self.log("probe_stopped", n=self.probes_run, why=why)
            self.probes_left_alert(); return False
        moved = math.hypot(last[0] - x0, last[1] - y0)
        ok = (len(turns) == 2 and moved < self.PROBE_MOVE and
              all(m is None and self.PROBE_BAND[0] * want <= sg * dth <= self.PROBE_BAND[1] * want for sg, dth, m in turns))
        self.log("probe_verdict", n=self.probes_run, ok=ok, at=[round(x0, 3), round(y0, 3)], moved=round(moved, 3),
                 turns=[dict(sign=sg, turned=round(dth, 3), moved=m) for sg, dth, m in turns], wanted=round(want, 3))
        if ok:
            q = self.snap(); cc = self.candidate(q) or last
            L = self.ours_lost or {}
            self.ours_lost = None; self.twins_seen = False; self.moves = [m for m in self.moves if m[1] is None]
            self.us_last = dict(x=cc[0], y=cc[1], t=self.clock() - (cc[3] if len(cc) > 3 else 0.0))
            self.log("ours_found", at=[round(cc[0], 3), round(cc[1], 3)], by="probe")
            self.found_text(L, f"our robot recognised by its turn at ({cc[0]:+.2f}, {cc[1]:+.2f})", self.snap())
            return True
        self.probe_bad.append((x0, y0, self.clock()))
        if self.ours_lost is None and L0 is not None:      # cleared by the tracking during the turns: lost again
            self.ours_lost = dict(L0, t_ok=None); self.log("ours_lost", why=L0["why"], after="a failed probe")
        msg = (f"the robot seen as {self.team} {self.our_id} at {at} did not follow a commanded turn: it is not ours, "
               f"or ours does not respond; the session holds")
        if self.probes_run >= self.probes: msg += f" (no identity probe left: {self.probes_run} used{self.no_verdict_text()})"; self.probes_out_sent = True
        self.alert(msg, kind="probe_fail", robot=at, force=True)
        return False
    def probes_left_alert(self):
        """once, when the last probe ended without a verdict"""
        if self.probes and self.probes_run >= self.probes and not self.probes_out_sent:
            self.probes_out_sent = True
            self.alert(f"no identity probe left ({self.probes_run} used{self.no_verdict_text()}): the session holds: {self.lost_tail()}", kind="probes_out", force=True)
    def no_verdict_text(self):
        return f", {self.probe_none} without a verdict, the last: {self.probe_none_why}" if self.probe_none else ""

    def our_pos(self, s):
        """our robot's position when it is fresh: ours in the raw view, else the page's; None = not known"""
        o = s.get("raw_us")
        if o is not None and o[2] <= self.robot_fresh: return (o[0], o[1])
        if "vision_dets" not in s and s.get("robot") is not None and s.get("robot_age", 9) <= self.robot_fresh: return tuple(s["robot"][:2])
        return None
    def no_info(self, s):
        """with a home area: why the supervisor cannot tell that the surroundings are clear, or None. Every listed
        camera must have sent a detection frame within VISION_FRESH s (a camera that never sent is no information);
        a fresh frame without another robot is clear"""
        if self.home is None: return None
        g = self.vision_gap(s)
        if g: return g
        if s.get("page_age", 0) > PAGE_FRESH: return f"the field page has not answered for {min(s['page_age'], 1e4):.1f} s"
        if s.get("us_lost"): return s["us_lost"]
        if s.get("twins"): return self.twin_text(s["twins"])
        if s.get("us_ambiguous"): return f"which robot is ours cannot be told: {s['us_ambiguous']}"
        if self.our_pos(s) is None: return self.unseen_text()
        return None
    def unseen_text(self):
        return f"our robot ({self.team} {self.our_id}) not seen in the vision"
    def vision_gap(self, s):
        """the alert channel dead, or a listed camera without a fresh detection frame -> why; else None"""
        d = self.dead_text()
        if d: return d
        if "vision_age" not in s: return "no raw vision view"
        if self.cameras is not None:
            ca = s.get("cam_age") or {}
            late = [c for c in self.cameras if ca.get(c, 1e9) > VISION_FRESH]
            if late:
                c = late[0]
                more = f", {len(late) - 1} more camera(s) late" if len(late) > 1 else ""
                if self.cameras_given is not None and c not in self.cameras_given: more = f", not on the camera list{more}"
                return f"no vision for {min(ca[c], 1e4):.1f} s (camera {c}{more})" if c in ca else f"no vision from camera {c} since the start{more}"
        if s["vision_age"] > VISION_FRESH: return f"no vision for {min(s['vision_age'], 1e4):.1f} s"
        return None
    def twin_text(self, twins):
        n = "two" if len(twins) == 2 else str(len(twins))
        at = ", ".join(f"({x:+.2f}, {y:+.2f})" for x, y in twins[:-1]) + f" and ({twins[-1][0]:+.2f}, {twins[-1][1]:+.2f})"
        return f"{n} robots are seen as {self.team} {self.our_id}: at {at}, the tools cannot tell which is ours"
    def mismatch(self, s):
        """the robot the tools drive (the page's, through the connector) is not ours (colour + id in the raw view).
        Ours kept older than MATCH_AGE (its camera no longer shows it) is not compared with the page's fresh position:
        our robot not seen, no information. Positions that agree with y flipped name --vision-mirror"""
        if self.home is None or "vision_dets" not in s: return None
        r = s.get("robot") if s.get("robot") is not None and s.get("robot_age", 9) <= VISION_FRESH else None
        if r is None or s.get("vision_age", 9) > VISION_FRESH: return None
        if s.get("twins") or s.get("us_ambiguous") or s.get("us_lost"): return None     # which one is ours cannot be told: no information
        o = s.get("raw_us")
        if o is not None and o[2] > MATCH_AGE: return None
        if o is not None:
            d = math.hypot(r[0] - o[0], r[1] - o[1])
            if d > MATCH_D:
                flip = " (the positions agree with y flipped: check --vision-mirror)" if math.hypot(r[0] - o[0], r[1] + o[1]) <= MATCH_D else ""
                return f"the tools' robot at ({r[0]:+.2f}, {r[1]:+.2f}) is {d:.2f} m from {self.team} {self.our_id} at ({o[0]:+.2f}, {o[1]:+.2f}){flip}"
            return None
        for name, x, y in s.get("others") or ():
            if math.hypot(r[0] - x, r[1] - y) < 0.1:
                return f"the tools' robot at ({r[0]:+.2f}, {r[1]:+.2f}) is {self.who(name)}; {self.team} {self.our_id} is not seen"
        return None
    def on_mismatch(self, why):
        """the connector's index points at another robot: all motion stops (first), one alert (queued), the session ends"""
        self.motion_halted = True
        if self.not_ours is None:
            if self.runner: self.runner.abort()
            self.not_ours = why; self.stop_why = "the robot the tools are driving is not ours"; self.stop_req = max(self.stop_req, 1)
            self.log("not_ours", why=why)
            self.alert(f"the robot the tools are driving is not ours: {why}; all motion stopped, the session ends", kind="not_ours", urgent=True)
    def motion_check(self, s):
        """the rule of every watched motion (a fetch included): -> why to stop now, or None"""
        mm = self.mismatch(s)
        if mm: self.on_mismatch(mm); return "not ours: " + mm
        ni = self.no_info(s)
        if ni: return "no information: " + ni
        w = self.intrusion(s)
        if w: self.last_intrusion = (w, self.our_pos(s)); return "another robot: " + ", ".join(f"{n} at ({x:+.2f}, {y:+.2f})" for n, x, y, _ in w)
        return None
    def preflight(self):
        """-> why the session may not start, or None. A home area needs our vision id, the list of cameras, an alert
        command and the raw vision view (every listed camera sending); the tools' robot and ours must be the same"""
        if self.home is None:
            print("WARNING: no home area: other robots are not watched", flush=True); return None
        if self.our_id is None: return "a home area needs our robot's vision id (--our-id N or SOLO_OUR_ID)"
        if not self.cameras: return "a home area needs the field's cameras (--cameras N or --camera-ids LIST, or SOLO_CAMERAS / SOLO_CAMERA_IDS)"
        if getattr(self.notifier, "cmd", None) == "": return "a home area needs an alert command (SOLO_NOTIFY_CMD): a person must be alerted"
        if self.vision_error: return f"other robots cannot be watched: {self.vision_error}"
        t0 = self.clock(); s = self.snap()
        while (self.no_info(s) or s.get("robot") is None or s.get("robot_age", 9) > VISION_FRESH) and self.clock() - t0 < self.robot_wait:
            self.sleep(0.5); s = self.snap()
        if s.get("vision_age", 9) > VISION_FRESH or (self.no_info(s) or "").startswith("no vision"):
            return f"{self.no_info(s) or 'no vision packets'}: other robots cannot be watched"
        extra = sorted(set(self.cameras) - set(self.cameras_given or ()))
        if extra:
            return (f"frames from camera(s) {', '.join(map(str, extra))} that are not on the camera list "
                    f"({', '.join(map(str, self.cameras_given))}): list every camera of the field (--camera-ids)")
        mm = self.mismatch(s)
        if mm: return f"the robot the tools are driving is not ours: {mm}"
        hd = self.heading_check(s)
        if hd: return hd
        self.send_now(self.start_text())
        if not str(self.last_send).startswith(("rc 0", "ok")):
            return f"the alert command did not deliver the start alert ({self.last_send}): a person must be alerted (SOLO_NOTIFY_CMD)"
        return None
    HEADING_TOL = 0.5                   # rad: our robot's heading in the raw view and the page's may differ this much at start
    def heading_check(self, s):
        """the start check of the vision's mirror setting: with the positions agreeing, our robot's heading in the raw
        view and the page's must agree within HEADING_TOL (a wrong --vision-mirror flips the raw view's y and heading:
        the positions then agree only near y = 0). -> why the session may not start, or None (the check skipped when a
        heading is missing: logged)"""
        r = s.get("robot"); c = self.candidate(s)
        if r is None or len(r) < 3 or r[2] is None or s.get("robot_age", 9) > VISION_FRESH:
            self.log("heading_check", skipped="the field page gives no fresh heading"); return None
        if c is None or c[2] is None: self.log("heading_check", skipped="our robot's heading not seen in the raw view"); return None
        d = abs(wrap(c[2] - r[2]))
        if d <= self.HEADING_TOL: return None
        flip = abs(wrap(c[2] + r[2])) <= self.HEADING_TOL
        return (f"our robot's heading in the raw vision ({c[2]:+.2f} rad) is {d:.2f} rad from the page's ({r[2]:+.2f} rad)"
                f"{', the same with its sign flipped' if flip else ''}: check --vision-mirror (it must match the connector's mirror setting)")
    def start_text(self):
        """the alert a home session sends in its start check; the session starts only when the alert command exits 0"""
        return (f"session starting: {self.head.get('name') or 'unnamed'}, the {half_name(self.home)} half"
                f"{' (or the other, chosen at start)' if self.home_auto else ''}, budget {round((self.t_end - self.t_start) / 60)} min")
    def look(self, why=""):
        s = self.snap()
        b = s["ball"] if s["ball_age"] <= self.ball_lost else None
        s["ball_class"], s["ball_where"] = ball_class(b)
        if s["robot"] is not None:
            c, name = obstacles(*s["robot"][:2])[0]; s["robot_clear"], s["robot_near"] = round(c, 3), name
        return s
    def wait_ball_rest(self, s, T=6.0):
        """a ball seen but rolling, or briefly hidden: up to T s for it to rest or to count as lost"""
        t0 = self.clock()
        while self.clock() - t0 < T and not self.out_of_time():
            if s["ball_class"] == "lost" or (s["ball_still"] and s["ball_age"] < 1.0): return s
            self.sleep(0.25); s = self.look()
        return s


    # ---- the home area: load-time checks
    def step_boxes(self, st):
        """-> (list of rectangles the step's robot centre may use, None) or (None, why it cannot be told)"""
        k = st["kind"]; env = self.env_for(st); box = rects(env.get("SOLO_BOX", ""))
        if k == "place": return box or [PLACER_AREA], None
        if k == "test":
            if st.get("test") == "goto" and len(st.get("args", [])) >= 2: return [[st["args"][0], st["args"][0], st["args"][1], st["args"][1]]], None
            return box or [CHAR_AREA], None
        if k == "goto": return [[st["x"], st["x"], st["y"], st["y"]]], None
        if k == "series":
            path = expand(st["plan"], self.vs); out = []
            try: lines = [json.loads(l) for l in open(path) if l.strip() and not l.lstrip().startswith("#")]
            except (OSError, ValueError) as e: return None, f"plan {path} cannot be read ({e.__class__.__name__})"
            first = int(st.get("from") or 1); last = first - 1 + int(st["max"]) if st.get("max") else len(lines)
            for e in lines[first - 1:last]: out += rects(e.get("box", "")) or box or [CHAR_AREA]
            return out, None
        return None, None
    def prepare(self, steps=None):
        """mark the steps the session refuses (st["_refused"]) and keep every motion step's area (st["_boxes"])"""
        for st in (self.steps if steps is None else steps):
            why = None
            if self.no_ball and (st["kind"] == "place" or st.get("needs_ball") or st.get("park")
                                 or (st["kind"] == "strategy" and st.get("on", True)) or (st["kind"] == "test" and st.get("test") == "ram")
                                 or (st["kind"] == "series" and self.plan_has_ram(st))):
                why = "a step that could touch a ball (placement, park, ram, strategy on, needs_ball) in a session without the ball"
            if st["kind"] == "series" and "--resume" in [str(a) for a in st.get("args", [])] and not why:
                why = "--resume in a series step (the plan lines it would run cannot be checked)"
            if self.home is not None and not why and self.short_step_test(st):
                why = "a step test without its explicit start (fwd|side v x0 y0 dir length): its built-in start is not mirrored"
            if st["kind"] == "cmd" and not why:
                if not known_cmd(expand(st.get("argv", []), self.vs)):
                    if self.home is not None: why = "a command that is not on the list of known non-motion commands (KNOWN_CMDS)"
                    else: st["_unchecked"] = "UNCHECKED: a command the supervisor does not know; it must not move the robot"
            if self.home is not None and st.get("park") and not why and not in_rect(st["park"], robot_area(self.home)):
                why = f"its parking spot {st['park']} is not inside the home area"
            if st["kind"] == "strategy" and not why:
                names = set(st.get("params") or {}) | ({"side"} if st.get("side") else set())
                unknown = sorted(n for n in names if n not in solo_procs.NODE_PARAMS)
                if unknown: why = f"node parameters the node does not declare: {', '.join(unknown)}"
                elif st.get("side") not in (None, "negative", "positive", "auto"): why = f"side {st.get('side')!r}: negative, positive or auto"
            if st["kind"] in MOTION and not why:
                bx, err = self.step_boxes(st); st["_boxes"] = bx
                if self.home is not None:
                    if st["kind"] == "strategy" and st.get("on", True) and not st.get("side"):
                        why = "a strategy step without its side (\"side\": \"negative\" or \"positive\") in a home area"
                    elif err: why = err
                    elif bx:
                        bad = [r for r in bx if not rect_in(r, robot_area(self.home))]
                        if bad: why = f"its area {','.join(f'{v:g}' for v in bad[0])} is not inside the home area {','.join(f'{v:g}' for v in self.home)}"
            if why: st["_refused"] = why
        if self.home is not None and self.park is not None and not in_rect(self.park, robot_area(self.home)):
            for st in (self.steps if steps is None else steps): st.setdefault("_refused", f"the session's parking spot {self.park} is not inside the home area")
    def short_step_test(self, st):
        """a characterize.py step test with fewer than 6 arguments (the step itself, or a line of its series plan)"""
        short = lambda e: e.get("test") == "step" and len(e.get("args", [])) < 6
        if st["kind"] == "test": return short(st)
        if st["kind"] != "series": return False
        try: lines = [l for l in open(expand(st["plan"], self.vs)) if l.strip() and not l.lstrip().startswith("#")]
        except OSError: return False                            # step_boxes refuses an unreadable plan
        for l in lines:
            try: e = json.loads(l)
            except ValueError: e = dict(test=l.split()[0], args=l.split()[1:]) if l.split() else {}
            if short(e): return True
        return False
    def plan_has_ram(self, st):
        """a series plan with a ram line (JSON "test": "ram" or the argument form "ram ..."); unreadable = True"""
        def ram(l):
            try: return json.loads(l).get("test") == "ram"
            except ValueError: return l.split()[:1] == ["ram"]
        try: return any(ram(l) for l in open(expand(st["plan"], self.vs)) if l.strip() and not l.lstrip().startswith("#"))
        except OSError: return True

    # ---- the home area: other robots
    def intrusion(self, s):
        """[(name, x, y, 'inside' | distance)] of the other robots inside the home area (its watched area: the edges
        along the field's boundary moved out past the walls, watch_area) or within intrude_dist of ours (ours: its
        fresh position only; the caller checks no_info first)"""
        if self.home is None: return []
        rb = self.our_pos(s); out = []; wa = watch_area(self.home)
        for name, x, y in s.get("others") or ():
            if in_rect((x, y), wa): out.append((name, round(x, 2), round(y, 2), "inside"))
            elif rb is not None and math.hypot(x - rb[0], y - rb[1]) < self.intrude_dist:
                out.append((name, round(x, 2), round(y, 2), round(math.hypot(x - rb[0], y - rb[1]), 2)))
        return out
    def ball_near_in_box(self, s, st):
        """no-ball sessions: a ball inside the step's area and within ball_guard of our robot"""
        b = s["ball"] if s.get("ball") is not None and s.get("ball_age", 9) < 1.0 else None
        boxes = rects(st["keep_ball_out"]) if st.get("keep_ball_out") else (st.get("_boxes") or [])
        if b is None or s.get("robot") is None or not any(in_rect(b, r, 0.15) for r in boxes): return False
        return math.hypot(b[0] - s["robot"][0], b[1] - s["robot"][1]) < self.ball_guard
    def watch_for(self, st, extra=None):
        """the check run 10 times a second while a motion step runs (run_step uses it for every motion step)"""
        if self.home is None and not self.no_ball and extra is None: return None
        def watch():
            s = self.snap()
            if self.home is not None:
                why = self.motion_check(s)
                if why: return why
            if self.no_ball and self.ball_near_in_box(s, st): return f"a ball in the test area at ({s['ball'][0]:+.2f}, {s['ball'][1]:+.2f})"
            return extra(s) if extra else None
        return watch
    @staticmethod
    def who(name):
        m = re.match(r"([a-z]+)(\d+)$", name); return f"{m.group(1)} {m.group(2)}" if m else name
    def outside_note(self, s=None, p=None):
        """", our robot at (x, y), outside our half" while our robot (its fresh position, else its last known) is outside
        the home area; else "" """
        if self.home is None: return ""
        p = p or (self.our_pos(s) if s is not None else None)
        if p is None and self.us_last is not None: p = (self.us_last["x"], self.us_last["y"])
        if p is None or in_rect(p, robot_area(self.home), 0.05): return ""
        return f", our robot at ({p[0]:+.2f}, {p[1]:+.2f}), outside our half"
    def robots_text(self, ww):
        """the robots of an intrusion in one phrase: "another robot (yellow 2) entered our half at (x, y)", or
        "robots yellow 2 at (x, y), yellow 5 at (x, y) entered our half; blue 0 at (x, y) came within 0.80 m of our robot" """
        if len(ww) == 1:
            name, x, y, how = ww[0]
            what = "entered our half" if how == "inside" else f"came within {how:.2f} m of our robot"
            return f"another robot ({self.who(name)}) {what} at ({x:+.2f}, {y:+.2f})"
        at = lambda q: f"{self.who(q[0])} at ({q[1]:+.2f}, {q[2]:+.2f})"
        ins = [q for q in ww if q[3] == "inside"]; near = [q for q in ww if q[3] != "inside"]; parts = []
        if ins: parts.append(("robots " if len(ins) > 1 else "another robot, ") + ", ".join(map(at, ins)) + (" entered" if len(ins) > 1 else ", entered") + " our half")
        if near: parts.append(("robots " if len(near) > 1 else "another robot, ") + ", ".join(map(at, near)) + f" came within {max(q[3] for q in near):.2f} m of our robot")
        return "; ".join(parts)
    def intrusion_hold(self, w, robot, during, inside_only=False, cause=None):
        """hold all motion until the surroundings have been known and clear (no other robot inside the home area or
        near ours; inside_only: inside the home area) for clear_s. Alerts: the robots inside or near when first seen
        in this hold, at once, one alert per poll naming every new one; two robots seen as ours, and which robot is
        ours lost, at once; no information without a break for info_alert s ("no vision for N s: the session holds",
        once per cause); a hold that has lasted info_alert s with nothing sent, whatever its causes (information that
        comes and goes; the causes seen are named); then one reminder every hold_limit s (at most max_alerts in all).
        While our robot is outside the home area every alert says where it stands. Repeats of one cause back off and
        alerts are spaced (alert); an alert held by the backoff is repeated while its cause lasts. No alert for a dead
        alert channel (it is itself the cause; the sender's notice reports it).
        While which robot is ours is lost, the identity probe runs when its preconditions hold (probe_ready, probe).
        A hold whose only cause was our robot not seen (cause: why the step was stopped; the vision fresh, no other
        robot inside or within intrude_dist of our watch area, at any poll of it) ends once our robot has been seen
        again for clear_own s; any other cause needs clear_s.
        -> True when clear, False when the session switched halves, ends or the robot is not ours"""
        t0 = self.clock()
        inc = dict(since=round(t0, 3), robots=w, robot=robot and [round(v, 3) for v in robot[:2]], during=during)
        self.intrusions.append(inc); self.log("intrusion", **inc)
        how_now = "stopped" if during else "waits"
        h = dict(n=0, t_alert=None, named=set(), first={}, twins_sent=False, twins_t=None, lost_sent=False, causes={}, ni_sent=set())
        def where(p): return f"({p[0]:+.2f}, {p[1]:+.2f})" if p else "(not seen)"
        def sent(ok, now):
            if ok:
                h["t_alert"] = now; h["n"] = max(h["n"], 1)
            return ok
        def entered(ww, rb, now):
            sent(self.alert_new_robots(ww, h, now, f"; our robot {how_now} at {where(rb)}{', outside our half' if self.outside_note(p=rb) else ''}"), now)
        self.holding = dict(since=t0, cause=self.robots_text(w) if w else "no information")
        own = not w and (cause is None or cause == "no information: " + self.unseen_text())
        try: return self._intrusion_hold(w, robot, t0, inc, h, sent, entered, inside_only, own)
        finally:
            if not self.out_of_time(): self.holding = None
    def alert_new_robots(self, ww, h, now, tail):
        """the robots of ww not yet named in this hold (h: named, first) in one alert, each robot its own backoff key
        (a robot first seen ALERT_LASTING s ago goes out whatever its backoff); -> True when an alert was accepted"""
        new = [q for q in ww if q[0] not in h["named"]]
        for q in new: h["first"].setdefault(q[0], now)
        lasting = lambda q: now - h["first"][q[0]] >= self.ALERT_LASTING
        due = [q for q in new if self.alert_due("intruder", q[0], h["first"][q[0]]) or lasting(q)]
        if not due: return False
        if self.alert(f"{self.robots_text(due)}{tail}", kind="intruder", robot=[q[0] for q in due],
                      since=min(h["first"][q[0]] for q in due), force=True):
            h["named"].update(q[0] for q in due); return True
        return False
    def _intrusion_hold(self, w, robot, t0, inc, h, sent, entered, inside_only, own_only):
        entered(w, robot, t0)
        t_seen = t0; t_ni = None; self.motion_halted = True
        sw = dict(t_in=None, last=-1e9, other_bad=t0, corr_bad=t0, limit_sent=False)
        while not self.out_of_time():
            s = self.look(); now = self.clock()
            mm = self.mismatch(s)
            if mm: self.on_mismatch(mm); return False
            ni = self.no_info(s); ww = [] if ni else self.intrusion(s)
            if s.get("us_lost") and self.ours_lost is not None:
                c = self.probe_ready(s, now)
                if c is not None:
                    self.probe(c)
                    s = self.look(); now = self.clock(); ni = self.no_info(s); ww = [] if ni else self.intrusion(s)
            if own_only and ((ni and ni != self.unseen_text()) or ww or self.near_half(s)): own_only = False
            if inside_only: ww = [q for q in ww if q[3] == "inside"]
            if self.switch_rule(s, ni, ww, now, sw): self.motion_halted = False; return False
            note = self.outside_note(s); lasting = lambda since: since is not None and now - since >= self.ALERT_LASTING
            if ni:
                t_seen = now; t_ni = now if t_ni is None else t_ni
                cause = re.sub(r"[-+]?\d+(?:\.\d+)?", "#", ni); h["causes"][cause] = ni; self.holding["cause"] = ni
                if self.dead is not None and ni == self.dead_text():       # the alert channel itself: nothing to send
                    h["ni_sent"].add(cause); sent(True, now)
                elif s.get("us_lost") and not h["lost_sent"]:               # which robot is ours was lost: at once
                    since = self.ours_lost and self.ours_lost["since"]
                    if sent(self.alert(f"{ni}{note}; the session holds: {self.lost_tail(s, probe=True)}", kind="ours_lost", since=since, force=lasting(since)), now):
                        h["lost_sent"] = True; h["ni_sent"].add(cause)
                elif s.get("twins") and not h["twins_sent"]:                # two robots as ours: at once
                    h["twins_t"] = h["twins_t"] or now
                    if sent(self.alert(f"{ni}{note}; the session holds", kind="twins", since=h["twins_t"], force=lasting(h["twins_t"])), now):
                        h["twins_sent"] = True; h["ni_sent"].add(cause)
            else:
                t_ni = None
                if ww:
                    t_seen = now; entered(ww, self.our_pos(s), now); self.holding["cause"] = self.robots_text(ww)
                    for name, x, y, how in ww: h["causes"]["robot " + name] = f"another robot ({self.who(name)}) at ({x:+.2f}, {y:+.2f})"
                elif now - t_seen >= (self.clear_own if own_only else self.clear_s):
                    self.motion_halted = False; self.log("intrusion_clear", held=round(now - inc["since"], 1), own_only=own_only); return True
            specific = False
            if ni and cause not in h["ni_sent"] and now - t_ni >= self.info_alert:   # no information without a break
                specific = sent(self.alert(f"{ni}{note}: the session holds", kind="no_info", robot=cause, since=t_ni, force=lasting(t_ni)), now)
                if specific: h["ni_sent"].add(cause)
            if specific: pass
            elif h["t_alert"] is None and now - t0 >= self.info_alert:      # nothing sent yet: the hold as a whole
                msg = (f"the session has held for {now - t0:.0f} s without {self.clear_s:.0f} s known and clear: "
                       f"{'; '.join(h['causes'].values()) or 'no cause recorded'}{note}; the session holds")
                sent(self.alert(msg, kind="hold", since=t0, force=lasting(t0)), now)
            elif h["t_alert"] is not None and now - h["t_alert"] >= self.hold_limit and h["n"] < self.max_alerts:
                n = h["n"] + 1
                if ww: msg = (f"another robot ({self.who(ww[0][0])}) still in our half at ({ww[0][1]:+.2f}, {ww[0][2]:+.2f}) after "
                              f"{now - t0:.0f} s{note}: the session keeps holding (reminder {n}/{self.max_alerts})")
                else: msg = (f"the session still holds after {now - t0:.0f} s: {ni or '; '.join(h['causes'].values()) or 'no cause recorded'}"
                             f"{note} (reminder {n}/{self.max_alerts})")
                if sent(self.alert(msg, kind="hold_reminder", since=h["t_alert"], force=True), now): h["n"] = n
            self.sleep(0.2)
        self.motion_halted = False
        return False
    def near_half(self, s):
        """another robot inside our watch area or within intrude_dist of it (whatever is known of our robot)"""
        wa = watch_area(self.home)
        return [n for n, x, y in s.get("others") or () if rect_dist((x, y), wa) < self.intrude_dist]
    def switch_rule(self, s, ni, ww, now, sw):
        """while holding: our half held by another robot for switch_after s without a break (gaps under 1 s), the other
        half known and empty of every robot but ours for that whole time and the way there clear for clear_s -> switch
        halves (-> True). Not within switch_cooldown s of the last switch; after max_switches: alert once, hold"""
        if self.switch_after is None or self.home is None: return False
        inside = [q for q in ww if q[3] == "inside"]
        if inside:
            if sw["t_in"] is None or now - sw["last"] > 1.0: sw["t_in"] = now
            sw["last"] = now; sw["who"] = inside[0][0]
        elif ni or now - sw["last"] > 1.0: sw["t_in"] = None
        other = mirror_rect(self.home); p = self.our_pos(s)
        if ni or intruders(s.get("others"), other): sw["other_bad"] = now
        if ni or p is None or blockers(s.get("others"), [(p, into(other, p))], self.corridor): sw["corr_bad"] = now
        if sw["t_in"] is None or now - sw["t_in"] < self.switch_after or now - sw["other_bad"] < self.switch_after: return False
        if now - sw["corr_bad"] < self.clear_s: return False                 # the way over is not clear: hold, re-evaluate
        if len(self.switches) >= self.max_switches:
            if not sw["limit_sent"]:
                sw["limit_sent"] = True
                self.alert(f"our robot changed halves {len(self.switches)} times: no more switches; the session holds in the {half_name(self.home)} half",
                           kind="switch_limit", since=now)
            return False
        if self.switches and now - self.switches[-1] < self.switch_cooldown: return False
        self.switch_half(f"another robot ({self.who(sw['who'])}) stayed in the {half_name(self.home)} half for {now - sw['t_in']:.0f} s")
        return True
    def plan_dir(self):
        return os.path.join(self.vs.get("OUT") or os.path.join(data_dir(), "sessions"), "mirrored")
    def mirror_all(self):
        """the home area, the parking spot and every step still to run mirrored into the other half"""
        self.home = mirror_rect(self.home); pd = self.plan_dir(); self.half_sign = -self.half_sign
        self.head = mirror_head(self.head)              # the header's env (SOLO_BOX ...), home and parking spot
        if self.park is not None: self.park = tuple(mirror_pose(self.park))
        if self.pending is not None: self.pending[:] = [mirror_step(st, pd, self.vs) for st in self.pending]
        if self.current is not None: self.current = mirror_step(self.current, pd, self.vs)
        self.prepare((self.pending or []) + ([self.current] if self.current is not None else []))
        self.fetch_inc = None; self.ball_incident = None; self.bad_ball = None
    def switch_half(self, why):
        old = half_name(self.home); self.mirror_all(); self.switches.append(self.clock())
        msg = f"moved to the {half_name(self.home)} half: {why}"
        self.log("half_switch", n=len(self.switches), home=self.home, old=old, why=why); self.notice(msg)
    def choose_half(self):
        """--home auto: the half our robot stands in when it has been known and empty of other robots (its watched
        area) for clear_s; else the other half when that one has; else hold until one of the two is true. No half
        chosen within clear_s + info_alert s: one alert naming the reason (no information, our robot not seen, both
        halves held), then a reminder every hold_limit s (at most max_alerts in all)"""
        t0 = self.clock(); bad = {"-x": t0, "+x": t0}; n = 0; t_alert = None
        while not self.out_of_time():
            s = self.look(); p = self.our_pos(s); now = self.clock(); ni = self.no_info(s)
            if ni or p is None:
                bad = {"-x": now, "+x": now}
                why = ni or f"our robot ({self.team} {self.our_id}) not seen in the vision"
            else:
                mine = self.home if (p[0] >= 0) == (self.home[0] + self.home[1] >= 0) else mirror_rect(self.home); other = mirror_rect(mine)
                occ = {half_name(r): [n for n, x, y in intruders(s.get("others"), r)] for r in (mine, other)}
                for h, names in occ.items():
                    if names: bad[h] = now
                pick = next((r for r in (mine, other) if now - bad[half_name(r)] >= self.clear_s), None)
                if pick is not None:
                    if pick != self.home: self.mirror_all()
                    self.log("half_chosen", home=self.home, half=half_name(self.home), robot=[round(v, 3) for v in p]); return True
                if all(occ.values()): why = f"both halves hold other robots ({', '.join(self.who(n) for n in occ[half_name(mine)] + occ[half_name(other)])})"
                else: why = f"no half known and empty of other robots for {self.clear_s:.0f} s yet"
            if n == 0 and now - t0 >= self.clear_s + self.info_alert:
                if self.alert(f"{why}: no half chosen after {now - t0:.0f} s, the session waits", kind="choose_half", since=t0): n = 1; t_alert = now
            elif 0 < n < self.max_alerts and now - t_alert >= self.hold_limit:
                if self.alert(f"{why}: still no half chosen after {now - t0:.0f} s, the session waits (reminder {n + 1}/{self.max_alerts})",
                              kind="choose_half_reminder", since=t0): n += 1; t_alert = now
            self.sleep(0.2)
        return False
    def ensure_home_clear(self, inside_only=False):
        """-> True when no other robot is inside the home area (or near ours, unless inside_only) and the information
        is complete; else the hold (intrusion_hold) decides. A robot seen in a single frame so far is looked at once
        more after RawVision.CONFIRM_S before a step starts"""
        s = self.look()
        me = (self.team, self.our_id)
        if any((d[0], int(d[1])) != me for d in s.get("vision_unconfirmed") or ()): self.sleep(RawVision.CONFIRM_S); s = self.look()
        ni = self.no_info(s); w = [] if ni else self.intrusion(s)
        if inside_only: w = [q for q in w if q[3] == "inside"]
        return True if not w and not ni else self.intrusion_hold(w, self.our_pos(s), during=False, inside_only=inside_only)
    def wait_clear(self, ok, limit):
        """-> True once ok(snapshot) has held for clear_s without a break, False after limit s"""
        t0 = t_bad = self.clock()
        while not self.out_of_time():
            q = self.look()
            if self.mismatch(q): self.on_mismatch(self.mismatch(q)); return False
            if not ok(q): t_bad = self.clock()
            elif self.clock() - t_bad >= self.clear_s: return True
            if self.clock() - t0 >= limit: return False
            self.sleep(0.5)
        return False

    # ---- the home area: the ball and our robot outside it
    def in_home(self, p):
        """for the ball: inside the home area or between it and the walls on our side"""
        return self.home is None or in_rect(p, ball_area(self.home))
    def fetch(self, s, wait=None):
        """bring a ball resting outside the home area (at most fetch_limit beyond it) back to the parking spot, only
        while the home area holds no other robot, none is near ours and the corridor (robot -> ball -> parking spot)
        is clear, all of it known and clear for clear_s (from the start of the incident: a first attempt waits too);
        the fetch runs with the full motion check plus the corridor; -> True when the ball is back inside"""
        wait = self.fetch_wait if wait is None else wait
        if self.fetch_inc is None: self.fetch_inc = dict(since=self.clock(), tries=0, alerts=0, t_block=self.clock())
        inc = self.fetch_inc; t0 = self.clock()
        if not (self.robot_seen(s) and s["ball"] is not None): return False
        b = s["ball"][:2]; beyond = rect_dist(b, ball_area(self.home))
        if beyond > self.fetch_limit - FETCH_ROOM:              # the robot must fit behind the ball within the limit
            if inc["alerts"] == 0 and self.alert(f"ball at ({b[0]:+.2f}, {b[1]:+.2f}) is {beyond:.1f} m beyond our half (fetch limit {self.fetch_limit:.1f} m): fetch it when you can",
                                                 kind="fetch_far", since=inc["since"]):
                inc["alerts"] = 1; inc["t_alert"] = self.clock(); self.log("fetch_too_far", ball=[round(v, 3) for v in b], beyond=round(beyond, 2))
            return False
        def blocked(q):
            if self.no_info(q) or q.get("robot") is None or q.get("ball") is None: return [("no information", 0.0, 0.0, 0.0)]
            segs = [(q["robot"][:2], q["ball"][:2]), (q["ball"][:2], self.park)]; bb = blockers(q.get("others"), segs, self.corridor)
            return bb + [w for w in self.intrusion(q) if w[0] not in [v[0] for v in bb]]
        while True:
            bl = blocked(s); now = self.clock()
            if bl: inc["t_block"] = now
            elif now - inc.get("t_block", -1e9) >= self.clear_s: break
            if now - t0 >= wait or self.out_of_time():
                if bl and inc["alerts"] == 0 and now - inc["since"] >= self.fetch_wait:
                    near = ", ".join(f"{self.who(n)} at ({x:+.2f}, {y:+.2f})" for n, x, y, _ in bl[:3])
                    if self.alert(f"ball at ({b[0]:+.2f}, {b[1]:+.2f}) in the other half, other robots near it ({near}): fetch it when you can",
                                  kind="fetch_blocked", since=inc["since"]):
                        inc["alerts"] = 1; inc["t_alert"] = now; self.log("fetch_blocked", ball=[round(v, 3) for v in b], robots=bl)
                return False
            self.sleep(1.0); s = self.look()
            if not (self.robot_seen(s) and s["ball"] is not None): return False
        b, r = s["ball"][:2], s["robot"][:2]
        if inc["tries"] >= self.fetch_tries:
            if inc["alerts"] == 0 and self.alert(f"ball at ({b[0]:+.2f}, {b[1]:+.2f}) in the other half: {inc['tries']} fetches were stopped: fetch it when you can",
                                                 kind="fetch_stopped", since=inc["since"]):
                inc["alerts"] = 1; inc["t_alert"] = self.clock()
            return False
        inc["tries"] += 1; park = self.park
        box = clip_rects(path_box([r, b, park]), fetch_area(self.home, self.fetch_limit))
        st = dict(kind="place", x=park[0], y=park[1], label="fetch the ball", n=0, env={"SOLO_BOX": box})
        self.log("fetch", ball=[round(v, 3) for v in b], robot=[round(v, 3) for v in r], park=list(park), area=box, n_try=inc["tries"])
        def corridor(q):
            if q.get("robot") is None or q.get("ball") is None: return None
            bb = blockers(q.get("others"), [(q["robot"][:2], q["ball"][:2]), (q["ball"][:2], park)], self.corridor)
            return ("a robot in the corridor: " + ", ".join(f"{n} at ({x:+.2f}, {y:+.2f})" for n, x, y, _ in bb)) if bb else None
        if not self.motion_ok(st) or not self.ensure_home_clear(): inc["tries"] -= 1; return False
        if self.park != park: inc["tries"] -= 1; return False          # the halves were switched meanwhile
        res = self.run_step(st, watch=self.watch_for(st, extra=corridor))
        q = self.wait_ball_rest(self.look())
        if res.get("stopped"):
            inc["t_block"] = self.clock()                       # a retry waits clear_s from here
            self.log("fetch_stopped", why=res["stopped"])
            if not res["stopped"].startswith("not ours"): self.retreat(q)
            return False
        self.after(st, res, home=False); self.after_home(self.look(), fetched=True)
        q = self.look()
        if q["ball"] is not None and self.in_home(q["ball"]):
            self.log("fetch_done", ball=[round(v, 3) for v in q["ball"]]); self.fetch_inc = None; return True
        return False
    def path_watch(self, tgt):
        """a drive back into the home area (after a fetch, a stop, a switch of halves): no information, another robot
        inside the home area (its watched area) or within the corridor of its way stops it"""
        def corridor(q):
            p = self.our_pos(q)
            bb = blockers(q.get("others"), [(p, tgt)], self.corridor) if p else []
            return ("a robot in the corridor: " + ", ".join(f"{n} at ({x:+.2f}, {y:+.2f})" for n, x, y, _ in bb)) if bb else None
        def watch():
            q = self.snap(); mm = self.mismatch(q)
            if mm: self.on_mismatch(mm); return "not ours: " + mm
            ni = self.no_info(q)
            if ni: return "no information: " + ni
            w = [v for v in self.intrusion(q) if v[3] == "inside"]
            if w: self.last_intrusion = (w, self.our_pos(q)); return "another robot: " + ", ".join(f"{n} at ({x:+.2f}, {y:+.2f})" for n, x, y, _ in w)
            return corridor(q)
        return watch
    def drive_home(self, s, label, after_fetch=False):
        """our robot outside the home area: first no other robot inside the home area (the hold of intrusion_hold, its
        switch rule included), then wait until no robot is inside it and its surroundings (no robot within
        intrude_dist) or its straight way back (no robot within corridor m) have been known and clear for clear_s,
        then one watched goto 0.3 m inside (path_watch). While it waits, ok() records why it is not clear: no
        information (no vision, which robot is ours), a robot inside our half, robots on the way back. The alert names
        that reason and where our robot stands: after info_alert s for the first two, after fetch_wait s for robots on
        the way; then reminders every realert s (at most max_alerts). A robot not yet named in this wait seen inside
        our half is alerted at once (as in intrusion_hold, each robot its own backoff), and so is which robot is ours
        lost. -> True when the robot arrived"""
        if not self.ensure_home_clear(inside_only=True): return False
        s = self.look()
        if not self.robot_seen(s): return False
        r0 = s["robot"][:2]; tgt = into(self.home, r0); why = dict(r=None, info=False)
        def ok(q):
            ni = self.no_info(q)
            if ni: why.update(r=ni, info=True); return False
            w = self.intrusion(q); inside = [v for v in w if v[3] == "inside"]
            if inside: why.update(r=self.robots_text(inside), info=True); return False
            p = self.our_pos(q) or r0
            bb = blockers(q.get("others"), [(p, tgt)], self.corridor) if len(inside) < len(w) else []
            if bb:
                at = ", ".join(f"{self.who(n)} at ({x:+.2f}, {y:+.2f})" for n, x, y, _ in bb)
                why.update(r=(f"robots {at} block the way back" if len(bb) > 1 else f"another robot ({at}) blocks the way back"), info=False)
                return False
            return True
        t0 = t_bad = self.clock(); n = 0; t_al = None; halted = self.motion_halted; h = dict(named=set(), first={}); lost_sent = False
        while True:
            if self.out_of_time(): return False
            q = self.look(); now = self.clock()
            mm = self.mismatch(q)
            if mm: self.on_mismatch(mm); return False
            if ok(q):
                if now - t_bad >= self.clear_s: break
            else:
                t_bad = now; self.holding = self.holding or dict(since=now, cause=why["r"]); self.holding["cause"] = why["r"]
                p = self.our_pos(q) or (self.us_last and (self.us_last["x"], self.us_last["y"])) or r0
                at = f"; our robot at ({p[0]:+.2f}, {p[1]:+.2f}), outside our half: the session waits"
                ins = [] if self.no_info(q) else [v for v in self.intrusion(q) if v[3] == "inside"]
                if q.get("us_lost") and not lost_sent:                   # which robot is ours was lost: at once
                    since = self.ours_lost and self.ours_lost["since"]
                    if self.alert(f"{why['r']}{at}; {self.lost_tail()}", kind="ours_lost", since=since,
                                  force=since is not None and now - since >= self.ALERT_LASTING):
                        lost_sent = True; n = max(n, 1); t_al = now; self.motion_halted = True
                elif ins and self.alert_new_robots(ins, h, now, at):    # a robot new in our half: at once
                    n = max(n, 1); t_al = now; self.motion_halted = True
                else:
                    due = (now - t0 >= (self.info_alert if why["info"] else self.fetch_wait)) if n == 0 else \
                          (n < self.max_alerts and now - t_al >= self.realert)
                    if due:
                        msg = f"{why['r']}{at}" + (f" (reminder {n + 1}/{self.max_alerts})" if n else "")
                        if self.alert(msg, kind="way_back", since=t0 if n == 0 else t_al, force=n > 0): n += 1; t_al = now; self.motion_halted = True
            self.sleep(0.5)
        self.motion_halted = halted; self.holding = None
        self.log(label, robot=[round(v, 3) for v in r0], to=list(tgt), after_fetch=after_fetch)
        st = dict(kind="goto", x=tgt[0], y=tgt[1], v=0.4, label="back into the home area", n=0, timeout=30)
        if not self.motion_ok(st): return False
        r = self.run_step(st, watch=self.path_watch(tgt))
        return r.get("ok")
    def retreat(self, s):
        """after a stopped fetch: back into the home area, only once the surroundings or the way back are clear"""
        if not self.robot_seen(s) or in_rect(s["robot"], robot_area(self.home), 0.05): return True
        return self.drive_home(s, "retreat", after_fetch=True)
    def after_home(self, s, fetched=False):
        """the robot outside the home area between steps: drive it back inside first (the same rule as the retreat)"""
        if self.home is None or not self.robot_seen(s) or in_rect(s["robot"], robot_area(self.home), 0.05): return True
        return self.drive_home(s, "return_home", after_fetch=fetched)

    # ---- the ball
    def at_bad(self, s):
        return self.bad_ball is not None and s["ball"] is not None and math.hypot(s["ball"][0] - self.bad_ball[0], s["ball"][1] - self.bad_ball[1]) < 0.05
    def ball_ready(self, s):
        return s["ball_class"] in PLACEABLE and not self.at_bad(s) and self.in_home(s["ball"])
    def ball_back(self, s):
        return s["ball_class"] in IN_FIELD and s["ball_still"] and not self.at_bad(s) and self.in_home(s["ball"])
    def ball_problem(self, s):
        """the ball cannot be used: try one placement toward the field if the placer can, else alert once"""
        s = self.wait_ball_rest(s)
        if self.ball_ready(s):
            if self.ball_incident: self.log("ball_back", where=s["ball_where"]); self.ball_incident = None
            self.bad_ball = None
            return True
        if self.home is not None and s["ball_class"] in PLACEABLE and not self.at_bad(s) and not self.in_home(s["ball"]):
            if not self.motion_halted and self.fetch(s): return True
            return False                                        # the fetch alerts once when the corridor stays blocked
        if self.ball_incident is None:
            self.ball_incident = dict(cls=s["ball_class"], since=self.clock(), alerts=0, tried=self.at_bad(s))
            self.log("ball_unusable", cls=s["ball_class"], where=s["ball_where"])
        inc = self.ball_incident
        if self.home is not None and s["ball"] is not None and not in_rect(s["ball"], self.home, PLACER_MARGIN + 0.05):
            inc["tried"] = True                                  # not in our half: no rescue across the field, a person
        if self.home is not None and not inc["tried"] and s["ball"] is not None:
            inc["tried"] = True; tgt = into(self.home, s["ball"]); can = placer_can(s["ball"], tgt)
            self.log("ball_rescue", target=tgt, placer_can=can)
            if can and not self.busy_now() and self.robot_seen(s) and not self.motion_halted:
                st = dict(kind="place", x=tgt[0], y=tgt[1], label="ball rescue", n=0,
                          env={"SOLO_BOX": clip_rects(path_box([s["robot"][:2], s["ball"], tgt]), robot_area(self.home))})
                if not self.motion_ok(st): return False
                r = self.run_step(st); self.after(st, r)
                s = self.wait_ball_rest(self.look())
                if self.ball_ready(s): self.log("ball_back", where=s["ball_where"]); self.ball_incident = None; return True
        if not inc["tried"] and s["ball_class"] != "lost" and not self.motion_halted and self.robot_seen(s):
            inc["tried"] = True
            tgt = reach_point(s["ball"]); can = placer_can(s["ball"], tgt)
            self.log("ball_rescue", target=tgt, placer_can=can)
            if can and not self.busy_now():
                self.side_place(tgt[0], tgt[1], "ball rescue")
                s = self.wait_ball_rest(self.look())
                if self.ball_ready(s): self.log("ball_back", where=s["ball_where"]); self.ball_incident = None; return True
        if inc["alerts"] == 0: self.ball_alert(s)
        return False
    def ball_alert(self, s):
        inc = self.ball_incident
        if s["ball_class"] == "lost":
            last = self.world.snapshot()["ball"]
            since = time.strftime("%H:%M:%S", time.localtime(self.clock() - s["ball_age"]))
            msg = f"ball not seen since {since}" + (f" (last at ({last[0]:+.2f}, {last[1]:+.2f}))" if last else "") + ": put a ball anywhere inside the field"
        elif self.at_bad(s):
            msg = f"{s['ball_where']}: the robot cannot push it from there: put it anywhere inside the field"
        else:
            msg = f"{s['ball_where']}: put it anywhere inside the field"
        if inc["alerts"] > 0: msg += f" (reminder {inc['alerts'] + 1}/{self.max_alerts})"
        if self.alert(msg, kind="ball_" + ("lost" if s["ball_class"] == "lost" else "bad" if self.at_bad(s) else "where"), since=inc["since"]):
            inc["alerts"] += 1; inc["t_alert"] = self.clock()
    def wait_for_ball(self):
        """nothing ball-free left: wait for a ball resting inside the field, re-alerting"""
        self.log("waiting", for_="ball")
        while not self.out_of_time():
            s = self.look()
            if self.ball_back(s): self.log("ball_back", where=s["ball_where"]); self.ball_incident = None; self.bad_ball = None; self.fetch_inc = None; return True
            if self.home is not None and s["ball_class"] in PLACEABLE and s["ball"] is not None and not self.in_home(s["ball"]) and not self.motion_halted:
                if self.fetch(s, wait=0): continue
                fi = self.fetch_inc
                if fi and 0 < fi["alerts"] < self.max_alerts and self.clock() - fi.get("t_alert", 0) >= self.realert:
                    if self.alert(f"ball at ({s['ball'][0]:+.2f}, {s['ball'][1]:+.2f}) still in the other half: fetch it when you can (reminder {fi['alerts'] + 1}/{self.max_alerts})",
                                  kind="fetch_reminder", since=fi["since"]):
                        fi["alerts"] += 1; fi["t_alert"] = self.clock()
            inc = self.ball_incident
            if inc and inc["alerts"] < self.max_alerts and self.clock() - inc.get("t_alert", 0) >= self.realert: self.ball_alert(s)
            self.sleep(1.0)
        return False

    # ---- the robot
    def hold(self, what, msg, cleared):
        """stop motion, alert (repeated while it is held back), wait until cleared(snapshot) (re-alerts), resume"""
        self.motion_halted = True; t0 = self.clock(); n = 1 if self.alert(msg, kind=what, since=t0) else 0; ta = self.clock()
        self.holding = dict(since=t0, cause=msg)
        while not self.out_of_time():
            s = self.look()
            if cleared(s): self.log("hold_cleared", what=what); self.motion_halted = False; self.holding = None; return True
            if n == 0:
                if self.alert(msg, kind=what, since=t0): n = 1; ta = self.clock()
            elif n < self.max_alerts and self.clock() - ta >= self.realert:
                if self.alert(msg + f" (reminder {n + 1}/{self.max_alerts})", kind=what + "_reminder", since=t0): n += 1; ta = self.clock()
            self.sleep(1.0)
        return False
    def robot_seen(self, s):
        return s["robot"] is not None and s["robot_age"] <= self.robot_lost
    def ensure_robot(self):
        s = self.look(); t0 = self.clock()
        while not self.robot_seen(s) and self.clock() - t0 < self.robot_wait and not self.out_of_time():
            self.sleep(0.5); s = self.look()
        if self.robot_seen(s): return True
        since = time.strftime("%H:%M:%S", time.localtime(self.clock() - s["robot_age"])) if s["robot"] else "the session started"
        last = f", last at ({s['robot'][0]:+.2f}, {s['robot'][1]:+.2f})" if s["robot"] else ""
        if s["robot"] and s["robot_clear"] < NEAR_WALL: last += f" against the {s['robot_near']}"
        if s.get("page_age", 0) > 3: msg = f"the field page does not answer (robot last seen {since}{last}): the session waits; restart hq.py"
        else: msg = f"robot not seen since {since}{last}: check it is on the field with its tag up"
        return self.hold("robot", msg, self.robot_seen)
    def robot_clear(self, s):
        return self.robot_seen(s) and s["robot_clear"] > CLEAR_WALL and abs(s["robot"][0]) < FX + 0.05 and abs(s["robot"][1]) < FY + 0.05
    def free_robot(self, s, why):
        """one slow goto FREE_IN inside the field, straight away from the wall; not moved 0.1 m: alert and wait"""
        x, y = s["robot"][:2]; tgt = free_point(x, y); name = s["robot_near"]
        if self.home is not None: ra = robot_area(self.home); tgt = (clip(tgt[0], ra[0], ra[1]), clip(tgt[1], ra[2], ra[3]))
        self.log("robot_stuck", at=[round(x, 3), round(y, 3)], near=name, clearance=s["robot_clear"], why=why, free_to=tgt)
        st = dict(kind="goto", x=tgt[0], y=tgt[1], v=0.25, label="free the robot", on_fail="skip", n=0, timeout=30)
        if not self.busy_now() and not self.not_ours and (self.home is None or self.ensure_home_clear()): self.run_step(st)
        s2 = self.look()
        if self.robot_seen(s2) and math.hypot(s2["robot"][0] - x, s2["robot"][1] - y) >= 0.1:
            self.log("robot_freed", at=[round(v, 3) for v in s2["robot"][:2]]); return True
        return self.hold("stuck", f"robot stuck against the {name} at ({x:+.2f}, {y:+.2f}): free it and put it inside the field",
                         lambda q: self.robot_clear(q) and math.hypot(q["robot"][0] - x, q["robot"][1] - y) >= 0.1)

    # ---- other programs
    def own_step(self):
        """the pids of the step this session runs now (its process group): not another program"""
        pid = self.runner.child_pid() if self.runner is not None and hasattr(self.runner, "child_pid") else None
        if pid is None or not self.procs: return set()
        return ({pid} | set(self.procs.group(pid))) if hasattr(self.procs, "group") else {pid}
    def busy_now(self, allow=()):
        b = [n for n in busy(self.procs, exclude=self.own_step()) if n not in allow] if self.procs else []
        if self.world is not None and self.world.snapshot().get("page_driving"): b.append("the field page (a click target or a placement)")
        return b
    def ensure_idle(self, allow=()):
        t0 = self.clock(); b = self.busy_now(allow)
        while b and self.clock() - t0 < self.busy_wait and not self.out_of_time():
            self.sleep(1.0); b = self.busy_now(allow)
        if not b: return True
        self.log("busy", programs=b)
        return self.hold("busy", f"another program commands the robot ({', '.join(b)}): the session waits", lambda q: not self.busy_now(allow))

    # ---- one step
    def command(self, st, retry=False):
        k = st["kind"]; py, T = self.py, HERE; e = lambda v: expand(v, self.vs)
        if k == "place":
            a = [py, f"{T}/place_ball.py", self.team, str(st["x"]), str(st["y"])] + [str(v) for v in e(st.get("opts", []))]
            if retry: a += ["--carry", "0.25"]
            return a
        if k == "test":
            return [py, f"{T}/characterize.py", self.team, st["test"]] + [str(v) for v in st.get("args", [])] + ["--out", e(st.get("out", "${SOLO_DATA}/logs"))]
        if k == "series":
            a = [py, f"{T}/series.py", self.team, e(st["plan"])] + [str(v) for v in e(st.get("args", []))]
            if st.get("from"): a += ["--from", str(st["from"])]
            if st.get("max"): a += ["--max", str(st["max"])]
            return a
        if k == "goto":
            return [py, f"{T}/goto.py", self.team, str(st["x"]), str(st["y"]), "--v", str(st.get("v", 0.4))]
        if k == "cmd":
            return [str(v) for v in e(st["argv"])]
        return None
    def env_for(self, st, retry=False):
        env = dict(os.environ)
        for src in (self.head.get("env", {}), st.get("env", {})):
            for k, v in expand(src, self.vs).items(): env[k] = str(v)
        if retry:
            env["SOLO_BOX"] = widen(env.get("SOLO_BOX", ""))
            if self.home is not None and st.get("label") != "fetch the ball": env["SOLO_BOX"] = clip_rects(env["SOLO_BOX"], robot_area(self.home))
        return env
    def timeout_of(self, st):
        if "timeout" in st: return float(st["timeout"])
        if st["kind"] == "strategy": return float(st.get("seconds", 0)) + 15.0
        if st["kind"] == "place":
            o = st.get("opts", []); return (float(o[o.index("--timeout") + 1]) if "--timeout" in o else 60.0) + 20.0
        return float(TIMEOUTS.get(st["kind"], 60))

    def run_step(self, st, retry=False, watch=None):
        """run one step; -> outcome dict(ok, line, data, rc, timed_out, stuck, stopped)"""
        k = st["kind"]; t0 = self.clock()
        if k == "marker":
            self.marker(expand(st["text"], self.vs), st.get("data", "")); return dict(ok=True, line="marker")
        if k == "wait":
            end = self.clock() + float(st["seconds"])
            while self.clock() < end and self.stop_req < 2: self.sleep(min(0.5, end - self.clock()))
            return dict(ok=True, line="waited")
        if k == "strategy":
            return self.run_strategy(st)
        argv = self.command(st, retry); env = self.env_for(st, retry)
        tee = expand(st["tee"], self.vs) if st.get("tee") else None
        if k in MOTION:                                  # every motion child is watched: the watch is chosen here
            watch = watch if watch is not None else self.watch_for(st)
            if watch is None and self.home is not None: raise RuntimeError(f"motion step {st.get('label', k)!r} without its watch")
        if k in MOTION: self.motion(True)
        try: r = self.runner.run(argv, env, self.timeout_of(st), tee=tee, watch=watch)
        finally: self.motion(False)
        r["ok"] = self.judge(st, r) and not r.get("stopped"); r["stuck"] = "robot stuck" in (r.get("line") or "")
        self.log("step", n=st.get("n"), label=st.get("label", ""), step_kind=k, cmd=argv,
                 env={v: env[v] for v in ("SOLO_BOX", "SOLO_RAMP", "SOLO_BRAKE") if v in env}, start=round(r.get("t0", t0), 3),
                 end=round(r.get("t1", self.clock()), 3), rc=r.get("rc"), timed_out=r.get("timed_out"), ok=r["ok"], result=r.get("line"), retry=retry,
                 **({"stopped": r["stopped"]} if r.get("stopped") else {}))
        return r
    def judge(self, st, r):
        k, d, line = st["kind"], r.get("data"), r.get("line") or ""
        if r.get("timed_out"): return False
        if k == "place": return bool(d and d.get("ok"))
        if k == "test": return r.get("rc") == 0 and isinstance(d, dict)
        if k == "goto": return line.startswith("arrived")
        return r.get("rc") == 0
    def run_strategy(self, st):
        t0 = self.clock()
        if not st.get("on", True):
            ok = self.strat.off(); self.log("step", n=st.get("n"), label=st.get("label", "strategy off"), step_kind="strategy", on=False, ok=ok, start=t0, end=self.clock())
            return dict(ok=ok, line="strategy off" if ok else "strategy still running")
        if not self.strat.can_start():
            self.log("step", n=st.get("n"), label=st.get("label", ""), step_kind="strategy", ok=False, result="no strategy command line (start it with run_real.sh, then off)")
            return dict(ok=False, line="strategy cannot start")
        params = dict(st.get("params") or {})
        if st.get("side"): params["side"] = st["side"]
        self.motion(True)
        on = self.strat.on(params) if params else self.strat.on(); t_on = self.clock(); end = t_on + float(st.get("seconds", 10))
        if not on: self.log("strategy_refused", n=st.get("n"), why=getattr(self.strat, "why", ""))
        track = []; stopped = None; ended = None; t_out = None; ball_box = rects(st.get("end_ball_out", "")); guard = float(st.get("edge_guard", self.edge_guard))
        while on and self.clock() < end and self.stop_req < 2:
            s = self.snap(); track.append((round(self.clock() - t_on, 2), s["robot"] and [round(v, 3) for v in s["robot"][:2]], s["ball"] and [round(v, 3) for v in s["ball"]]))
            if self.home is not None:                   # the rule of every watched motion, then the edge (fresh positions only)
                why = self.motion_check(s)
                if why: stopped = why; break
                p = self.our_pos(s)
                if near_open_edge(p, self.home, guard):
                    ended = f"robot at ({p[0]:+.2f}, {p[1]:+.2f}), within {guard} m of the home area's open edge"; break
            if ball_box and s.get("ball") is not None and not any(in_rect(s["ball"], r) for r in ball_box):
                t_out = t_out if t_out is not None else self.clock()
                if self.clock() - t_out >= 0.5: ended = f"ball out of {st['end_ball_out']} at {t_out - t_on:.1f} s"; break
            self.sleep(0.1)
        off = self.strat.off(); self.motion(False)
        self.log("step", n=st.get("n"), label=st.get("label", ""), step_kind="strategy", on=True, seconds=st.get("seconds"), started=on, stopped=off,
                 ok=on and off and not stopped, start=round(t0, 3), end=round(self.clock(), 3), params=params, ended_early=ended,
                 interrupted=stopped, track=track[::2])
        if not off: self.stop_why = "the strategy node did not stop"; self.stop_req = max(self.stop_req, 1); self.alert("the strategy node did not stop: the session ends; stop it on the field page", kind="strategy", urgent=True); self.stop_req = max(self.stop_req, 1)
        return dict(ok=on and off and not stopped, line=f"strategy on {st.get('seconds')} s" + ("" if on and off else " FAILED") + (f" (ended: {ended})" if ended else ""),
                    stopped=stopped)

    def after(self, st, r, home=True):
        """look at the world after a step and handle the robot; -> True when the robot can go on"""
        s = self.look(); s = self.wait_ball_rest(s) if is_motion(st) else s
        self.log("world", after=st.get("n"), robot=s["robot"] and [round(v, 3) for v in s["robot"]], robot_age=s["robot_age"],
                 ball=s["ball"] and [round(v, 3) for v in s["ball"]], ball_age=s["ball_age"], ball_where=s["ball_where"], ball_class=s["ball_class"],
                 robot_clear=s.get("robot_clear"), robot_near=s.get("robot_near"))
        if not is_motion(st): return True
        if not self.robot_seen(s): return self.ensure_robot()
        failed = not r.get("ok")
        if r.get("stuck") or (failed and s["robot_clear"] < NEAR_WALL):
            return self.free_robot(s, r.get("line") or "")
        if self.home is not None and home and not self.motion_halted:
            self.after_home(s)                                   # our robot outside the home area: back inside first
            q = self.look()
            if (not self.no_ball and q["ball"] is not None and q["ball_class"] in PLACEABLE and not self.in_home(q["ball"])
                    and q["ball_still"] and not self.at_bad(q)):
                self.fetch(q)                                    # a ball resting in the other half: fetch it now
        return True

    def handle_fail(self, st, r, pending):
        how = st.get("on_fail", "skip")
        if how == "retry" and not st.get("_retried"):
            st["_retried"] = True; pending.insert(0, st); self.log("decision", n=st.get("n"), action="retry"); return
        if how == "stop":
            self.stop_why = f"step {st.get('n')} failed (on_fail stop)"; self.stop_req = max(self.stop_req, 1)
        if how == "alert":
            self.alert(f"step {st.get('n')} ({st.get('label', st['kind'])}) failed: {(r.get('line') or '')[:160]}; the session goes on",
                       kind="step_failed", robot=f"step {st.get('n')}")
        if st.get("group"): self.skipped_groups.add(st["group"])
        self.skipped.append(st.get("n")); self.log("decision", n=st.get("n"), action=how, group_skipped=st.get("group"))

    def motion_ok(self, st):
        """the guard before every motion step: the robot seen and ours, no other commanding program, the page not driving"""
        if self.not_ours: return False
        if self.home is not None:
            mm = self.mismatch(self.snap())
            if mm: self.on_mismatch(mm); return False
        return self.ensure_robot() and self.ensure_idle(allow=("solo_node",) if st["kind"] == "strategy" else ())
    def execute(self, st, pending):
        k = st["kind"]
        if is_motion(st):
            if self.motion_halted: self.skipped.append(st.get("n")); self.log("decision", n=st.get("n"), action="skipped: motion stopped"); return
            self.current = st                          # a switch of halves while it waits mirrors it (self.current)
            try:
                if not self.motion_ok(st): pending.insert(0, self.current); return
                if self.home is not None:
                    if not self.after_home(self.look()): pending.insert(0, self.current); return
                    if not self.ensure_home_clear(): pending.insert(0, self.current); return
                    if self.current is not st: pending.insert(0, self.current); return
            finally:
                self.current = None
            if self.no_ball and not self.foreign_ball_clear(st, pending): return
        if self.out_of_time() and not st.get("final"): pending.insert(0, st); return
        r = self.run_step(st)
        if r.get("stopped"):
            return self.interrupted(st, r, pending)
        if k == "place" and not r["ok"]:
            status = (r.get("data") or {}).get("status", "")
            if status.startswith("unreachable") and "spot" not in status:      # the ball's position, not the target
                st["_requeued"] = st.get("_requeued", 0) + 1
                b = (r.get("data") or {}).get("ball") or self.look()["ball"]
                if b: self.bad_ball = tuple(b[:2])
                if st["_requeued"] <= 2:
                    self.after(st, r); pending.insert(0, st); self.log("decision", n=st.get("n"), action="ball unreachable: the step waits for the ball"); return
            elif (r.get("timed_out") or status.startswith("failed: timeout") or status.startswith("failed: robot left")) and not r.get("stuck") and st.get("n") not in self.retried_place:
                self.retried_place.add(st.get("n")); self.log("decision", n=st.get("n"), action="placement retry: wider box, carry 0.25")
                if self.after(st, r) and not self.out_of_time() and self.motion_ok(st) and (self.home is None or self.ensure_home_clear()):
                    r = self.run_step(st, retry=True)
                    if r.get("stopped"): return self.interrupted(st, r, pending)
        ok_go = self.after(st, r)
        if r["ok"]: self.done.append(st.get("n"))
        else: self.handle_fail(st, r, pending)
        return ok_go


    def interrupted(self, st, r, pending):
        """a step the supervisor stopped: another robot (hold, alert, repeat the step from its start) or a foreign ball
        near our robot in a session without the ball (quiet: the step goes to the end of the list once)"""
        why = r["stopped"]
        if why.startswith("not ours"): return False
        if why.startswith("another robot") or why.startswith("no information"):
            st["_interrupted"] = st.get("_interrupted", 0) + 1
            self.log("decision", n=st.get("n"), action="step interrupted: " + why.split(":")[0], why=why, times=st["_interrupted"])
            w = getattr(self, "last_intrusion", ([("another robot", 0.0, 0.0, "inside")], None))[0] if why.startswith("another robot") else []
            if st["_interrupted"] <= 5: pending.insert(0, st)            # queued first: a switch of halves mirrors it
            else: self.skipped.append(st.get("n")); self.log("decision", n=st.get("n"), action="skipped: interrupted 5 times")
            s = self.look(); self.intrusion_hold(w, self.our_pos(s), during=True, cause=why)
            self.after(st, dict(r, ok=True))
            return True
        self.log("foreign_ball", n=st.get("n"), action="step stopped: a ball near our robot inside the test area", why=why)
        self.defer(st, pending); self.after(st, dict(r, ok=True))
        return True
    def defer(self, st, pending):
        if not st.get("_deferred"):
            st["_deferred"] = 1; pending.append(st); self.log("decision", n=st.get("n"), action="moved to the end of the list (a ball in its area)")
        else:
            self.skipped.append(st.get("n")); self.log("decision", n=st.get("n"), action="skipped: a ball in its area twice")
    def foreign_ball_clear(self, st, pending):
        """no-ball sessions: a ball within 0.15 m of the step's area -> hold up to ball_hold s (quiet), then the step goes
        to the end of the list; the ball is never touched"""
        boxes = rects(st["keep_ball_out"]) if st.get("keep_ball_out") else (st.get("_boxes") or [])
        def inside(q):
            b = q["ball"] if q.get("ball") is not None and q.get("ball_age", 9) < 1.5 else None
            return b is not None and any(in_rect(b, r, 0.15) for r in boxes)
        s = self.look()
        if not boxes or not inside(s): return True
        self.log("foreign_ball", n=st.get("n"), ball=[round(v, 3) for v in s["ball"]], action=f"a ball in the test area: waiting up to {self.ball_hold:.0f} s")
        t0 = self.clock()
        while self.clock() - t0 < self.ball_hold and not self.out_of_time():
            self.sleep(0.5); s = self.look()
            if not inside(s): self.log("foreign_ball", n=st.get("n"), action="the ball left the test area", waited=round(self.clock() - t0, 1)); return True
        self.defer(st, pending)
        return False
    def needs(self, st):
        return False if self.no_ball else st.get("needs_ball", st["kind"] == "place")
    def ball_blocks(self, st, s):
        """the ball lies inside the keep_ball_out box this ball-free step must keep clear"""
        return bool(st.get("keep_ball_out")) and ball_in_box(s["ball"] if s["ball_age"] < 1.5 else None, st["keep_ball_out"])
    def side_place(self, x, y, label):
        """a placement the supervisor adds (park the ball, rescue it): the same guard as every motion step before it,
        the robot checked after it; None when the guard did not let it start"""
        st = dict(kind="place", x=x, y=y, label=label, n=0)
        if label == "ball rescue": st["env"] = {"SOLO_BOX": ""}           # the placer's own area: field + margin
        if self.home is not None:                                          # never beyond the home area
            st["env"] = {"SOLO_BOX": clip_rects(self.env_for(st).get("SOLO_BOX") or ",".join(map(str, PLACER_AREA)), robot_area(self.home))}
        if self.motion_halted or not self.motion_ok(st) or (self.home is not None and not self.ensure_home_clear()):
            self.log("decision", action=f"{label}: not started (motion stopped, robot not seen or another program)"); return None
        r = self.run_step(st); self.after(st, r); return r
    def pick(self, pending):
        """-> (step, None), or (None, blocked step), or (None, None) when only ball steps are left and the ball is unusable.
        Ball steps keep their order; ball-free steps go ahead of them while the ball cannot be used."""
        for st in [p for p in pending if p.get("group") in self.skipped_groups]:
            pending.remove(st); self.skipped.append(st.get("n")); self.log("decision", n=st.get("n"), action="skipped: its group failed")
        if self.no_ball: return (pending[0], None) if pending else (None, None)
        s = self.look(); ball_ok = None; blocked = None
        for st in pending:
            if self.needs(st):
                if ball_ok is None:
                    ball_ok = self.ball_ready(s) or self.ball_problem(s)
                    s = self.look()
                if ball_ok: return st, None
                continue
            if self.ball_blocks(st, s):
                if st.get("park") and self.ball_ready(s) and not self.motion_halted:
                    self.log("decision", n=st.get("n"), action=f"ball inside the test box: parked at {st['park']} first")
                    if self.side_place(st["park"][0], st["park"][1], "park the ball") is None: return None, None
                    s = self.look()
                    if not self.ball_blocks(st, s): return st, None
                blocked = blocked or st
                continue
            if st is not pending[0]: self.log("decision", n=st.get("n"), action="ball-free step taken ahead of the ball steps")
            return st, None
        return None, (blocked if ball_ok is not False else None)

    def run(self):
        main = [st for st in self.steps if not st.get("final") and not st.get("_refused")]; final = [st for st in self.steps if st.get("final") and not st.get("_refused")]
        self.log("session_start", name=self.head.get("name", ""), steps=len(self.steps), budget_s=round(self.t_end - self.t_start),
                 home=self.home, no_ball=self.no_ball, park=self.park)
        for st in self.steps:
            if st.get("_refused"): self.skipped.append(st.get("n")); self.log("refused", n=st.get("n"), label=st.get("label", ""), why=st["_refused"])
        pending = list(main); self.pending = pending
        why = self.preflight()
        if why:
            self.stop_why = f"refused to start: {why}"; self.log("refused_start", why=why); print(f"refused to start: {why}", flush=True)
            self.alert(f"session not started: {why}", kind="not_started", urgent=True); self.drain(); return self
        try:
            if self.home_auto and self.home is not None: self.choose_half()
            self.loop(pending)
        except Exception as e:                              # a fault of the supervisor itself: end cleanly
            import traceback
            self.log("error", error=repr(e), trace=traceback.format_exc()[-800:]); self.stop_why = f"supervisor error {e!r}"
        self.stop_why = self.stop_why or ("budget used" if self.clock() >= self.t_end else ("all steps done" if not pending else "stopped"))
        self.finish(final, pending)
        return self

    def loop(self, pending):
        while pending and not self.out_of_time():
            st, blocked = self.pick(pending)
            if st is None and blocked is not None:          # the ball sits in its box and cannot be parked: on_fail
                pending.remove(blocked); self.log("step", n=blocked.get("n"), label=blocked.get("label", ""), ok=False, result="ball inside the test box")
                self.handle_fail(blocked, dict(line=f"the ball lies inside the test box {blocked['keep_ball_out']}"), pending); continue
            if st is None:
                if not pending or not self.wait_for_ball(): break
                continue
            pending.remove(st)
            if self.home is not None and st.get("_half", 1) != self.half_sign:    # queued before a switch by another path
                st = mirror_step(st, self.plan_dir(), self.vs); self.prepare([st])
            if st.get("_refused"):                      # a step mirrored into a half where it does not fit
                self.skipped.append(st.get("n")); self.log("refused", n=st.get("n"), label=st.get("label", ""), why=st["_refused"]); continue
            self.execute(st, pending)

    def finish(self, final, pending):
        try:
            if self.strat and self.strat.running(): self.log("end", action="strategy off", ok=self.strat.off())
        except Exception as e:
            self.log("end", action="strategy off", error=str(e))
        for st in final:
            if st["kind"] in ("marker", "cmd", "wait"): self.run_step(st)
        if not any(st["kind"] == "marker" for st in final): self.marker("auto session end", "skip")
        summary = f"session ended: {self.stop_why} ({len(self.done)} steps done, {len(self.skipped)} skipped, {len(pending)} not run)"
        held = {k: v["between"] for k, v in self.alert_keys.items() if v["between"]}
        if held:
            summary += "; repeats not alerted: " + ", ".join(f"{v} {k[0]}" + (f" ({self.who(str(k[1]))})" if k[1] else "") for k, v in sorted(held.items(), key=str))
        if self.dead is not None: summary += f"; {self.dead_text()}"
        if self.holding is not None:
            h = self.holding
            summary += f"; ended while holding since {time.strftime('%H:%M', time.localtime(h['since']))}: {h['cause']}{self.outside_note()}"
        self.log("session_end", why=self.stop_why, done=self.done, skipped=self.skipped, not_run=[s.get("n") for s in pending], alerts=len(self.alerts))
        self.alert(summary, kind="end", urgent=True)
        if self.sender.threaded and not self.sender.alive():           # nobody sends the queue: once, directly
            try: r = self.notifier.send(summary, "field session") if self.notifier is not None else "no notifier"
            except Exception as e: r = f"error {e!r}"
            self.log("alert_sent_directly", message=summary, sent=r, note="the alert sender's thread had ended")
            return
        self.drain()
    DRAIN = 20.0                        # s: at the end, how long the session waits for the queued alerts to go out
    def drain(self):
        ok = self.sender.drain(self.DRAIN, self.sleep)
        if not ok: self.log("alerts_not_delivered", pending=self.sender.pending(), note=f"the session ends with alerts not delivered after {self.DRAIN:.0f} s")
        return ok


# ------------------------------------------------------------------ simulation (tests and --sim): no process, no network
class SimClock:
    def __init__(self): self.t = time.time()
    def __call__(self): return self.t
    def sleep(self, dt): self.t += max(dt, 0.01)


class SimProcs:
    """a process table: [(pid, argv)]; this process is pid 1"""
    def __init__(self, table=()): self.rows = list(table)
    def table(self): return list(self.rows)
    def mine(self): return {1}


class SimStrategy:
    def __init__(self, procs, can=True):
        self.procs = procs; self.can = can; self.ons = 0; self.offs = 0; self.cmdf = "solo.cmd"; self.params = []; self.last_argv = None
    def running(self): return any(is_instance(a, "solo_node") for _, a in self.procs.table())
    def can_start(self): return self.can or self.running()
    def on(self, params=None):
        self.ons += 1; self.params.append(dict(params or {}))
        self.last_argv = with_params(["python3", "/x/lib/solo_strategy/solo_node", "--ros-args", "-p", "side:=positive"], params or {})
        if not self.running(): self.procs.rows.append((999, self.last_argv))
        return True
    def off(self):
        self.offs += 1; self.procs.rows = [r for r in self.procs.rows if r[0] != 999]; return True


class SimNotifier:
    def __init__(self, echo=False): self.sent = []; self.echo = echo; self.quiet = []
    def notice(self, msg, title): self.quiet.append(msg); return "ok"
    def send(self, msg, title):
        self.sent.append(msg)
        if self.echo: print(f"*** NOTIFY: {title}: {msg}", flush=True)
        return "ok"


class SimRunner:
    """script: [dict(match="substring of the command", line=..., rc=0, dur=s, world={...}, timed_out=False, call=f)],
    consumed in order when the command matches the head; otherwise a default result: placements succeed and move
    the ball to the spot, gotos arrive, tests and series end well. A robot the step moves (a goto's point, the
    robot of "world", a move made by "call") drives there during the step: a straight line at no more than SPEED.
    REACT, DECEL: a step stopped while the robot drives leaves it moving for REACT s, then braking at DECEL m/s^2
    (ScriptedWorld.coast); DECEL None: it halts where it is. A characterize.py spin turns the robot on the spot at its
    rate for its time (default 1.5 s), starting SPIN_DELAY s into the step, unless TURNS is False (a robot that does
    not respond)"""
    SPEED = 1.0
    REACT, DECEL = 0.0, None
    SPIN_DELAY, TURNS = 1.0, True
    def __init__(self, clock, world, script=()):
        self.clock, self.world, self.script, self.calls, self.stops = clock, world, list(script), [], []
    def _robot(self):
        w = self.world
        return tuple(w.robot) if getattr(w, "robot_on", True) and getattr(w, "robot", None) is not None else None
    def _pass(self, dur, watch, move=None, turn=None):
        """the step's time on the simulated clock in 0.1 s polls; -> the watch's reason when it stops the step.
        move (a, b, T): the robot from a to b in T s; turn (rate, delay, T): the robot turns at rate from delay s for T s"""
        t0 = self.clock.t; end = t0 + dur; th0 = None
        while self.clock.t < end - 1e-9:
            t_prev = self.clock.t; self.clock.t = min(end, self.clock.t + 0.1)
            if turn and self._robot() is not None and t_prev - t0 < turn[1] + turn[2]:
                r = self._robot(); th0 = r[2] if th0 is None else th0
                k = min(max(self.clock.t - t0 - turn[1], 0.0), turn[2])
                self.world.set(robot=(r[0], r[1], th0 + turn[0] * k))
            if move:
                a, b, T = move; k = min(1.0, (self.clock.t - t0) / T) if T > 0 else 1.0
                self.world.set(robot=tuple(a[i] + k * (b[i] - a[i]) for i in range(2)) + tuple(b[2:]))
            if watch:
                why = watch()
                if why:
                    self.stops.append((self.clock.t, why))
                    if move and self.DECEL and self.clock.t < t0 + move[2] - 1e-9 and hasattr(self.world, "coast"):
                        a, b, T = move; self.world.coast(((b[0] - a[0]) / T, (b[1] - a[1]) / T), self.REACT, self.DECEL)
                    return why
        return None
    def run(self, argv, env, timeout, tee=None, watch=None):
        cmd = " ".join(argv); self.calls.append(dict(cmd=cmd, env=env, timeout=timeout, watched=watch is not None)); t0 = self.clock()
        def way(a, b, T):
            if a is None or b is None or math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-9: return None
            return (a, tuple(b), max(T, math.hypot(b[0] - a[0], b[1] - a[1]) / self.SPEED))
        if self.script and self.script[0]["match"] in cmd:
            e = self.script.pop(0); dur = e.get("dur", 5.0); r0 = self._robot(); move = None
            if e.get("call"):
                e["call"](); r1 = self._robot()
                if r0 is not None and r1 is not None and r1 != r0:         # the call moved the robot: it drives there
                    self.world.set(robot=r0); move = way(r0, r1, 0.0)
            wr = (e.get("world") or {}).get("robot")
            if move is None and wr is not None and r0 is not None: move = way(r0, wr, dur)
            why = self._pass(dur, watch, move)
            if why: return dict(rc=-2, timed_out=False, line=f"stopped by the supervisor: {why}", data=None, t0=t0, t1=self.clock(), stopped=why)
            if "world" in e: self.world.set(**e["world"])
            line = e["line"] if isinstance(e["line"], str) else json.dumps(e["line"])
            data, line = result_of([line])
            return dict(rc=e.get("rc", 0), timed_out=e.get("timed_out", False), line=line, data=data, t0=t0, t1=self.clock())
        move = None; turn = None
        if "goto.py" in cmd: move = way(self._robot(), (float(argv[3]), float(argv[4]), 0.0), 5.0)
        if "characterize.py" in cmd and argv[3] == "spin" and self.TURNS:
            turn = (float(argv[4]), self.SPIN_DELAY, float(argv[5]) if len(argv) > 5 and not argv[5].startswith("--") else 1.5)
        why = self._pass(5.0, watch, move, turn)
        if why: return dict(rc=-2, timed_out=False, line=f"stopped by the supervisor: {why}", data=None, t0=t0, t1=self.clock(), stopped=why)
        if "place_ball.py" in cmd:
            x, y = float(argv[3]), float(argv[4]); self.world.set(ball=(x, y))
            line = json.dumps(dict(ok=True, status="done", ball=[x, y], error=0.02))
        elif "goto.py" in cmd:
            x, y = float(argv[3]), float(argv[4]); self.world.set(robot=(x, y, 0.0)); line = f"arrived: robot at ({x:+.2f}, {y:+.2f}) after 3.0 s"
        elif "characterize.py" in cmd:
            line = json.dumps(dict(test=argv[3], args=argv[4:]))
        elif "series.py" in cmd:
            line = "series done"
        else:
            line = "ok"
        data, line = result_of([line])
        return dict(rc=0, timed_out=False, line=line, data=data, t0=t0, t1=self.clock())
    def abort(self): pass
    def cmds(self, key=""):
        return [c["cmd"] for c in self.calls if key in c["cmd"]]


SIM = {   # --sim scenarios: scripted results [match, line, world after] on top of the default "everything works"
    "ok": [],
    "corner": [dict(match="ram 1.5 0", line=dict(test="ram"), world=dict(ball=(2.52, 1.73)))],
    "goal": [dict(match="ram 1.2 0", line=dict(test="ram"), world=dict(ball=(2.33, 0.05)))],
    "stuck": [dict(match="place_ball.py blue 0.85", line=dict(ok=False, status="failed: robot stuck at (1.00, 1.65)"), world=dict(robot=(1.0, 1.65, 0.0))),
              dict(match="goto.py", line="vision lost the robot", world=dict(robot=(1.0, 1.65, 0.0)))],
    "timeout": [dict(match="place_ball.py", line=dict(ok=False, status="failed: timeout (60 s)")),
                dict(match="place_ball.py", line=dict(ok=False, status="failed: timeout (60 s)"))],
}
SIM_FIX = {"corner": (150, dict(ball=(0.2, -0.3))), "goal": (150, dict(ball=(0.2, -0.3))), "stuck": (150, dict(robot=(0.0, 0.0, 0.0)))}


# ------------------------------------------------------------------ dry run
DECISIONS = [
    ("ball inside the field / in a defense area / past a line within %.2f m" % PLACER_MARGIN, "ball steps run", "-"),
    ("ball in a goal, against a goal frame, in a corner, against a wall", "one place_ball run toward the nearest point 0.3 m inside, if BallPlacer can; ball-free steps go on", "'<where>: put it anywhere inside the field' (once; reminders only when nothing ball-free is left)"),
    ("ball not seen > ball-lost s", "ball-free steps go on; ball steps wait", "'ball not seen since T (last at ...): put a ball anywhere inside the field'"),
    ("place_ball: unreachable (ball)", "as the line above; the place step stays first among the ball steps (twice at most)", "as above"),
    ("place_ball: failed timeout / left the area", "once more: box +0.15 m, --carry 0.25; then on_fail", "only with on_fail alert"),
    ("place_ball: robot stuck, or abort/timeout with the robot within %.2f m of a wall or frame" % NEAR_WALL, "one goto.py at 0.25 m/s to 0.5 m inside the field; moved < 0.1 m: motion stops, wait", "'robot stuck against the <wall> at (x, y): free it and put it inside the field'"),
    ("robot not seen before a motion step", "wait robot-wait s, then motion stops and the session waits", "'robot not seen since T, last at (x, y): ...'"),
    ("another commanding program runs", "no motion step; after busy-wait s the session waits", "'another program commands the robot (...): the session waits'"),
    ("a test ABORT / non-zero exit / step time limit", "on_fail: skip (and the rest of its group) / retry once / stop / alert", "only with on_fail alert or stop"),
    ("budget used / Ctrl-C", "current step ends, strategy off, final steps, end marker", "'session ended: ...'"),
]
HOME_DECISIONS = [
    ("a step whose area is not inside the home area (or a strategy step without its side)", "refused when the session is loaded; listed here", "-"),
    ("another robot inside the home area, or within intrude-dist of ours, while a motion step runs (10 polls/s)",
     "the step is stopped at once (as the time limit stops it: SIGINT to its group, the tool's zeros); all motion holds until the home area has been clear for clear s; the step is repeated from its start",
     "'another robot (yellow 2) entered our half at (x, y); our robot stopped at (x, y)' once; after hold-limit s 'still in our half ...' (at most 3)"),
    ("the same between steps", "no motion step starts; the same hold", "'another robot (...) entered our half at (x, y); our robot waits at (x, y)'"),
    ("no information while a motion step runs: no vision frame for 0.5 s, the page silent for 1 s, our robot not seen for 0.5 s",
     "the step is stopped as for another robot; all motion holds until the information is back and clear for clear s; the step is repeated",
     "after info-alert s: 'no vision for N s: the session holds'; information that comes and goes: after info-alert s 'the session has held for N s ...' naming the causes; reminders every hold-limit s"),
    ("two or more robots seen as our colour + id (one camera frame, or two cameras farther apart than one robot)",
     "no information: the step is stopped, all motion holds while it lasts", "at once: 'two robots are seen as blue 3: at (x, y) and (x, y), the tools cannot tell which is ours; the session holds'"),
    ("frames from a camera that is not on the list", "at start: the session does not start; later: the camera is listed from then on (its silence is no information), logged once", "at start: 'session not started: frames from camera(s) N ...'"),
    ("the start check", "the alert 'session starting: ...' is sent; the session starts only when SOLO_NOTIFY_CMD exits 0", "'session starting: <name>, the -x half, budget N min'"),
    ("the robot the tools drive is not ours (more than 0.25 m from our colour + id in the raw vision)",
     "at start: the session does not start; later: the step is stopped, all motion stops, the session ends", "'the robot the tools are driving is not ours: ...'"),
    ("our half held by another robot for switch-after s (gaps < 1 s), the other half known and empty all that time, the way over clear for clear s",
     "switch halves: every step still to run mirrored (x -> -x, directions 180 - d, side swapped, plans written mirrored), the robot drives over (watched, corridor rule), the interrupted step is repeated there; not within switch-cooldown s of the last; at most max-switches",
     "a quiet notice (SOLO_NOTICE_CMD; else the log): 'moved to the +x half: another robot (yellow 2) stayed in the -x half for 20 s'; at the limit one alert"),
    ("a fetch stopped by another robot", "our robot drives back only when its surroundings or its way back have been clear for clear s; a new fetch waits clear s", "blocked for fetch-wait s: 'our robot is outside our half ...'"),
    ("ball resting outside the home area, on the field, placeable (ball sessions)",
     "a fetch: one place_ball run to the parking spot with a box round robot, ball and spot, only while no other robot is within corridor m of robot->ball and ball->spot (checked 10 times a second during it; a robot in the corridor stops it, a goto back inside, a retry later; at most fetch-tries)",
     "after fetch-wait s blocked: 'ball at (x, y) in the other half, other robots near it (...): fetch it when you can' (reminders every realert s while nothing ball-free is left)"),
    ("ball outside the home area in a goal / corner / against a wall / beyond the margin", "no rescue across the field", "'<where>: put it anywhere inside the field'"),
    ("our robot outside the home area between steps (after a fetch, a strategy step, a stop)",
     "one goto into the home area (0.3 m inside) when its straight way is clear; blocked for fetch-wait s: motion stops", "'<the reason>; our robot at (x, y), outside our half: the session waits' (no information, a robot inside: after info-alert s)"),
    ("a strategy step: the robot within edge-guard m of the home area's open edge", "strategy off at once (the step counts as done; its track is logged)", "-"),
    ("no-ball session: a ball within 0.15 m of the next step's area", "quiet hold up to ball-hold s, then the step moves to the end of the list (twice: skipped); the ball is never touched", "-"),
    ("no-ball session: a ball inside the running step's area within ball-guard m of our robot", "the step is stopped (zeros) and moves to the end of the list", "-"),
]

def dry_run(sess, out=sys.stdout, tables=True):
    p = lambda *a: print(*a, file=out)
    p(f"session {sess.head.get('name', '')}: {len(sess.steps)} steps, budget {round(sess.t_end - sess.t_start)} s; log {sess.log_path}; events {sess.events_path}")
    p(f"notify command: {'set' if os.environ.get('SOLO_NOTIFY_CMD') else 'NOT SET (alerts only printed' + ('; a session with a home area does not start)' if sess.home is not None else ')')}")
    if sess.procs: p(f"commanding programs running now: {busy(sess.procs) or 'none'}")
    if sess.strat: p(f"strategy: {'running' if sess.strat.running() else 'off'}; restart line {'present' if os.path.exists(sess.strat.cmdf) else 'MISSING'} ({sess.strat.cmdf})")
    if sess.home is not None:
        p(f"home area {','.join(f'{v:g}' for v in sess.home)} (open edges: {', '.join(k for k, v in open_edges(sess.home).items() if v) or 'none'}); "
          f"ball {'IGNORED (no-ball session: no placement, fetch or ball alert)' if sess.no_ball else f'fetched to the parking spot {sess.park}'}; "
          f"corridor {sess.corridor} m, fetch-wait {sess.fetch_wait:.0f} s; intruder distance {sess.intrude_dist} m, clear {sess.clear_s:.0f} s, hold-limit {sess.hold_limit:.0f} s")
        p(f"half {half_name(sess.home)}" + (" (--home auto: chosen at start)" if sess.home_auto else "") + "; switch of halves: " +
          (f"after {sess.switch_after:.0f} s, cooldown {sess.switch_cooldown:.0f} s, at most {sess.max_switches}" if sess.switch_after else "off"))
        p(f"start check: would send the alert {sess.start_text()!r} through SOLO_NOTIFY_CMD (not sent in a dry run); "
          f"the session starts only when the command exits 0")
    elif sess.no_ball: p("no-ball session (no placement, fetch or ball alert); no home area")
    refused = [st for st in sess.steps if st.get("_refused")]
    p(f"refused at load: {len(refused)}" + "".join(f"\n    step {st.get('n')} ({st.get('label', st['kind'])}): REFUSED: {st['_refused']}" for st in refused))
    for st in sess.steps:
        k = st["kind"]; needs = sess.needs(st)
        flags = f"needs_ball={needs} on_fail={st.get('on_fail', 'skip')} limit={sess.timeout_of(st):.0f}s" + (f" group={st['group']}" if st.get("group") else "") + (" FINAL" if st.get("final") else "")
        if k == "marker": what = f"marker {expand(st['text'], sess.vs)!r} {st.get('data', '')}"
        elif k == "wait": what = f"wait {st['seconds']} s"
        elif k == "strategy":
            what = f"strategy on {st.get('seconds')} s, then off" if st.get("on", True) else "strategy off"
            prm = dict(st.get("params") or {}); prm.update({"side": st["side"]} if st.get("side") else {})
            if prm and st.get("on", True):
                what += " with " + " ".join(f"-p {a}:={str(b).lower() if isinstance(b, bool) else b}" for a, b in prm.items())
                try: what += "\n        node line: " + " ".join(with_params(json.load(open(sess.strat.cmdf)), prm))
                except Exception: what += "\n        node line: the saved line (solo.cmd) with these parameters replaced or added"
            if st.get("end_ball_out"): what += f"; ends 0.5 s after the ball leaves {st['end_ball_out']}"
        else:
            env = sess.env_for(st); keys = sorted(set(sess.head.get("env", {})) | set(st.get("env", {})))
            what = " ".join(f"{v}={env.get(v, '')!r}" for v in keys) + " " + " ".join(sess.command(st))
            if k == "place": what += f"\n        retry: SOLO_BOX={sess.env_for(st, retry=True).get('SOLO_BOX', '')!r} ... --carry 0.25"
        p(f"  {st.get('n'):>3} [{k}] {st.get('label', '')}  ({flags})\n        {what}")
        if st.get("keep_ball_out"): p(f"        ball kept 0.15 m clear of {st['keep_ball_out']}" + (f", parked at {st['park']} if not" if st.get("park") else ""))
        if st.get("_boxes") and sess.home is not None: p(f"        area: {'; '.join(','.join(f'{v:g}' for v in r) for r in st['_boxes'][:4])}{' ...' if len(st['_boxes']) > 4 else ''}")
        if st.get("_refused"): p(f"        REFUSED: {st['_refused']}")
        if st.get("_unchecked"): p(f"        {st['_unchecked']}")
    if not tables: return
    p("\nsituation -> what the supervisor does -> what the person receives")
    for a, b, c in DECISIONS + (HOME_DECISIONS if sess.home is not None or sess.no_ball else []): p(f"  {a}\n      -> {b}\n      -> {c}")


def main():
    args = sys.argv[1:]
    def opt(name, default):
        if name in args:
            i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
        return default
    budget = float(opt("--budget", "0")); hq = opt("--hq", "http://127.0.0.1:8700")
    kw = dict(ball_lost=float(opt("--ball-lost", "5")), robot_lost=float(opt("--robot-lost", "2")), robot_wait=float(opt("--robot-wait", "10")),
              busy_wait=float(opt("--busy-wait", "20")), realert=float(opt("--realert", "180")))
    hk = dict(corridor=float(opt("--corridor", "0.7")), fetch_wait=float(opt("--fetch-wait", "60")), intrude_dist=float(opt("--intrude-dist", "1.0")),
              clear_s=float(opt("--clear", "5")), hold_limit=float(opt("--hold-limit", "120")), ball_hold=float(opt("--ball-hold", "30")))
    home = opt("--home", None); park = opt("--park", None)
    our_id = opt("--our-id", os.environ.get("SOLO_OUR_ID") or None); hk["fetch_limit"] = float(opt("--fetch-limit", "1.0"))
    hk.update(info_alert=float(opt("--info-alert", "10")), alert_dead=float(opt("--alert-dead", "300")), switch_after=float(opt("--switch-after", "20")),
              switch_cooldown=float(opt("--switch-cooldown", "60")), max_switches=int(opt("--max-switches", "4")),
              settle=float(opt("--settle", "1.5")), max_speed=float(opt("--max-speed", "2.0")), reappear=float(opt("--reappear", "0.15")),
              probes=int(opt("--probes", "6")), probe_rate=float(opt("--probe-rate", "1.5")), probe_time=float(opt("--probe-time", "0.7")),
              clear_own=float(opt("--clear-own", "1.5")))
    vg, _, vp = opt("--vision", f"{VISION_GROUP}:{VISION_PORT}").rpartition(":"); viface = opt("--vision-iface", os.environ.get("SOLO_IFACE", "0.0.0.0"))
    vmirror = "--vision-mirror" in args; args = [a for a in args if a != "--vision-mirror"]
    ncam = opt("--cameras", os.environ.get("SOLO_CAMERAS") or None); cam_ids = opt("--camera-ids", os.environ.get("SOLO_CAMERA_IDS") or None)
    try:
        cameras = [int(v) for v in cam_ids.split(",") if v.strip()] if cam_ids else (list(range(int(ncam))) if ncam else None)
    except ValueError: sys.exit(f"--cameras N (a count) or --camera-ids 0,1,... (a list): got {ncam or cam_ids!r}")
    if cameras is not None and not cameras: sys.exit("--cameras / --camera-ids: at least one camera")
    if park: hk["park"] = [float(v) for v in park.split(",")]
    if "--no-ball" in args: hk["no_ball"] = True; args.remove("--no-ball")
    sim = opt("--sim", None)
    dry = "--dry-run" in args; args = [a for a in args if a != "--dry-run"]
    extra = {}
    while "--var" in args:
        k, _, v = opt("--var", "").partition("="); extra[k] = v
    if len(args) < 2 or args[0] not in ("blue", "yellow"): sys.exit(__doc__)
    team, path = args[0], args[1]
    head, steps, vs = load_session(path, team, extra)
    budget_s = 60.0 * (budget or float(head.get("budget_min", 10)))
    if home == "auto": hk["home_auto"] = True; home = None                         # the half chosen at start
    hk["home"] = home_rect(home if home is not None else head.get("home", ""))   # --home "" = the whole field
    if hk.get("home_auto") and hk["home"] is None: sys.exit("--home auto needs the session's \"home\" (the half it is written for)")
    kw.update(hk); kw["our_id"] = our_id; kw["cameras"] = cameras
    if hk["home"] is not None and our_id is None:
        sys.exit("a session with a home area needs our robot's vision id: --our-id N (or SOLO_OUR_ID=N)")
    if hk["home"] is not None and cameras is None and sim is None:
        sys.exit("a session with a home area needs the field's cameras: --cameras N (ids 0..N-1) or --camera-ids 0,1,... "
                 "(or SOLO_CAMERAS / SOLO_CAMERA_IDS); a listed camera that does not send is no information")
    if hk["home"] is not None and not os.environ.get("SOLO_NOTIFY_CMD") and not dry and sim is None:
        sys.exit("a session with a home area needs SOLO_NOTIFY_CMD (an executable called as CMD \"message\" \"title\"): a person must be alerted")
    stamp = time.strftime("%Y%m%d_%H%M%S"); dd = data_dir()
    log_path = os.path.join(dd, "auto", f"{stamp}_session.jsonl"); events = os.path.join(dd, time.strftime("events_%Y%m%d.jsonl"))
    procs = Procs()
    if dry:
        if hk["home"] is not None: print(f"vision listener (not opened in a dry run): {vg}:{vp} on {viface}{', y mirrored' if vmirror else ''}; our robot {team} {our_id}; "
                                         f"cameras {cameras if cameras is not None else 'NOT GIVEN (the session would not start)'}")
        else: print("WARNING: no home area: other robots are not watched")
        dry_run(Session(team, steps, head, vs, procs=procs, strategy=Strategy(procs), log_path=log_path, events_path=events, budget_s=budget_s, **kw))
        if hk["home"] is not None:                     # the same session mirrored into the other half
            import tempfile
            pd = tempfile.mkdtemp(prefix="auto_session_dry_mirrored_"); mk = dict(kw, home=mirror_rect(hk["home"]))
            if hk.get("park"): mk["park"] = mirror_pose(hk["park"])
            print(f"\n==== the same session in the {half_name(mk['home'])} half (a switch of halves; mirrored plans in {pd})")
            dry_run(Session(team, [mirror_step(st, pd, vs) for st in steps], mirror_head(head), vs, procs=procs, strategy=Strategy(procs),
                            log_path=log_path, events_path=events, budget_s=budget_s, **mk), tables=False)
        return
    if sim is not None:                                # the whole session on a simulated clock, fakes for every step
        clk = SimClock(); fix = SIM_FIX.get(sim); kw["cameras"] = cameras or [0]      # the simulated world has camera 0
        world = ScriptedWorld(clk, robot=(0.0, 0.6, 0.0), ball=(0.3, -0.9), timeline=[(clk.t + fix[0], fix[1])] if fix else ())
        procs = SimProcs(); sess = Session(team, steps, head, vs, world=world, runner=SimRunner(clk, world, SIM[sim]), notifier=SimNotifier(echo=True),
                                           procs=procs, strategy=SimStrategy(procs), clock=clk, sleep=clk.sleep, budget_s=budget_s,
                                           log_path=os.path.join(dd, "auto", f"{stamp}_sim_{sim}.jsonl"), **kw)
        sess.run(); print(f"simulated session log {sess.log_path}"); return
    vision, verr = open_vision(vg, int(vp), viface, vmirror, cameras) if hk["home"] is not None else (None, None)
    if verr: print(f"refused to start: other robots cannot be watched: {verr}", flush=True); sys.exit(1)
    sess = Session(team, steps, head, vs, world=PageWorld(hq, vision=vision), runner=Runner(), notifier=Notifier(), procs=procs, strategy=Strategy(procs),
                   log_path=log_path, events_path=events, budget_s=budget_s, python=os.environ.get("SOLO_PYTHON", "python3"), **kw)
    signal.signal(signal.SIGINT, sess.request_stop); signal.signal(signal.SIGTERM, sess.request_stop)
    time.sleep(1.5)                                    # the first /state polls
    if sess.world.snapshot()["page_age"] > 5: print(f"the field page does not answer at {hq}/state: start hq.py first", flush=True); sys.exit(1)
    print(f"session log {log_path}", flush=True)
    sess.run()
    if sess.stop_why and sess.stop_why.startswith("refused to start"): sys.exit(1)


if __name__ == "__main__":
    main()
