# tools/solo: one robot on the real field

Start script, field page, ball placement, recorders and characterization tools for the one-robot setup
(`src/solo_strategy`). Each tool's docstring or header has its full usage.

## Running from this branch

Build the workspace once (`colcon build` at the repository root), then source `tools/solo/env.sh` in every shell
that runs a tool. It sources ROS 2 Humble and `install/setup.bash`, adds `../ssl-rules/src` (the rules package,
checked out next to this repository) to `PYTHONPATH`, and sets the ROS domain.

Environment variables:

| Variable | Needed | Meaning |
|---|---|---|
| `SOLO_SIM_DIR` | required by `run_real.sh start` | checkout of the strategy simulator package (import `sslenv`) with its `.venv`; the policy server runs from it |
| `ROBOT` | required by `robot_stack.sh` | `user@address` of the robot's computer |
| `SSHPASS` | required by `robot_stack.sh` | the robot login's password, read by `sshpass -e` |
| `ROBOT_CONFIG` | optional | robot_control configuration for `robot_stack.sh` (default `three_wheel`) |
| `SOLO_DATA` | optional | folder for recordings, logs, the vision map and the field page's event log (default `<repo>/log/solo`, created if missing) |
| `SOLO_DOMAIN` | optional | ROS domain on both sides, laptop (`env.sh`) and robot (`robot_stack.sh`) (default 77) |
| `SOLO_IFACE` | optional | this computer's address on the robot's network: DDS uses only that address; the field page joins the game controller's multicast on it, `auto_session.py` the vision multicast |
| `SOLO_ROBOT_INDEX` | optional | our robot's index among its colour in the vision connector's topics (default 0) |
| `SOLO_OUR_ID` | required by `auto_session.py` with a home area (or `--our-id`) | our robot's vision id: the supervisor tells our robot from every other one by colour and this id in the raw vision |
| `SOLO_CAMERAS`, `SOLO_CAMERA_IDS` | required by `auto_session.py` with a home area (or `--cameras N` / `--camera-ids 0,1,...`) | the field's cameras: a count N (ids 0..N-1) or a list of camera ids; a listed camera that has not sent a detection frame for 0.5 s, or never, is no information; a camera that sends without being listed refuses the start, and during a session is listed from its first frame on |
| `SOLO_NOTICE_CMD` | optional | an executable for quiet notices of `auto_session.py` (a switch of halves), called as `CMD "message" "title"`; unset: a log line |
| `SOLO_OWN_COUNT`, `SOLO_OTHER_COUNT` | optional | robots per team the vision connector reports, ours and the other team's (defaults 1 and 3) |
| `SOLO_VISION_MAP_MODULE` | optional | path of `vision_map.py` for the strategy node's recovery (`env.sh` sets it to this repository's copy) |
| `SOLO_VISION_PB_DIR` | optional | folder with generated SSL vision protobuf modules for `vision_map.py`, when `ssl_vision_connector` is not importable |
| `SOLO_RUN_DIR` | optional | where the field page finds the strategy node's saved command line (default `/tmp/solo_run`, which `run_real.sh` uses) |
| `SOLO_OVERLAY` | optional | a second `setup.bash` (a pinned build of `solo_strategy`) sourced on top by `env.sh` |
| `SOLO_ARENA`, `SOLO_SRC` | optional | for `run_real.sh`: another simulator source tree, another `src` folder for the policy server |
| `SOLO_CMD_TOPIC`, `SOLO_CMD_RELIABLE` | optional | velocity topic of the test tools (default `/cmd_vel`); set `SOLO_CMD_RELIABLE` for a reliable publisher |
| `SOLO_BOX`, `SOLO_BRAKE`, `SOLO_BRAKE_BOX`, `SOLO_BRAKE_FIRST`, `SOLO_SETTLE`, `SOLO_RAMP` | optional | test area and braking of `characterize.py`, `series.py` and `place_ball.py` (see their docstrings) |
| `SOLO_GOTO_BOX` | optional | `series.py`: the area the robot may cross on its way to a test's start (default: the test's own box, so a robot that starts outside it stops at once); `auto_session.py` sets it for its series steps: with a home area always the home area less 0.25 m at its open edges (`characterize.py goto` stops only when it leaves the box, at up to 0.6 m/s; walls stay), a given value (shell, header or step) clipped to that and refused at load when it does not intersect it, mirrored on a switch of halves; without a home area the given value, else the session header's `SOLO_BOX` |
| `SOLO_EXTRA_GROUPS`, `SOLO_EXTRA_IFACE` | optional | additional multicast groups for `listen.py` (empty by default) |
| `SOLO_NOTIFY_CMD` | required by `auto_session.py` with a home area, else optional | an executable called as `CMD "message" "title"` for its alerts (a failing call is tried once more); unset, alerts are only printed. A session with a home area sends "session starting: ..." in its start check and starts only when the command exits 0; later alerts go out from a queue in the background (retried with backoff, never delaying a stop), and when none has been delivered for `--alert-dead` s (300: from the first failed send or the oldest alert still waiting) no motion step runs until one is; a sending thread that ends is started again once, and ends the channel the second time |
| `SOLO_PYTHON` | optional | the Python `auto_session.py` runs its steps with (default `python3`) |

Usual start sequence:

```bash
export SOLO_SIM_DIR=<simulator checkout>  ROBOT=<user>@<robot address>  SSHPASS=<password>
export SOLO_IFACE=<this computer's address on the robot's network>      # when it has several interfaces
source tools/solo/env.sh
python3 tools/solo/listen.py $(ip -4 -br addr | awk '{print $3}' | cut -d/ -f1)   # vision and game controller packets arrive
tools/solo/robot_stack.sh start                 # the robot's control stack (enables the motors)
tools/solo/run_real.sh start <blue|yellow>      # policy server, vision, referee, strategy node
tools/solo/run_real.sh status
python3 tools/solo/hq.py <blue|yellow>          # field page on http://localhost:8700
tools/solo/run_real.sh stop; tools/solo/robot_stack.sh stop
```

The field page starts locked (view only) and refuses to drive while another program that commands the robot runs
on the same computer, including the strategy node; its STRATEGY button stops and restarts the node. It accepts only
JSON POST requests addressed to its own host and port, so other clients must send
`Content-Type: application/json` and a matching `Host` header.

Status of the sessions on a shared field (a home area, the `left_*` session files): run so far only against the
simulator (start checks, the stop and alert when another robot enters the half, the identity probe after the robot was
moved by hand) and not yet on the real robot. Until a session has run on the real field with a person at the stop
button, run them with a person watching.

Unattended test sessions: `auto_session.py <team> tools/solo/sessions/short_10.jsonl --budget 10` runs the steps of
a session file (placements, tests, series, strategy on/off) with the field page running and locked, and the strategy
started with `run_real.sh` and switched off; `--dry-run` prints every command, `--sim ok` runs it offline. The
session files' `P` (default `tools/solo/sessions/plans`, the plans shipped with them, or `--var P=DIR`) is the folder
of their series plans, `OUT` (default `$SOLO_DATA/sessions`, or `--var OUT=DIR`) the folder of their logs and results.
A series step with a home area gets `SOLO_GOTO_BOX` from the supervisor: the home area less 0.25 m at its open edges (the edges toward the other half), the area the robot may cross on its way to each test's start.
Each session writes its step log to `$SOLO_DATA/auto/` and its markers to `$SOLO_DATA/events_<date>.jsonl`: one line
per marker with its text and `"data": "use"` or `"skip"`, so that the recordings taken at the same time (vision,
robot) can be cut into the stretches that are test data and the ones that are not.
With a home area the supervisor keeps our robot's last unambiguous position (the anchor) with the time of its
detection. A single robot of our colour and id seen after it is ours when ours was unseen for less than 0.5 s and
it is within 0.15 m + `--max-speed` (2.0 m/s) x the time since the anchor's detection (0.15 m when no motion step or
strategy node ran then, or stopped within `--settle` s, 1.5, before: the robot stood; the settle window says only
that it may move, it adds no distance), or when ours was unseen for 0.5 s or more, it reappears within `--reappear`
m (0.15) of the anchor and no second robot of our colour and id was seen meanwhile (on the field's camera most
dropouts end within 0.15 m; a farther reappearance is settled by the probe). The rule runs on the camera frames, not
on the supervisor's polls: the vision reader keeps every sighting of our colour and id of the last 10 s, and each
snapshot moves the anchor forward along them, frame to frame, with the same bound (0.15 m + `--max-speed` x the time
between the sightings), so the time the supervisor takes no snapshot (a child being stopped, a sleep, the start alert)
is not counted as our robot unseen; "unseen for 0.5 s" means no sighting continuous with the track for that long.
Anything else is "which robot is ours
was lost": the session holds, alerts at once (adding, while probes are left, that the session will try to recognise
it by a short turn on the spot, when the hold is the one that runs the probe and the candidate, or with none in view
the anchor, stands where a probe is accepted; a new loss goes out whatever the backoff of the last), and clears when
ours is seen back at the anchor for `--clear` s, or through the identity probe (a failed probe leaves it lost at the
same anchor). When the lost message went out in any alert, the recovery is an alert too ("our robot recognised by its
turn at (x, y): the session resumes", "our robot seen again at its place (x, y): the session resumes", or "...; the
session still holds: <cause>" while another robot or missing vision keeps the hold), with any held alert about the
loss folded into it: a person on the way to the robot must know that it will move again.

The identity probe: while identity is lost and exactly one robot of our colour and id is in view, standing still,
with no other robot in our half or within 1 m of it, for `--clear` s, the supervisor runs two turns on the spot in a
random order of signs (`characterize.py <team> spin <+-rate> <time>`, `--probe-rate` 1.5 rad/s, `--probe-time` 0.7
s, 0.5 s apart), each watched. If the candidate's heading follows both turns (0.4 to 1.6 times the commanded angle,
in the commanded direction) and it moves less than 0.10 m, it is ours: identity is anchored on it (alerted as above, or a
quiet notice when no lost alert went out). Otherwise one alert says that the robot did not follow a commanded turn,
the session holds, and that place is not probed again for 60 s. A probe child that did not complete its turn (a
non-zero exit, a result line marked interrupted or aborted, e.g. "robot not in vision") gives no verdict: no alert,
no ban, the reason in the log (`probe_verdict`), the next probe not before 10 s. A child that ends with a Python
exception (its last line, e.g. "ValueError: ...") twice in a row with the same message is a set-up fault: probing
ends for the session, with one alert naming the message. The probe child's braking settings are its own, whatever
the header or the step set: `SOLO_BRAKE` 1.5, `SOLO_RAMP` 0, `SOLO_BRAKE_FIRST` and `SOLO_BRAKE_BOX` unset,
`SOLO_BOX` a 0.5 m box around the candidate; a `test` step with `SOLO_BRAKE=auto` gets `SOLO_BRAKE_FIRST` (0.8 when
unset), as `series.py` does before any braking is measured. At most `--probes` (6) per session,
those without a verdict included (the alert when none is left says how many gave no verdict and the last reason);
`--probes 0` switches it off. A pure turn on
the spot moves nobody across the field: if the candidate is another team's robot, ours only turns where it stands.
`characterize.py spin` sends only an angular rate, never drives to a start pose, and uses its pose only to stop.

A hold whose only cause was our robot not seen (the vision fresh throughout, no other robot in or near our half)
ends once our robot has been seen again for `--clear-own` s (1.5); every other hold needs `--clear` s, except after
a probe pass, when it may end at once (the probe itself was watched). Another robot
counts from its second sighting by one camera within 0.25 s (any two frames: a robot detected in every other frame
counts), or within 1 s when the second lies within 0.2 m + 2 m/s x the gap of the first (a robot detected only now
and then), and stays counted until its sighting expires (0.5 s); a sighting within 0.2 m of a recent one of another
colour or id counts at once (an id read differently from frame to frame); a single isolated detection is ignored,
and so are two sightings more than 1 s apart. The robot the tools drive is compared with ours only while ours was detected within the last 0.1 s. At
start its heading on the page is compared with ours in the raw view as well: more than 0.5 rad apart (with the
positions agreeing) refuses the start, naming `--vision-mirror`, since a wrong mirror setting flips y and the heading
and lets the positions agree only near y = 0; when either heading is missing the check is skipped (logged). A "not
ours" text whose two positions agree with y flipped says so and names `--vision-mirror`. If the
alert sender's thread has ended, "session ended" is sent once directly.

`hq.py`, `auto_session.py` and `place_ball.py` share `procs.py`: the list of programs that command the robot (the
strategy node and every tool here that publishes velocity commands), the test for a running instance of one, and the
check of the strategy node's saved command line before the field page or a session starts it again.

`frame_check.py <team>` (field frame and robot id) needs the vision connector running and no strategy node: it
drives the robot about 0.3 m.

Tests (no ROS, no robot, no network): `uv run --with pytest --with numpy pytest -q tools/solo/tests/`.
