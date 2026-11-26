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

Unattended test sessions: `auto_session.py <team> tools/solo/sessions/short_10.jsonl --budget 10` runs the steps of
a session file (placements, tests, series, strategy on/off) with the field page running and locked, and the strategy
started with `run_real.sh` and switched off; `--dry-run` prints every command, `--sim ok` runs it offline. The
session files' `P` (default `tools/solo/sessions/plans`, the plans shipped with them, or `--var P=DIR`) is the folder
of their series plans, `OUT` (default `$SOLO_DATA/sessions`, or `--var OUT=DIR`) the folder of their logs and results.
Each session writes its step log to `$SOLO_DATA/auto/` and its markers to `$SOLO_DATA/events_<date>.jsonl`: one line
per marker with its text and `"data": "use"` or `"skip"`, so that the recordings taken at the same time (vision,
robot) can be cut into the stretches that are test data and the ones that are not.
With a home area the supervisor keeps our robot's last unambiguous position with the time of its detection; a
robot of our colour and id seen farther from it than 0.15 m + `--max-speed` (2.0 m/s) x the time our robot may have
been moving since then (every interval a motion step or the strategy node ran, each extended by `--settle` s, 1.5,
after its stop for the reaction delay and braking) is "which robot is ours was lost": the session holds until our
robot is seen back at that position. A camera dropout or a stop at full speed stays inside that allowance; a robot
standing still allows 0.15 m. Another robot counts from its second consecutive frame of one camera (or after
0.1 s without a newer frame of that camera), so a single false detection stops nothing; the robot the tools drive
is compared with ours only while ours was detected within the last 0.1 s.

`hq.py`, `auto_session.py` and `place_ball.py` share `procs.py`: the list of programs that command the robot (the
strategy node and every tool here that publishes velocity commands), the test for a running instance of one, and the
check of the strategy node's saved command line before the field page or a session starts it again.

`frame_check.py <team>` (field frame and robot id) needs the vision connector running and no strategy node: it
drives the robot about 0.3 m.

Tests (no ROS, no robot, no network): `uv run --with pytest --with numpy pytest -q tools/solo/tests/`.
