# Robot events: impact, skid, stall, push

The `robot_events` node (runs on the Pi, started by `launch/robot.yaml`)
watches the sensors and publishes what happens to the robot on `events`
(one JSON per event) and `events/status` (what is active now, 10 Hz).

```bash
ros2 topic echo /events
```

| Event | Signal | Detected when |
|---|---|---|
| `impact` linear | accelerometer | horizontal acceleration above `impact_accel` (8 m/s², beyond what the motors produce), after removing the centripetal part; reports peak, direction, and whether the ±4 g range saturated |
| `impact` rotational | gyro | angular acceleration above `impact_ang_accel` (60 rad/s²), e.g. hit on a corner |
| `skid` | odometry residuals | one wheel disagrees with the fused velocity (> 3 σ) |
| `stall` | current + setpoints + wheels | motion commanded, ≥ 2 motors at current limit, velocity not approaching the command (pinned on a wall, blocked or shoved off course by a robot); a robot shoved while commanded still is a `push`, not a stall |
| `push` | wheels + current | commanded not to translate for 0.5 s, but the robot moves or the motors fight back (rotation commands, such as `velocity_controller` damping a shove-induced spin, don't count) |

Inputs: `imu`, `wheels/state` (speed + current in `effort`), `wheels/cmd`,
`odom/wheel_residuals` (from `odometry`, see [ODOMETRY.md](ODOMETRY.md)).

**Which wheel skids is only certain up to a pair.** Our wheels at −60/60/−135/135°
form two nearly opposite pairs (0/3 and 1/2, 195° apart), whose skid
signatures are almost identical. The event gives `wheel` (best guess) and
`candidates` (the pair); the skid itself is reliable.

## See it work without the simulator

Synthetic 30 s scenario with a wall hit, being pinned, a wheel skid, a
side hit and a push, plus hard driving and spins to catch false alarms:

```bash
docker compose run control ros2 run robot_control events_demo
```

```
true event     at    detected   delay    ended   details
impact       8.00        8.00     0ms      n/a   kind=linear accel=39.5 saturated=True direction_deg=-179.3
stall        8.00        8.18   180ms     9.76   wheels=[0, 1, 2, 3]
skid        12.00       12.04    40ms    13.12   wheel=2 candidates=[1, 2]
impact      15.00       15.00     0ms      n/a   kind=rotational ang_accel=315.0
push        19.00       19.12   120ms    20.42
false alarms: 0
```

Over 20 random runs: every event detected, 0 false alarms. Tests:

```bash
docker compose run control bash -c "cd /ssl_ws/src/robot_control && python3 -m pytest -p no:cacheprovider test -v"
```

Through ROS (on the PC, never on the robot: the fake robot publishes the
robot's sensor topics); it logs `TRUE EVENT ...` as each scripted event
happens:

```bash
docker compose run control ros2 run robot_control odometry
docker compose run control ros2 run robot_control robot_events
docker compose run control ros2 run robot_control fake_robot --ros-args -p scenario:=events
```

The synthetic accelerometer and motor-current models are guesses, not
measurements of our hardware. They prove the logic; the thresholds must be
checked on the real robot.

## Check it on the real robot

Rebuild the robot image once: `docker compose build robot`. Then:

1. **Accelerometer axes** (imu.py flips the gyro axes but not the
   accelerometer's, so do not assume they match): with the robot on the
   field, `ros2 topic echo /imu --field linear_acceleration` and drive
   forward with a sharp start. The axis that jumps positive is forward.
   Repeat to the left. Set `accel_axes` for `robot_events` in `robot.yaml`,
   e.g. `['-y', '+x']`. Impact detection works without this (it uses the
   magnitude), but `direction_deg` and the centripetal correction need it.
2. **No false alarms**: drive hard with teleop for a few minutes, spins
   included. `/events` should stay quiet. Impacts while driving normally:
   raise `impact_accel` (vibration); stalls on hard starts: raise
   `stall_current_frac` or lower `stall_max_progress`.
3. **Each event**, with `ros2 topic echo /events` open:
   * drive slowly into a wall: `impact` (direction ≈ ±180° when hitting
     front-first), then keep pushing forward: `stall`
   * hit a corner of the robot by hand: `impact` rotational
   * drive with one wheel on tape or lifted slightly: `skid`, candidates
     containing that wheel
   * robot idle and enabled, push it by hand: `push`
4. **Record it** so thresholds can be tuned offline:
   `ros2 bag record /imu /wheels/state /wheels/cmd /odom /odom/wheel_residuals /events`

`current_limit` must match `can_bridge`'s `current_limit_a` (1.3 A).

**Stall during shoving fights.** When two robots shove each other, a robot
alternates between being blocked and moving off course, and `stall` follows
that: in the simulator it switched on and off 5 times in 7 s of shoving
(intervals of 0.3-0.8 s). Raising `stall_off_s` merges the gaps but makes the
stall sticky: the next hard maneuver after the robot frees itself saturates
the current too and keeps the stall alive (0.5 s made stalls end up to 1.1 s
late). Lengthening `stall_window_s` causes false stalls at every hard start
from rest. Tune `stall_on_s` / `stall_off_s` / `stall_window_s` on recordings
of the real robot.
