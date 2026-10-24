# Simulator tests

End-to-end tests of the vision filter, odometry, robot events and
velocity controller on rSim, compared against the simulator's ground truth.
No camera, robot or screen needed.

The simulator has no camera, IMU, wheel encoders or motor current, so
`sim_harness.py` creates them from the true state:

| Real system | Here |
|---|---|
| ssl-vision | SSL-Vision UDP packets from the true poses: 5 mm / 0.03 rad noise, 20 ms latency, 3% loss, 0.5% ghosts, optional blackouts |
| gyro + accelerometer (`imu`) | true rotation rate + bias (0.02 rad/s ramping to 0.05 over 60 s, a warm-up assumption) + noise; body acceleration + noise, clipped at 4 g |
| ESC telemetry (`wheels/state`) | wheel speeds of our wheel geometry from the true motion (no slip) + 0.3 rad/s noise; current from a simple ESC model |

Blue robot 0 runs the robot-side nodes as on the Pi (`velocity_controller`,
`kinematics`, `odometry`, `robot_events`); everything runs on simulated
time (`/clock`), so results do not depend on machine load.

## Run

Build the x86 image once, then run with plain `docker run`. The scripts
publish `/blue/robot0/cmd_vel`, which a real robot on the same network would
obey, so they keep everything on the container's loopback:
`ROS_LOCALHOST_ONLY=1` for ROS and unicast to `127.0.0.1` for the fake camera.
This also keeps two test runs on one machine from mixing.

```bash
docker compose build control
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v $PWD/tools/sim_test:/sim_test pequi_ssl_el-control \
    bash /sim_test/run_all.sh          # ~12 min, writes tools/sim_test/results/summary.txt
```

`--user` and `HOME` keep the results owned by you instead of root.

Single scenarios: `run_match.sh` (3v3 rl_strategy match, 2 min),
`run_straight.sh <use_odom> <bias_end> <out.pkl>` (heading hold on a
rectangle with spins), `run_events.sh <out.pkl>` (wall hit, push, side hit,
injected skid). Analyses: `analyze.py`, `vision_ab.py`, `straight_analysis.py`,
`events_analysis.py`, each taking the recorded `.pkl`.

## Simulator facts that matter for testing

- `get_state()` computes velocities from the position change since the
  *previous* call: call it once per step (a second call returns zeros).
- Velocity commands are robot frame; reported velocities are field frame.
- Its published wheel speeds are the commanded ones, not measured.
- Collisions occasionally rotate robots by tens of degrees in one 16 ms step,
  and once ejected the ball 8-20 m off the field; with no referee, play does
  not resume.
