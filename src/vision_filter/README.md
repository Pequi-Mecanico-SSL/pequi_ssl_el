# vision_filter

One extended Kalman filter per robot over SSL-Vision. Smooths poses, rejects
ghost detections, bridges short vision dropouts, compensates vision latency
(outputs are predicted to *now*), and estimates body-frame velocity, which raw
vision does not give at all.

```
ssl-vision --UDP--> vision_connector --/vision/stamped/*--> vision_filter --/vision/filtered/*-->
                                     \-/vision/poses/* (raw, unchanged)     ^
                                           /<team>/robot<i>/cmd_vel --------/ (optional prior)
```

| Topic | Type | Content |
|---|---|---|
| `/vision/filtered/{blue,yellow}/robot<i>` | `geometry_msgs/Pose2D` | drop-in replacement for `/vision/poses/*` |
| `/vision/filtered/odom/{blue,yellow}/robot<i>` | `nav_msgs/Odometry` | pose (field frame) + velocity (body frame: x forward, y left) + covariance |
| `/vision/stamped/{blue,yellow}/robot<i>` | `geometry_msgs/PoseStamped` | raw pose stamped with capture time (published by the connector) |

Nothing consumes the filtered topics yet: `rl_strategy` still reads raw
`/vision/poses/*`. Switch it over once the checks below pass.

## Run

```bash
docker compose build vision_connector vision_filter
docker compose up vision_connector vision_filter
```

## How to know it works

### 1. Unit tests (no ROS, a few seconds)

Synthetic trajectories with known truth: smoothing, velocity accuracy,
ghost rejection, robot picked up and moved, 150 ms dropout, out-of-order
packets, heading wrap at +-180 deg, cmd_vel prior, filter consistency.

```bash
docker compose run vision_filter bash -c "cd /ssl_ws/src/vision_filter && python3 -m pytest test -v"
```

### 2. Full pipeline with fake vision (no camera, no simulator)

`fake_vision` sends ssl-vision packets with noise, 20 ms latency, packet loss,
blackouts and ghosts, and publishes the true state. Uses port 10020, so it
never mixes with the real camera on 10006.

```bash
# terminal 1: connector listening to the fake source
docker compose run vision_connector ros2 launch ssl_vision_connector \
    ssl_vision_connector_launch.yaml vision_port:=10020 mirror:=false
# terminal 2: filter
docker compose up vision_filter
# terminal 3: fake source
docker compose run vision_filter ros2 run vision_filter fake_vision
# terminal 4: measure (blue/robot0 moves, yellow/robot0 stands still)
docker compose run vision_filter ros2 run vision_filter vision_check \
    --ros-args -p robot:=blue/robot0 -p seconds:=15.0
```

Expected (reference run):

```
                        raw               filtered
  position  median    19.5 mm          8.1 mm
            RMS      115.7 mm         10.5 mm
  heading   median    1.37 deg        1.09 deg
  velocity  median   0.493 m/s       0.092 m/s
RESULT: PASS - filtered beats raw
```

### 3. Real field

* **Robot standing still** (measures your camera noise):
  `vision_check --ros-args -p robot:=blue/robot0 -p seconds:=20.0` with no
  fake_vision running. Expect filtered speed about 0 and jitter no worse than
  raw. It prints `sigma_xy`/`sigma_theta` values for your camera; pass them
  to the launch file.
* **Filter log**, every 5 s: `rejected` should stay near 0 in normal play,
  `resets` should only increase when a robot is picked up and moved,
  and `mean_latency` shows how old vision is when it reaches the filter.
* **Plot raw vs filtered** while pushing or driving a robot
  (`rqt_plot`, or PlotJuggler if installed):
  `/vision/poses/blue/robot0/x` vs `/vision/filtered/blue/robot0/x`. The
  filtered curve should be smooth and slightly ahead (latency compensation),
  never behind.
* **Velocity sanity check**: drive straight at a known `cmd_vel` (e.g. 0.5 m/s)
  and compare `/vision/filtered/odom/blue/robot0` `twist.linear.x`.

## Tuning

| Parameter | Default | Meaning |
|---|---|---|
| `sigma_xy`, `sigma_theta` | 0.005 m, 0.03 rad | vision noise; measure with `vision_check` on a still robot |
| `accel_noise` | 0.5 | higher = follows sudden maneuvers faster, noisier velocity |
| `ang_accel_noise` | 3.0 | same for rotation |
| `use_cmd_vel`, `team`, `cmd_tau` | true, blue, 0.15 s | our robots' commands as a motion prior (much faster velocity response after a command change); `cmd_tau` is how quickly the real robot reaches a commanded velocity. **Caution:** it assumes the robot follows its commands. In the simulator it cut velocity error by about 40% while driving freely, but when a robot was blocked (pushing an opponent or a wall while still commanded to move) position error doubled (14-16 mm vs 6.5 mm) and velocity error went from 0.07 to 0.3 m/s. Turn it off with `use_cmd_vel:=false` if your robots spend a lot of time in contact. |
| `gate`, `max_rejects` | 16.27, 5 | outlier threshold; consecutive outliers before snapping to the new position |
