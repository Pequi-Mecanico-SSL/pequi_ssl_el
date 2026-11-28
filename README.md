Build with:
```
docker compose build bash simulator control
```
And run simulator with:
```
docker compose run simulator
```
To build for the raspberry pi 4, you can build all services:
```
docker compose build bash simulator control ping pong robot
```

To test communication latency, run `docker compose run ping` on the computer and `docker compose run pong` on the raspberry pi.

## Robots on the network

Every robot runs the same code (`docker compose up -d robot`) and differs only in its `.env`:

```
ROBOT_ID=1        # the robot's number
IMU_I2C_BUS=0     # 0 on the Orange Pi 4 Pro, 1 on the Raspberry Pi 4
```

All robots and the computers that command them share ROS domain 77 (`ROS_DOMAIN_ID` in
`docker-compose.yml`). A robot's nodes and topics live in the namespace `/robot<ROBOT_ID>`:

| topic | |
|---|---|
| `/robot1/cmd_vel` | body velocity command (Twist, best effort) |
| `/robot1/odom`, `/robot1/imu`, `/robot1/wheels/state`, `/robot1/cmd_vel_ref` | what the robot reports |
| `/robot1/wheels/cmd`, `/robot1/wheels/estop`, `/robot1/wheels/enable` | bench commands |
| `/diagnostics` | shared by all robots; status names start with the robot (`robot1/drive: bus`) |

The full list is in `src/robot_control/robot_control/drive.py`. `tools/diagwatch` shows the
diagnostics of every robot it hears, one block each. Vision topics (`/vision/poses/...`) are not
per robot and stay where they are.
