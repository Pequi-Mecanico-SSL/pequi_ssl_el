# Speed tests on the 3-wheel robot (ssl2), 2026-11-26

What limits how fast the robot can drive, measured on the ground with the
overhead camera, and what we learned along the way. Robot: ssl2 (Orange Pi),
`ROBOT_CONFIG=three_wheel`, ESC firmware 8, bridge current limit 3 A, 24 V.

## Results

The kinematics node caps the fastest wheel at `max_wheel_rad_s` (scaling the
whole twist, so the direction is kept); the default is now 80 rad/s. Each run
drove the robot through `cmd_vel` with vision feedback: 6 legs back and forth
along the field x axis (robot facing +x, braking into each end) and a 4 s spin
in place each way, always commanding more than the cap allows, while logging
vision, wheel telemetry and ESC diagnostics (temperature, Uq, bus voltage).
Runs stopped automatically on leaving a safe area, losing vision, an ESC fault
or a board above 75 °C.

| cap rad/s | straight peak | spin | wheels in spin | spin Iq m0/m1/m2 | Uq max | board max |
|---|---|---|---|---|---|---|
| 40 | 1.02 m/s | ±11.2 rad/s | 38 rad/s | 0.69/0.53/0.84 A | 1.7 V | 57 °C |
| 50 | 1.23 m/s | ±14.2 rad/s | 48 rad/s | 0.58/0.46/1.30 A | 1.9 V | 56 °C |
| 60 | 1.35 m/s | ±17.2 rad/s | 58 rad/s | 0.47/0.39/1.52 A | 2.2 V | 56 °C |
| 80 | 1.26 m/s | ±23.5 rad/s | 78 rad/s | 0.39/0.31/1.59 A | 2.7 V | 56 °C |

- **Current** reaches the 3 A limit only while the speed changes; cruising
  and steady spinning stay well below it.
- **Voltage** is far from a limit: Uq at 80 rad/s is 2.7 V of the ~14 V a
  24 V bus can give.
- **Temperature** did not move in 30 s runs (idle is 53–56 °C on the NTC). A
  long run is still needed to find the thermal limit. The firmware has no
  thermal protection (a false overvoltage fault trips near 125 °C), so tests
  stop at 75 °C.
- Vision keeps tracking the robot at 23.5 rad/s.

## Straight line: grip, not current

The 1.05 m lane available in the lab caps the straight-line test near
1.3 m/s, so above ~60 rad/s only the spin reaches the wheel cap. Inside that
lane, acceleration is limited by grip:

| direction | acceleration | wheel speed ahead of the ground | braking |
|---|---|---|---|
| forward (front wheels leading) | 1.5 m/s² | up to 0.77 m/s | 1.1 m/s² |
| backward | 2.0 m/s² | up to 0.39 m/s | 2.3–2.6 m/s² |

- The wheels slip while accelerating (wheel odometry ahead of vision; camera
  latency explains < 0.1 m/s of it), and the front wheels' current jumps
  between 3 A and ~0.5 A instead of sitting at the limit. A higher current
  limit would mostly spin the wheels.
- Neither direction reaches the kinematics' 3 m/s² limit. Forward is worse:
  accelerating forward moves weight to the back wheel and off the two front
  wheels, which do all the pushing along x (the centre of mass is also behind
  the centre, see below).
- The robot follows speed changes ~0.2 s late, so anything that brakes into
  a target (a vision position controller) must start early.

## Spinning: the wheels fight because the rotation centre is off

In a spin the back wheel (m2) drove the robot alone while one front wheel
braked (which one depends on the direction), and m2's current grew with speed
(1.3 A at 16 rad/s). Splitting each wheel's current into a part that follows
the spin direction and a part that does not:

- The direction-independent part is a forward push of the front wheels that
  grows with speed (0.3 N at 4 rad/s, 0.85 N at 8, 1.5 N at 12): centripetal
  force from spinning about a point that is not the centre of mass.
- Spinning about a point 10 mm behind and 1.5 mm left of the kinematics
  origin (the origin comes from the camera calibration, close to the tag
  centre) removes it at every speed: currents 0.25/0.43/0.15 A from 4 to
  16 rad/s, no push. 5 mm back halves it, 5 mm forward makes it worse.
- The equivalent kinematics is `robot_radius: [0.0798, 0.0804, 0.0722]` (same
  wheel angles and radii). It is **not** in `three_wheel.yaml`: with it,
  `cmd_vel` describes the motion of that point instead of the tag, and the
  tag draws a ~7.5 mm circle while spinning, which vision controllers would
  have to know about.
- Ruled out: the speed loops (wheels within 0.1 rad/s of their targets),
  the current loops (Iq follows its reference, Id ≈ 0), the vision position
  hold (it asked for ~0.01 m/s) and controller history (same currents after
  forward, backward or reset approaches).

## Parked: integrators kept the wheels pushing

Omni rollers hold sideways force by friction, so a parked robot's three speed
loops can all sit at zero error while pushing against each other with what
the last move left in their integrators: ±0.4–0.9 A per wheel, unchanged for
as long as the robot stays parked. ESC firmware commit 820587b
(stm32-motor-controller, `fw8-spi-encoder`) drains the integrator when the
target is 0 and the wheel is still (parameter `drain`, time constant 1 s):
the currents fall below 0.1 A within 3 s and the robot moves less than 1 mm.

## Vision in the lab

- Camera API on the vision PC: `http://<vision pc>:8080/` (picture with
  `/api/cameras/0/frame.jpg?overlay=true`, detections, pixel↔field, full
  calibration at `/api/cameras/0`). Only the SSL-Vision multicast
  224.5.23.2:10006 (~80 Hz) is published, no ROS topics.
- The calibration is right-handed (field-to-camera rotation ≈ diag(1, −1, −1)),
  so run `ssl_vision_connector` with `mirror:=false`; the launch default
  `true` flips y and θ.
- A machine with several network interfaces joins the multicast group on its
  default-route interface: when that is not the Wi-Fi, add
  `ip route add 224.5.23.0/24 dev wlan0`.

## Open

- Long run at 80 rad/s for the thermal limit (m2 is the hottest candidate
  while the spin fight remains).
- Spin-only runs above 80 rad/s.
- The same program with a 4–5 A current limit, to confirm acceleration is
  grip-limited.
- Moving the kinematics origin to the centre of mass (above), and on-ground
  tuning of the ESC velocity loop.
