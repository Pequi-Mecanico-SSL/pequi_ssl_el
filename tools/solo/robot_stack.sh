#!/bin/bash
# The robot's control stack, run from the laptop over ssh: ROBOT_LAUNCH names the launch file of robot_control,
# robot.yaml (default: can_bridge, imu, velocity_controller, kinematics, teleop) or drive.yaml (drive, teleop).
#   ROBOT=<user>@<robot address> SSHPASS=<robot password> tools/solo/robot_stack.sh start | stop | status | logs [N] | save
# start ENABLES THE MOTORS (they hold zero speed until a command arrives). The stack listens on /cmd_vel, ROS domain
# $SOLO_DOMAIN (default 77; the laptop side uses the same through tools/solo/env.sh).
# save copies the container's whole log to $SOLO_DATA/robot_logs/ (SOLO_DATA default: log/solo in this repository):
# do it before the next start, which
# removes the old container and its log.
ROBOT=${ROBOT:?set ROBOT to user@address of the robot}; DOM=${SOLO_DOMAIN:-77}; CFG=${ROBOT_CONFIG:-three_wheel}
LAUNCH=${ROBOT_LAUNCH:-robot.yaml}
R="sshpass -e ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $ROBOT"
[ -n "$SSHPASS" ] || { echo "set SSHPASS to the robot's password"; exit 1; }
export SSHPASS
case "$1" in
start)
  $0 save >/dev/null 2>&1
  $R "docker rm -f ssl_play >/dev/null 2>&1; docker run -d --name ssl_play --privileged --network host --ipc host -e ROS_DOMAIN_ID=$DOM -e IMU_I2C_BUS=0 -e ROBOT_CONFIG=$CFG -v \$HOME/pequi_ssl_el/src:/ssl_ws/src:ro -v /dev:/dev pequi_ssl_el-robot bash -lc 'cd /ssl_ws && rm -rf build install && colcon build --packages-select robot_control && source install/setup.bash && ros2 launch robot_control $LAUNCH' | cut -c1-12"
  echo "started ($LAUNCH); the build takes about 30 s: check with '$0 logs'";;
stop)   $R "docker stop -t 3 ssl_play >/dev/null 2>&1; docker ps --format '{{.Names}}' | grep -c ssl_play" | sed 's/^/ssl_play containers running: /';;
status) $R "docker ps -a --format '{{.Names}} | {{.Status}}' | grep ssl_play; docker logs ssl_play 2>&1 | grep -c 'watchdog tripped' | sed 's/^/watchdog trips in the log: /'; docker logs ssl_play 2>&1 | grep -v joystick | tail -3 | cut -c1-150";;
logs)   $R "docker logs ssl_play 2>&1 | grep -v joystick | tail -${2:-25} | cut -c1-170";;
save)   D=${SOLO_DATA:-$(dirname "$(readlink -f "$0")")/../../log/solo}/robot_logs; mkdir -p $D; F=$D/ssl_play_$(date +%Y%m%d_%H%M%S).log
        $R "docker logs -t ssl_play 2>&1" > $F 2>/dev/null; [ -s $F ] && echo "saved $F ($(wc -l < $F) lines)" || rm -f $F;;
*) echo "usage: SSHPASS=... $0 start | stop | status | logs [N] | save";;
esac
