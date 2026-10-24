# shared helpers for scenario scripts (sourced). Run inside the project image.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=${WS:-/ssl_ws}                       # colcon workspace (project image: /ssl_ws)
OUT=${OUT:-$HERE/results}
# keep all ROS traffic on this container's loopback: these scripts publish
# /blue/robot0/cmd_vel etc., which a real robot (or a second test run) on the
# same network would otherwise receive
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-87}
source $WS/install/setup.bash
export PYTHONUNBUFFERED=1
LOG=$OUT/logs/$SCENARIO; rm -rf $LOG; mkdir -p $LOG
NS=/blue/robot0
SIMT='-p use_sim_time:=true'
# each background command gets its own process group (setsid), so stop_all
# also reaches the node processes that `ros2 run` / `ros2 launch` start
bg() { local name=$1; shift; setsid "$@" > $LOG/$name.log 2>&1 & echo $! >> $LOG/pids; }
start_sim() {
  bg simulator python3 $HERE/sim_node.py
  sleep 2
  bg harness python3 $HERE/sim_harness.py --ros-args $SIMT "$@"
  bg connector ros2 run ssl_vision_connector protobuf_to_ros --ros-args $SIMT -p vision_port:=10020 -p mirror:=false
  bg vision_filter ros2 run vision_filter vision_filter --ros-args $SIMT
}
start_robot_stack() {  # $1 = use_odom true|false
  bg odometry ros2 run robot_control odometry --ros-args $SIMT -r __ns:=$NS
  bg robot_events ros2 run robot_control robot_events --ros-args $SIMT -r __ns:=$NS
  bg velocity_controller ros2 run robot_control velocity_controller --ros-args $SIMT -r __ns:=$NS -p use_odom:=$1
  bg kinematics ros2 run robot_control kinematics --ros-args $SIMT -r __ns:=$NS -r cmd_vel:=cmd_vel_corrected
}
stop_all() {
  for pid in $(cat $LOG/pids); do kill -INT -- -$pid 2>/dev/null; done
  sleep 3
  for pid in $(cat $LOG/pids); do kill -9 -- -$pid 2>/dev/null; done
  pkill -9 -f "$WS/install/" 2>/dev/null
  pkill -9 -f "$HERE/(sim_node|sim_harness|recorder|drive_straight|drive_events)\.py" 2>/dev/null
  sleep 1
  # leftovers would keep driving the robots in the next scenario
  local left
  left=$(pgrep -af "$WS/install/|$HERE/(sim_node|sim_harness|recorder|drive_)" | grep -v pgrep)
  if [ -n "$left" ]; then echo "WARNING: processes survived stop_all:"; echo "$left"; fi
}
