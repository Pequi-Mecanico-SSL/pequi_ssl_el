SCENARIO=match
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_sim -p blackout_every_s:=7.0
start_robot_stack true
# step 2: same filter with the cmd_vel hint off, on separate topics (paired comparison)
REMAPS=""
for c in blue yellow; do for i in 0 1 2; do
  REMAPS="$REMAPS -r /vision/filtered/$c/robot$i:=/vision/filtered_nohint/$c/robot$i -r /vision/filtered/odom/$c/robot$i:=/vision/filtered_nohint/odom/$c/robot$i"
done; done
bg vision_filter_nohint ros2 run vision_filter vision_filter --ros-args $SIMT -r __node:=vision_filter_nohint -p use_cmd_vel:=false $REMAPS
for team in blue yellow; do
  bg rl_$team ros2 run rl_strategy rl_strategy --ros-args $SIMT -r __node:=rl_$team \
     -p team:=$team -p field_length_m:=9.0 -p field_width_m:=6.0
done
sleep 4
bg vision_check_blue1 ros2 run vision_filter vision_check --ros-args $SIMT -p robot:=blue/robot1 -p seconds:=60.0
bg vision_check_yellow0 ros2 run vision_filter vision_check --ros-args $SIMT -p robot:=yellow/robot0 -p seconds:=60.0
bg odom_check ros2 run robot_control odom_check --ros-args $SIMT -r __ns:=$NS -p seconds:=60.0
python3 $HERE/recorder.py $OUT/match.pkl ${REC_S:-120} $NS > $LOG/recorder.log 2>&1
stop_all
echo "done"; grep -l -iE "traceback|exception" $LOG/*.log
