# usage: run_straight.sh <use_odom true|false> <gyro_bias_end> <out.pkl>
SCENARIO=straight_$1_$2
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_sim -p gyro_bias_end:=$2
start_robot_stack $1
REMAPS=""
for c in blue yellow; do for i in 0 1 2; do
  REMAPS="$REMAPS -r /vision/filtered/$c/robot$i:=/vision/filtered_nohint/$c/robot$i -r /vision/filtered/odom/$c/robot$i:=/vision/filtered_nohint/odom/$c/robot$i"
done; done
bg vision_filter_nohint ros2 run vision_filter vision_filter --ros-args $SIMT -r __node:=vision_filter_nohint -p use_cmd_vel:=false $REMAPS
sleep 3
python3 $HERE/recorder.py $OUT/$3 100 $NS > $LOG/recorder.log 2>&1 &
REC=$!
sleep 1
python3 $HERE/drive_straight.py > $LOG/driver.log 2>&1
wait $REC
stop_all
echo "done $SCENARIO"; grep -l -iE "traceback|exception" $LOG/*.log
