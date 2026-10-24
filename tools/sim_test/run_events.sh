# usage: run_events.sh <out.pkl>
SCENARIO=events
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_sim
start_robot_stack true
sleep 3
python3 $HERE/recorder.py $OUT/$1 48 $NS > $LOG/recorder.log 2>&1 &
REC=$!
sleep 1
python3 $HERE/drive_events.py > $LOG/driver.log 2>&1
wait $REC
stop_all
echo "done $SCENARIO"; grep -l -iE "traceback|exception" $LOG/*.log
