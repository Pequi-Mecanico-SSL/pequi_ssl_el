#!/bin/bash
# Runs every scenario (about 12 minutes) and writes results/summary.txt.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-$HERE/results}; mkdir -p $OUT
bash $HERE/run_match.sh
for mode in false true; do for bias_end in 0.05 0.02; do
  bash $HERE/run_straight.sh $mode $bias_end straight_${mode}_${bias_end}.pkl
done; done
bash $HERE/run_events.sh events.pkl
{
  echo "=== match: vision filter, odometry, events vs truth"; python3 $HERE/analyze.py $OUT/match.pkl
  echo; echo "=== match: vision filter with vs without the cmd_vel hint"; python3 $HERE/vision_ab.py $OUT/match.pkl
  echo; echo "=== match: robot_events vs truth"; python3 $HERE/events_analysis.py $OUT/match.pkl
  echo; echo "=== driving straight: heading hold, imu vs odom feedback"
  python3 $HERE/straight_analysis.py imu_warmup=$OUT/straight_false_0.05.pkl odom_warmup=$OUT/straight_true_0.05.pkl \
      imu_constant=$OUT/straight_false_0.02.pkl odom_constant=$OUT/straight_true_0.02.pkl
  echo; echo "=== driving straight: vision filter with vs without the hint (free driving)"; python3 $HERE/vision_ab.py $OUT/straight_true_0.05.pkl
  echo; echo "=== event course: robot_events vs truth"; python3 $HERE/events_analysis.py $OUT/events.pkl
} 2>&1 | tee $OUT/summary.txt
