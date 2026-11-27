#!/bin/bash
# The strategy's `latency` (solo_node / duo_node: the field velocity is turned into the body frame the robot will have
# `latency` s later) checked on the robot: six field-frame moves at 0.8 m/s while spinning at 3 rad/s (characterize.py
# fspin), latency 0.1 / 0.04 / 0.0, each once per spin direction, alternating up and down the lane x = X.
#   source tools/solo/env.sh; tools/solo/fspin_runs.sh <team> [OUT_DIR]       (X=-1.1 default: the -x half)
# SHORT=1: the first two runs only (0.1 up, 0.04 down; about 20 s).
# Needs: the strategy OFF and nothing else commanding the robot (checked: procs.py), the ball off the lane. The robot
# first drives to (X, -1.0) at 0.5 m/s (goto.py, round robots and the ball, not round the half line: start it in this
# half). Free carpet: x X-0.65 .. X+0.65, y -1.25 .. 1.25 (SOLO_BOX). About 1 min. Stop: Ctrl-C (zeros, log kept).
# Prints at the end (fspin_report.py): "DECIDE: set latency to X" (the zero of bend against latency).
set -e
TEAM=${1:?team colour}; X=${X:--1.1}
HERE=$(dirname "$(readlink -f "$0")")
OUT=${2:-${SOLO_DATA:-$HERE/../../log/solo}/fspin}
mkdir -p "$OUT"
python3 - "$HERE" <<'PY'
import os, sys; sys.path.insert(0, sys.argv[1]); import procs
dry = os.environ.get('FSPIN_DRY') == '1' and os.environ.get('ROS_LOCALHOST_ONLY') == '1'   # offline test (tests/fake_field.py)
r = [] if dry else procs.running(exclude=procs.ancestors())
if r: sys.exit(f"refusing: these command the robot now: {r} (switch the strategy off / stop them first)")
PY
x0=$(python3 -c "print(round($X - 0.65, 2))"); x1=$(python3 -c "print(round($X + 0.65, 2))")
echo "== to the start of the lane ($X, -1.0)"      # a characterize test that starts outside its box aborts
python3 "$HERE/goto.py" "$TEAM" "$X" -1.0 --v 0.5 | tail -1
export SOLO_BOX="$x0,$x1,-1.25,1.25" SOLO_RAMP=${SOLO_RAMP:-1.7}
run() {   # latency, dir, rate, y0
  echo "== latency $1, toward $2 deg, rate $3"
  python3 "$HERE/characterize.py" "$TEAM" fspin 0.8 "$2" "$3" "$1" "$X" "$4" 2.2 --out "$OUT" | tail -1 | tee -a "$OUT/fspin_results.jsonl"
}
run 0.1  90  3.0 -1.0
run 0.04 -90 -3.0 1.0
if [ -z "$SHORT" ]; then
run 0.0  90  3.0 -1.0
run 0.1  -90 -3.0 1.0
run 0.04 90  3.0 -1.0
run 0.0  -90 -3.0 1.0
fi
python3 -I "$HERE/fspin_report.py" "$OUT"
