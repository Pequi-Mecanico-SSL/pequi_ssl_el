#!/bin/bash
# Spin-kick session on the real robot. THE ROBOT SPINS FAST AND THROWS THE BALL. Run only with the area clear and
# nothing else publishing /cmd_vel. CSV logs go to $SPINKICK_OUT/logs, result lines to $SPINKICK_OUT/results.jsonl.
#   spinkick_go.sh pose X Y TH_DEG                  drive there (characterize.py goto; BOX defaults to the goto area)
#   spinkick_go.sh plain RATE [SECONDS]             Enter when the ball is in the pocket, then characterize.py spin
#   spinkick_go.sh sling RATE [SECONDS ACCEL TURN]  spinkick.py sling (waits for the resting ball, then the operator's Enter)
#   spinkick_go.sh carry SPEED RATE [CARRY_M SECONDS KEEP LAT]   spinkick.py carry
#   spinkick_go.sh ram SPEED DIR_DEG RUNUP PAST     characterize.py ram (PAST 0.33 = capture-hold-brake at 1.0 m/s)
#   spinkick_go.sh prop SPEED DIR_DEG RATE X0 Y0 SECONDS   characterize.py fspin (latency $LAT, default 0.10)
#   spinkick_go.sh an                               the analysis table of every log so far (no ROS)
# BOX="x0,x1,y0,y1" overrides the test box (default: 0.8 m square round the field centre; carry/prop/ram: the lane).
# Environment: SPINKICK_OUT (default $SOLO_DATA/spinkick, SOLO_DATA defaulting to log/solo in this repository),
# CH_TEAM (blue), CH_ID (the robot's vision id, 1), SOLO_IFACE / SOLO_DOMAIN as for the other tools in tools/solo.
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); T=$R/tools/solo
OUT=${SPINKICK_OUT:-${SOLO_DATA:-$R/log/solo}/spinkick}
[ "$1" = an ] && { python3 -I $T/spinkick_analyze.py $OUT/logs/*.csv --json $OUT/analysis.jsonl; exit; }
[ -n "$1" ] || { sed -n 2,13p "$0"; exit; }
mkdir -p $OUT/logs
cd $R && source tools/solo/env.sh
export SOLO_ROBOT_INDEX=${CH_ID:-1} SOLO_GOTO_BOX="-2.0,2.0,-1.25,1.25"
TEAM=${CH_TEAM:-blue}; SMALL="-0.4,0.4,-0.4,0.4"; LANE="-1.6,1.0,-0.5,0.5"
[ "$(ps -eo args | grep -E 'solo_node|duo_node|hq\.py|probe\.py|frame_check|series\.py|characterize\.py|spinkick\.py|place_ball\.py|auto_session\.py|goto\.py|nudge\.py|duo_check\.py|ram_series\.py' | grep -v grep | wc -l)" = 0 ] || { echo "REFUSED: something else may publish /cmd_vel"; exit 1; }
log() { tee -a $OUT/session.out | grep --line-buffered -v "sequence size"; }   # line-buffered: prompts appear at once
keep() { grep '^{' | tail -1 >> $OUT/results.jsonl; }      # the tool's JSON result line
echo "=== $(date +%H:%M:%S) $*" | log
case "$1" in
pose)  SOLO_BOX=${BOX:-$SOLO_GOTO_BOX} python3 $T/characterize.py $TEAM goto $2 $3 $4 --out $OUT/logs 2>&1 | log;;
plain) read -p "ball in the pocket, hands away, then Enter (Ctrl-C = no spin) "
       SOLO_BOX=${BOX:-$SMALL} python3 $T/characterize.py $TEAM spin $2 ${3:-0.5} --out $OUT/logs 2>&1 | log | tee >(keep);;
sling) SOLO_BOX=${BOX:-$SMALL} python3 $T/spinkick.py $TEAM sling "${@:2}" --out $OUT/logs 2>&1 | log | tee >(keep);;
ram)   SOLO_BOX=${BOX:-$LANE} python3 $T/characterize.py $TEAM ram $2 $3 $4 $5 --out $OUT/logs 2>&1 | log | tee >(keep);;
carry) SOLO_BOX=${BOX:-$LANE} python3 $T/spinkick.py $TEAM carry "${@:2}" --out $OUT/logs 2>&1 | log | tee >(keep);;
prop)  SOLO_BOX=${BOX:-$LANE} python3 $T/characterize.py $TEAM fspin $2 $3 $4 ${LAT:-0.10} $5 $6 $7 --out $OUT/logs 2>&1 | log | tee >(keep);;
*)     sed -n 2,13p "$0";;
esac
