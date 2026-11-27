#!/bin/bash
# The match set-up by one number; everything after it is the same for all (the field robot id is ignored by 3 and 4):
#   tools/solo/run_rung.sh 1 <blue|yellow> <keeper id> <field id> [mirror] [node ros args...]
#       two robots, duo_att with profiles/duo_att.json (the keeper keeps goal, the field robot plays every ball it
#       reaches first while their robots are slow; once one of theirs is seen faster than a_nfv in play: duo_nk's wall)
#   tools/solo/run_rung.sh 2 ...  two robots, duo_nk with the one-robot field profile (profiles/field.json) on the keeper
#   tools/solo/run_rung.sh 3 ...  one robot, the keeper id: solo_nk with profiles/field.json, drive-back on (run_real.sh)
#   SOLO_CKPT=<checkpoint> tools/solo/run_rung.sh 4 ...  one robot, the keeper id: the trained policy of that checkpoint
#       (run_real.sh has its settings: SOLO_RL_ROBOT, the node's ramp off, 1.5 m/s, 6 rad/s); no drive-back
#   tools/solo/run_rung.sh show <1|2|3|4> ...   print the command it would run, run nothing
# Speed cap 1.5 m/s for all of them (give -p speed_cap:=<v> after the mirror to change it). Stop: tools/solo/run_duo.sh
# stop (it stops either set-up). Ids are vision ids; SOLO_DOMAINS as for run_duo.sh (default "77 78").
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
SHOW=""
[ "$1" = show ] && { SHOW=echo; shift; }
RUNG=$1; TEAM=$2; KID=$3; FID=$4; MIRROR=${5:-false}
[ -n "$TEAM" ] && [ -n "$KID" ] && [ -n "$FID" ] || { sed -n 2,15p "$0"; exit 1; }
shift 5 2>/dev/null || shift $#
FIELD=$(cat "$HERE/profiles/field.json") || exit 1
case "$RUNG" in
1) $SHOW env SOLO_BOT=duo_att "$HERE/run_duo.sh" start "$TEAM" "$KID" "$FID" "$MIRROR" "$(cat "$HERE/profiles/duo_att.json")" -p speed_cap:=1.5 "$@";;
2) $SHOW env SOLO_BOT=duo_nk "$HERE/run_duo.sh" start "$TEAM" "$KID" "$FID" "$MIRROR" "$FIELD" -p speed_cap:=1.5 "$@";;
3) $SHOW env SOLO_OWN_COUNT=16 SOLO_ROBOT_INDEX="$KID" "$HERE/run_real.sh" start "$TEAM" "$MIRROR" "$FIELD" -p speed_cap:=1.5 -p recover:=true "$@";;
4) [ -n "$SOLO_CKPT" ] || { echo "rung 4 needs SOLO_CKPT=<the trained policy's checkpoint>"; exit 1; }
   $SHOW env SOLO_OWN_COUNT=16 SOLO_ROBOT_INDEX="$KID" "$HERE/run_real.sh" start "$TEAM" "$MIRROR" '{}' "$@";;
*) echo "rung must be 1, 2, 3 or 4, not '$RUNG'"; exit 1;;
esac
