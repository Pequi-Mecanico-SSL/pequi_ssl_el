#!/bin/bash
# Two-robot match setup on the real field: policy server (duo_nk, both robots) + vision connector + referee connector
# + one strategy node (duo_node) that sends each robot its own commands on the robot's own ROS domain.
#   tools/solo/run_duo.sh start <blue|yellow> <keeper id> <field id> [mirror: true|false] [policy params JSON] [extra duo_node ros args...]
#   tools/solo/run_duo.sh stop | status
#   tools/solo/run_duo.sh bench <vision id> | bench none     a robot off the field (a card, taken off): zero, the other plays alone
#   tools/solo/run_duo.sh keeper <vision id>                 swap roles (tell the game controller the same goalkeeper id)
# Ids are vision ids (the pattern's id). The robots' domains: SOLO_DOMAINS="<keeper's> <field robot's>" (default
# "77 78"); each robot's stack is started with the same number: SOLO_DOMAIN=<n> ROBOT=<user@address> tools/solo/robot_stack.sh start.
# This computer's own topics (vision, referee, the page) are on SOLO_DOMAIN (default 77, tools/solo/env.sh).
# Logs and pid files in /tmp/solo_run/, as run_real.sh (either script's stop stops both setups).
# SOLO_SIM_DIR (required): checkout of the strategy simulator package (import sslenv), with its .venv; SOLO_ARENA
# overrides only the source tree the policy server runs from.
D=/tmp/solo_run; mkdir -p $D
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source $REPO/tools/solo/env.sh
stop() { for f in $D/*.pid; do [ -f "$f" ] || continue; p=$(cat $f); pkill -P $p 2>/dev/null; kill $p 2>/dev/null; rm -f $f; done; }
int() { [[ "$1" =~ ^[0-9]+$ ]]; }
case "$1" in
stop) stop; echo stopped;;
status) for f in $D/*.pid; do [ -f "$f" ] && (kill -0 $(cat $f) 2>/dev/null && echo "$(basename $f .pid): running" || echo "$(basename $f .pid): DEAD"); done
        grep "mode:\|keeper id" $D/duo.log 2>/dev/null | tail -3 | cut -c1-220; tail -1 $D/policy.log 2>/dev/null | cut -c1-200;;
bench)
  [ -n "$2" ] || { echo "usage: $0 bench <vision id> | none"; exit 1; }
  if [ "$2" = none ]; then V="[-1]"; else int "$2" || { echo "not a vision id: $2"; exit 1; }; V="[$2]"; fi
  ros2 param set /duo_strategy bench "$V";;
keeper)
  int "$2" || { echo "usage: $0 keeper <vision id>"; exit 1; }
  ros2 param set /duo_strategy keeper "$2";;
start)
  SIM=${SOLO_SIM_DIR:?set SOLO_SIM_DIR to the strategy simulator checkout}
  TEAM=${2:?team colour}; KID=${3:?keeper vision id}; FID=${4:?field robot vision id}; MIRROR=${5:-false}; PARAMS=${6:-'{"nk_hold": 2.0}'}
  int "$KID" && int "$FID" || { echo "REFUSED: vision ids must be numbers: '$KID' '$FID'"; exit 1; }
  shift 6 2>/dev/null || shift $#
  read -r KDOM FDOM REST <<< "${SOLO_DOMAINS:-77 78}"
  int "$KDOM" && int "$FDOM" && [ -z "$REST" ] || { echo "REFUSED: SOLO_DOMAINS must be two domain numbers, not '$SOLO_DOMAINS'"; exit 1; }
  # duo_nk plays both robots from one program only with d_role -1 (the team); any other value is refused
  PARAMS=$(python3 -c 'import json,sys; p=json.loads(sys.argv[1]); assert p.get("d_role",-1)==-1, "d_role must be -1 (the team) for run_duo.sh"; p["d_role"]=-1; print(json.dumps(p))' "$PARAMS") || { echo "REFUSED: policy params"; exit 1; }
  stop; sleep 0.5
  OTHER=$([ "$TEAM" = blue ] && echo yellow || echo blue)
  (A=${SOLO_ARENA:-$SIM}; cd $A && PYTHONPATH=$A/src:$A/ssl-rules/src JAX_PLATFORMS=cpu nohup $SIM/.venv/bin/python ${SOLO_SRC:-$REPO/src}/solo_strategy/solo_strategy/policy_server.py --bot duo_nk --params "$PARAMS" > $D/policy.log 2>&1 & echo $! > $D/policy.pid)
  # our colour's count 16: the connector's topic number is then the vision id itself (duo_node reads robot<id>)
  (nohup ros2 run ssl_vision_connector protobuf_to_ros --ros-args -p vision_port:=10006 -p mirror:=$MIRROR -p ${TEAM}_robot_count:=16 -p ${OTHER}_robot_count:=${SOLO_OTHER_COUNT:-3} > $D/vision.log 2>&1 & echo $! > $D/vision.pid)
  (nohup ros2 launch referee_connector referee_connector_launch.yaml > $D/referee.log 2>&1 & echo $! > $D/referee.pid)
  (nohup ros2 run solo_strategy duo_node --ros-args -p team:=$TEAM -p ids:="[$KID, $FID]" -p keeper:=$KID -p domains:="[$KDOM, $FDOM]" "$@" > $D/duo.log 2>&1 & echo $! > $D/duo.pid)
  echo "started: team $TEAM, keeper id $KID on domain $KDOM, field robot id $FID on domain $FDOM, mirror $MIRROR, params $PARAMS, extra: $*"
  echo "the policy server compiles for about a minute; until then both robots stand ('policy server does not answer')";;
*) echo "usage: $0 start <blue|yellow> <keeper id> <field id> [mirror] [params json] [ros args] | stop | status | bench <id>|none | keeper <id>";;
esac
