#!/bin/bash
# One-robot match setup on the real field: policy server + vision connector + referee connector + strategy node.
#   tools/solo/run_real.sh start <blue|yellow> [mirror: true|false] [policy params JSON] [extra solo_node ros args...]
#   tools/solo/run_real.sh stop | status
# Logs and pid files in /tmp/solo_run/. The robot must run robot_control (it listens on /cmd_vel).
# SOLO_SIM_DIR (required): checkout of the strategy simulator package (import sslenv), with its .venv; SOLO_ARENA
# overrides only the source tree the policy server runs from.
D=/tmp/solo_run; mkdir -p $D
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source $REPO/tools/solo/env.sh
stop() { for f in $D/*.pid; do [ -f "$f" ] || continue; p=$(cat $f); pkill -P $p 2>/dev/null; kill $p 2>/dev/null; rm -f $f; done; }
case "$1" in
stop) stop; echo stopped;;
status) for f in $D/*.pid; do [ -f "$f" ] && (kill -0 $(cat $f) 2>/dev/null && echo "$(basename $f .pid): running" || echo "$(basename $f .pid): DEAD"); done
        grep "mode:" $D/solo.log 2>/dev/null | tail -3 | cut -c1-200; tail -1 $D/policy.log 2>/dev/null | cut -c1-200;;
start)
  SIM=${SOLO_SIM_DIR:?set SOLO_SIM_DIR to the strategy simulator checkout}
  TEAM=${2:?team colour}; MIRROR=${3:-false}; PARAMS=${4:-'{"nk_hold": 2.0}'}; shift 4 2>/dev/null
  stop; sleep 0.5
  OTHER=$([ "$TEAM" = blue ] && echo yellow || echo blue)
  (A=${SOLO_ARENA:-$SIM}; cd $A && PYTHONPATH=$A/src:$A/ssl-rules/src JAX_PLATFORMS=cpu nohup $SIM/.venv/bin/python ${SOLO_SRC:-$REPO/src}/solo_strategy/solo_strategy/policy_server.py --params "$PARAMS" > $D/policy.log 2>&1 & echo $! > $D/policy.pid)
  (nohup ros2 run ssl_vision_connector protobuf_to_ros --ros-args -p vision_port:=10006 -p mirror:=$MIRROR -p ${TEAM}_robot_count:=${SOLO_OWN_COUNT:-1} -p ${OTHER}_robot_count:=${SOLO_OTHER_COUNT:-3} > $D/vision.log 2>&1 & echo $! > $D/vision.pid)
  (nohup ros2 launch referee_connector referee_connector_launch.yaml > $D/referee.log 2>&1 & echo $! > $D/referee.pid)
  (nohup ros2 run solo_strategy solo_node --ros-args -p team:=$TEAM -p robot_index:=${SOLO_ROBOT_INDEX:-0} -p cmd_topic:=/cmd_vel "$@" > $D/solo.log 2>&1 & echo $! > $D/solo.pid)
  echo "started: team $TEAM, mirror $MIRROR, params $PARAMS, extra: $*";;
*) echo "usage: $0 start <blue|yellow> [mirror] [params json] [ros args] | stop | status";;
esac
