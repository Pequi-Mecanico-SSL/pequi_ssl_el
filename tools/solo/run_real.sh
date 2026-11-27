#!/bin/bash
# One-robot match setup on the real field: policy server + vision connector + referee connector + strategy node.
#   tools/solo/run_real.sh start <blue|yellow> [mirror: true|false] [policy params JSON] [extra solo_node ros args...]
#   tools/solo/run_real.sh stop | status
# Logs and pid files in /tmp/solo_run/. The robot must run robot_control (it listens on /cmd_vel).
# SOLO_SIM_DIR (required): checkout of the strategy simulator package (import sslenv), with its .venv; SOLO_ARENA
# overrides only the source tree the policy server runs from.
# SOLO_CKPT=<a checkpoint of the strategy simulator>: a trained policy plays instead of the scripted one (policy_server.py
# --ckpt; its params JSON defaults to {}), told it has the robot SOLO_RL_ROBOT (default three_wheel_fast: command ramps
# 1.7 / 1.5 m/s^2, braking 3; three_wheel_fast_5a for the robot at an ESC limit of 5 A), a robot definition from the
# files SOLO_RL_ROBOTS (default robots/three_wheel_fast.json:robots/three_wheel_fast_5a.json, relative to the simulator
# checkout). The server's commands then carry that robot's ramps and must reach it as they are: the node runs with its
# own ramp off, speed cap 1.5 m/s and yaw cap 6 rad/s (node args given after them win, but a lower cap is not what the
# policy's memory holds).
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
  TEAM=${2:?team colour}; MIRROR=${3:-false}; PARAMS=${4:-'{"nk_hold": 2.0}'}
  POLICY=(); RL_NODE=()
  if [ -n "$SOLO_CKPT" ]; then
    CK=$(readlink -f "$SOLO_CKPT"); [ -f "$CK" ] || { echo "REFUSED: no checkpoint at SOLO_CKPT=$SOLO_CKPT"; exit 1; }
    PARAMS=${4:-'{}'}
    POLICY=(--ckpt "$CK" --robot "${SOLO_RL_ROBOT:-three_wheel_fast}")
    RL_NODE=(-p ramp_fwd:=0.0 -p ramp_side:=0.0 -p speed_cap:=1.5 -p angular_cap:=6.0)
    export SSL_ROBOTS=${SOLO_RL_ROBOTS:-robots/three_wheel_fast.json:robots/three_wheel_fast_5a.json}
  fi
  shift 4 2>/dev/null || shift $#
  stop; sleep 0.5
  OTHER=$([ "$TEAM" = blue ] && echo yellow || echo blue)
  (A=${SOLO_ARENA:-$SIM}; cd $A && PYTHONPATH=$A/src:$A/ssl-rules/src JAX_PLATFORMS=cpu nohup $SIM/.venv/bin/python ${SOLO_SRC:-$REPO/src}/solo_strategy/solo_strategy/policy_server.py "${POLICY[@]}" --params "$PARAMS" > $D/policy.log 2>&1 & echo $! > $D/policy.pid)
  (nohup ros2 run ssl_vision_connector protobuf_to_ros --ros-args -p vision_port:=10006 -p mirror:=$MIRROR -p ${TEAM}_robot_count:=${SOLO_OWN_COUNT:-1} -p ${OTHER}_robot_count:=${SOLO_OTHER_COUNT:-3} > $D/vision.log 2>&1 & echo $! > $D/vision.pid)
  (nohup ros2 launch referee_connector referee_connector_launch.yaml > $D/referee.log 2>&1 & echo $! > $D/referee.pid)
  (nohup ros2 run solo_strategy solo_node --ros-args -p team:=$TEAM -p robot_index:=${SOLO_ROBOT_INDEX:-0} -p cmd_topic:=/cmd_vel "${RL_NODE[@]}" "$@" > $D/solo.log 2>&1 & echo $! > $D/solo.pid)
  echo "started: team $TEAM, mirror $MIRROR, ${SOLO_CKPT:+trained policy $CK on ${SOLO_RL_ROBOT:-three_wheel_fast}, }params $PARAMS, extra: ${RL_NODE[*]} $*";;
*) echo "usage: $0 start <blue|yellow> [mirror] [params json] [ros args] | stop | status";;
esac
