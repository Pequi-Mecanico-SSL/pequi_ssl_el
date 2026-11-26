#!/bin/bash
# Passive recording of whatever is on the field network (vision + referee): nothing is sent except multicast joins.
#   tools/solo/record_match.sh start [name]   e.g. start field_test
#   tools/solo/record_match.sh status | stop
OUT=${SOLO_DATA:-$(dirname "$(readlink -f "$0")")/../../log/solo}/raw      # SOLO_DATA default: log/solo in this repository
PIDF=/tmp/solo_rec/record_match.pid; mkdir -p /tmp/solo_rec $OUT
case "$1" in
start)
  [ -f $PIDF ] && kill -0 $(cat $PIDF) 2>/dev/null && { echo "already recording"; exit 1; }
  setsid nohup python3 -u $(dirname $(readlink -f $0))/record_raw.py --out $OUT ${2:+--tag $2} >> $OUT/record_match.log 2>&1 &
  echo $! > $PIDF; sleep 2; tail -3 $OUT/record_match.log;;
status)
  [ -f $PIDF ] && kill -0 $(cat $PIDF) 2>/dev/null && echo "recording (pid $(cat $PIDF))" || echo "not recording"
  tail -2 $OUT/record_match.log; ls -la --time-style=+%H:%M $OUT/raw_*.bin.gz | tail -3 | awk '{print $5, $6, $7}'
  df -h $OUT | tail -1 | awk '{print "free disk:", $4}';;
stop)
  [ -f $PIDF ] && kill -TERM $(cat $PIDF) 2>/dev/null; sleep 1.5; rm -f $PIDF; tail -1 $OUT/record_match.log;;
*) echo "usage: $0 start [name] | status | stop";;
esac
