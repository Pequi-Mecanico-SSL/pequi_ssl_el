#!/bin/bash
# Crash recorder: /var/log is RAM-backed and lost on a hang or power loss, so this copies
# the journal plus a 5 s heartbeat to the SD card, one file per boot.
D=/var/lib/blackbox
mkdir -p $D
ls -1 $D/*.log 2>/dev/null | head -n -30 | xargs -r rm -f
last=$(ls -1 $D/*.log 2>/dev/null | tail -1 | sed 's|.*/0*\([0-9]\+\)-.*|\1|')
F=$(printf '%s/%05d-%s.log' $D $((${last:-0} + 1)) "$(cut -c1-8 /proc/sys/kernel/random/boot_id)")
exec >> "$F" 2>&1
journalctl -b -f --no-tail -o short-precise &
n=0
while :; do
  if [ $((n++ % 5)) = 0 ]; then
    echo "HB $(date '+%F %T') up=$(cut -d' ' -f1 /proc/uptime) load=$(cut -d' ' -f1 /proc/loadavg)" \
      "temp=$(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1)" \
      "memavail_kB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
      "can0=$(cat /sys/class/net/can0/operstate 2>/dev/null),rx=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null)"
  fi
  sync -d "$F"
  sleep 1
done
