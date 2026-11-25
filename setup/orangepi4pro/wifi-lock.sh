#!/bin/bash
# Lock the active Wi-Fi profile to the access point the robot is connected to, and turn Wi-Fi power saving off.
# Run it once on the robot, as root, while it is connected to the network it plays on (and again after a router change).
#
# Why: NetworkManager lets wpa_supplicant scan in the background on a profile that has no BSSID (bgscan
# "simple:30:-65:300": every 300 s, every 30 s when the signal is below -65 dBm). During a scan the radio leaves the
# channel and the link to the strategy computer stalls for up to seconds. With a BSSID set there are no background scans
# and no roaming.
#
# If the gateway does not answer 40 s after the change, the profile is put back as it was.
set -u
C=$(nmcli -t -f NAME,TYPE con show --active | grep wireless | cut -d: -f1)
B=$(nmcli -t -f IN-USE,BSSID dev wifi | grep '^\*' | cut -d: -f2- | sed 's/\\//g')
GW=$(ip route | awk '/default/ {print $3; exit}')
[ -n "$C" ] && [ -n "$B" ] || { echo "no active Wi-Fi connection"; exit 1; }
echo "profile '$C': bssid $B, power save off (gateway $GW)"
nmcli con mod "$C" 802-11-wireless.bssid "$B" 802-11-wireless.powersave 2 || exit 1
nmcli con up "$C"
sleep 40
if ping -c 3 -W 2 "${GW:-192.168.1.1}" > /dev/null 2>&1; then
    echo "ok: still connected"
else
    echo "gateway unreachable: undoing"
    nmcli con mod "$C" 802-11-wireless.bssid "" 802-11-wireless.powersave 0
    nmcli con up "$C"
fi
