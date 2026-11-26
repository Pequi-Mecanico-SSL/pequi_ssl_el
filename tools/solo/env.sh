# ROS environment for the one-robot match setup on the computer that runs the strategy (source it)
source /opt/ros/humble/setup.bash
_solo_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source $_solo_repo/install/setup.bash
[ -n "$SOLO_OVERLAY" ] && source "$SOLO_OVERLAY"      # a pinned build of solo_strategy on top (optional)
export PYTHONPATH=$PYTHONPATH:$_solo_repo/../ssl-rules/src     # the rules package (import sslrules), next to this repository
unset ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT ROS_LOCALHOST_ONLY   # discovery-server settings a shell profile may carry
# default 77, the same default as tools/solo/robot_stack.sh, so that both sides meet on one ROS domain
export ROS_DOMAIN_ID=${SOLO_DOMAIN:-77}
# the vision coverage map module solo_node's recovery imports (its recover_map_module parameter defaults to this)
export SOLO_VISION_MAP_MODULE=${SOLO_VISION_MAP_MODULE:-$_solo_repo/tools/solo/vision_map.py}
# SOLO_IFACE=<this machine's address on the robot's network>: announce and use only that address for DDS. A machine
# with many interfaces (bridges, VPN, a second address on one port) announces every address, and a peer that keeps
# only the first few never learns the one it can reach: topics are listed but no data arrives.
if [ -n "$SOLO_IFACE" ]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/solo_fastdds_$SOLO_IFACE.xml
  cat > $FASTRTPS_DEFAULT_PROFILES_FILE <<XML
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>udp_one_interface</transport_id>
      <type>UDPv4</type>
      <interfaceWhiteList><address>$SOLO_IFACE</address></interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="one_interface" is_default_profile="true">
    <rtps>
      <userTransports><transport_id>udp_one_interface</transport_id></userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
</profiles>
XML
fi
