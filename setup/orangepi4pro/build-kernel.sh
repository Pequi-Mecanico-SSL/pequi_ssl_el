#!/bin/bash
# Runs inside debian:trixie with /w holding linux/ (orange-pi-6.6-sun60iw2 @ 1ecd47b8f)
# and config-stock (/boot/config-6.6.98-sun60iw2 from the board); output in /w/out:
#   docker run --rm -v $PWD:/w debian:trixie /w/build-kernel.sh
set -e
apt-get update -qq
apt-get install -y -qq gcc-aarch64-linux-gnu make bc bison flex libssl-dev kmod cpio rsync python3 u-boot-tools xz-utils >/dev/null
cd /w/linux
export ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- LOCALVERSION=-sun60iw2
cp /w/config-stock .config
for o in CAN CAN_RAW CAN_BCM CAN_GW CAN_ISOTP CAN_DEV CAN_VCAN CAN_VXCAN CAN_SLCAN CAN_MCP251X CAN_MCP251XFD CAN_GS_USB; do
  scripts/config --module $o
done
scripts/config --enable CAN_NETLINK --enable CAN_CALC_BITTIMING --module UHID
make olddefconfig
grep -E "CONFIG_CAN(_RAW|_DEV|_MCP251X|_GS_USB|_SLCAN|_NETLINK)?=" .config
make -j32 KBUILD_BUILD_VERSION=1.1.0+can Image modules 2>&1 | grep -E "error|Error|warning: .*undefined" || true
test -f arch/arm64/boot/Image
rm -rf /w/out && mkdir -p /w/out
make -s INSTALL_MOD_PATH=/w/out/mod INSTALL_MOD_STRIP=1 modules_install
cp arch/arm64/boot/Image .config System.map /w/out/
mkimage -A arm64 -O linux -T kernel -C none -a 0x41000000 -e 0x41000000 -n "Linux kernel" -d arch/arm64/boot/Image /w/out/uImage >/dev/null
make -s kernelrelease
chown -R $(stat -c %u:%g /w) /w/linux /w/out
