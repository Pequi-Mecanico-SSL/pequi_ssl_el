# Orange Pi 4 Pro robot setup

Board: Orange Pi 4 Pro (Allwinner A733), Orange Pi Debian 13 image, kernel 6.6.98-sun60iw2.
Same HAT and IMU wiring as the Raspberry Pi 4: the 40-pin header puts TWI0 on pins 3/5,
SPI3 on pins 19/21/23/24 and PD23 on pin 22 (the Pi's GPIO25).

| Function            | Raspberry Pi 4          | Orange Pi 4 Pro                |
|---------------------|-------------------------|--------------------------------|
| IMU (I2C, pins 3/5) | `/dev/i2c-1`            | `/dev/i2c-0` (`IMU_I2C_BUS=0` in `.env`) |
| CAN HAT (SPI CS0)   | `mcp2515-can0` overlay  | `mcp2515-can0.dts` merged into the DTB |
| Compose             | `docker-compose`        | `docker compose`               |

## Kernel with CAN

The stock kernel has `CONFIG_CAN` disabled, and CAN can't be added as out-of-tree modules
(it changes `struct net`), so the kernel is rebuilt from the same source with CAN as modules.
`CONFIG_UHID` is added too: BlueZ needs it to expose the BLE joystick as `/dev/input/js0`.
`build-kernel.sh` cross-compiles it in a Debian container (done on ryzen in
`~/Workspace/ssl/opi4pro-kernel`). Install on the board:

```sh
K=6.6.98-sun60iw2
# kernel: wrap Image with the same header as the stock uImage (arch "arm")
mkimage -A arm -O linux -T kernel -C none -a 0x41000000 -e 0x41000000 -n "Linux kernel" -d Image /boot/uImage
# modules: replace /lib/modules/$K, keeping the out-of-tree GPU driver extra/pvrsrvkm.ko
depmod -a $K && update-initramfs -u -k $K
apt-mark hold linux-image-current-sun60iw2 linux-dtb-current-sun60iw2 linux-u-boot-orangepi4pro-current
```

Stock kernel, initrd and modules are kept as `/boot/*.stock` and `/lib/modules/$K.stock`.

## Overlays and CAN interface

The i2c0 and MCP2515 overlays are merged into the DTB instead of listed in `overlays=` /
`user_overlays=`: the vendor U-Boot (2018.07) misreads the small `.dtbo` files from the SD
card on some boots, which silently leaves `can0` and `/dev/i2c-0` missing.

```sh
dtc -@ -I dts -O dtb -o /boot/overlay-user/mcp2515-can0.dtbo mcp2515-can0.dts
D=/boot/dtb/allwinner
fdtoverlay -i $D/sun60i-a733-orangepi-4-pro.dtb -o $D/sun60i-a733-orangepi-4-pro-ssl.dtb \
    $D/overlay/sun60i-a733-i2c0.dtbo /boot/overlay-user/mcp2515-can0.dtbo
# /boot/orangepiEnv.txt, with no overlays= / user_overlays= lines
fdtfile=allwinner/sun60i-a733-orangepi-4-pro-ssl.dtb

cp can0.service /etc/systemd/system/ && systemctl enable can0.service
apt install can-utils i2c-tools
```

## Crash recorder

`/var/log` (journal included) lives in RAM and only reaches the SD card on a clean shutdown,
so a hang or power loss leaves no trace. `blackbox.service` copies the journal and a 5 s
heartbeat (uptime, load, max temperature, free memory, can0) to `/var/lib/blackbox/`, one
file per boot, last 30 boots kept.

```sh
cp blackbox.sh /usr/local/sbin/ && cp blackbox.service /etc/systemd/system/
systemctl enable --now blackbox.service
```

## Wi-Fi: no background scans while connected

NetworkManager gives every Wi-Fi profile without a fixed BSSID background scanning (`bgscan "simple:30:-65:300"`: a scan
every 300 s, every 30 s with a weak signal). Each scan takes the radio off its channel for a moment, which shows as a
latency spike on the link to the strategy computer (worst ping 170 ms with a scan every 5 s, 80-125 ms without, on the
team router). Two files turn that off, and Wi-Fi power saving with it, for every network the robot joins:

```sh
sudo install -m 755 setup/orangepi4pro/50-wifi-no-bgscan /etc/NetworkManager/dispatcher.d/
sudo install -m 644 setup/orangepi4pro/wifi-powersave-off.conf /etc/NetworkManager/conf.d/
sudo nmcli general reload conf
sudo nmcli con up <profile>        # reconnect once so both take effect
```

`50-wifi-no-bgscan` runs each time a Wi-Fi interface comes up and clears the background-scan setting of the network just
joined. Nothing is pinned: the robot still finds and joins any known network (another router, a phone hotspot), and
after losing the link it scans as usual until it is connected again.

Check: `sudo wpa_cli -i wlan0 get_network 0 bgscan` answers `""`, `journalctl -t wifi-no-bgscan` has a line per
connection, and the driver's scan count stays put: `sudo dmesg | grep -c rwnx_cfg80211_scan`.

## Joystick (RadioMaster Boxer, BLE)

The agent must live in the same `bluetoothctl` session as `pair`, otherwise pairing fails
with `AuthenticationFailed`:

```sh
bluetoothctl
> agent NoInputNoOutput
> default-agent
> scan le          # wait for "ExpressLRS Joystick", then: scan off
> pair D4:8C:49:BB:D9:42
> trust D4:8C:49:BB:D9:42
```
