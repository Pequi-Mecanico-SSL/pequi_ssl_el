# Orange Pi 4 Pro robot setup

Board: Orange Pi 4 Pro (Allwinner A733), Orange Pi Debian 13 image, kernel 6.6.98-sun60iw2.
Same HAT and IMU wiring as the Raspberry Pi 4: the 40-pin header puts TWI0 on pins 3/5,
SPI3 on pins 19/21/23/24 and PD23 on pin 22 (the Pi's GPIO25).

| Function            | Raspberry Pi 4          | Orange Pi 4 Pro                |
|---------------------|-------------------------|--------------------------------|
| IMU (I2C, pins 3/5) | `/dev/i2c-1`            | `/dev/i2c-0` (`IMU_I2C_BUS=0` in `.env`) |
| CAN HAT (SPI CS0)   | `mcp2515-can0` overlay  | `mcp2515-can0.dts` user overlay |
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

```sh
dtc -@ -I dts -O dtb -o /boot/overlay-user/mcp2515-can0.dtbo mcp2515-can0.dts
# /boot/orangepiEnv.txt
overlays=i2c0
user_overlays=mcp2515-can0

cp can0.service /etc/systemd/system/ && systemctl enable can0.service
apt install can-utils i2c-tools
```

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
