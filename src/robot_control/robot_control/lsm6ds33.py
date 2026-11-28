"""LSM6DS33 gyro + accelerometer (Pololu MinIMU-9 v5), read in a background thread.

Axes as published on /imu: the board sits upside down, rates are negated so that
z is positive CCW seen from the top; accelerations are the chip's own axes.

A chip that stops sampling while it still answers on the bus (reset into
power-down, or hung) keeps returning its last reading. Such readings are not
passed on, so take() returns None and the reader sees the gyro as missing:
- the configuration is read back once a second and rewritten when it differs;
- STUCK_READS identical gyro readings in a row (all three axes) stop the
  samples; the chip is then reset and configured, once a second, until its
  readings change again.
No sample is passed on for SETTLE_S after the configuration is written.
"""
import math
import struct
import threading
import time

from smbus2 import SMBus

ADDRESS = 0x6B
CTRL1_XL, CTRL2_G, CTRL3_C, CTRL10_C, OUTX_L_G = 0x10, 0x11, 0x12, 0x19, 0x22
GYRO_LSB = math.radians(0.070)          # rad/s per count at 2000 dps
ACCEL_LSB = 0.122e-3 * 9.80665          # m/s^2 per count at 4 g
CONFIG = ((CTRL10_C, 0x38),             # gyro axes on
          (CTRL2_G, 0x5C),              # gyro 208 Hz, 2000 dps
          (CTRL1_XL, 0x58),             # accelerometer 208 Hz, 4 g
          (CTRL3_C, 0x44))              # block data update, register auto-increment
STUCK_READS = 50                        # about 0.24 s; a sampling chip repeated a reading at most 5 times in recordings
SETTLE_S = 0.15                         # the gyro takes up to 80 ms to start from power-down


class Lsm6ds33:
    def __init__(self, bus, rate_hz=208.0):
        self.bus = SMBus(bus)
        self.period = 1.0 / rate_hz
        self.configure()
        self.last_gyro_raw = None
        self.same_reads = 0
        self.frozen = False         # the readings stopped changing
        self.stuck = 0              # times they did
        self.reconfigured = 0       # times the configuration was found changed and rewritten
        self.next_check = time.monotonic() + 1.0
        self.lock = threading.Lock()
        self.gyro_sum = [0.0, 0.0, 0.0]
        self.accel_sum = [0.0, 0.0, 0.0]
        self.count = 0
        self.last_read = None
        self.errors = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def configure(self):
        for reg, val in CONFIG:
            self.bus.write_byte_data(ADDRESS, reg, val)
        self.settle_until = time.monotonic() + SETTLE_S

    def reset(self):
        """Gyro off, software reset (registers at their defaults), reboot (trimming loaded again, 20 ms), configuration."""
        self.bus.write_byte_data(ADDRESS, CTRL2_G, 0x00)
        self.bus.write_byte_data(ADDRESS, CTRL3_C, 0x01)
        time.sleep(0.01)
        self.bus.write_byte_data(ADDRESS, CTRL3_C, 0x80)
        time.sleep(0.03)
        self.configure()

    def read_once(self):
        """One read of the output registers (and, once a second, the configuration check or the reset of a stuck chip)."""
        try:
            now = time.monotonic()
            if now >= self.next_check:
                self.next_check = now + 1.0
                if self.frozen:
                    self.reset()
                elif any(self.bus.read_byte_data(ADDRESS, reg) != val for reg, val in CONFIG):
                    self.reconfigured += 1
                    self.configure()
            raw = self.bus.read_i2c_block_data(ADDRESS, OUTX_L_G, 12)
            d = struct.unpack('<6h', bytes(raw))
            self.same_reads = self.same_reads + 1 if d[:3] == self.last_gyro_raw else 0
            self.last_gyro_raw = d[:3]
            settled = now >= self.settle_until
            if not self.frozen and self.same_reads >= STUCK_READS:
                self.frozen = True
                self.stuck += 1
                self.next_check = now           # reset at the next read
            elif self.frozen and settled and self.same_reads == 0:
                self.frozen = False
            if self.frozen or not settled:
                return
            with self.lock:
                for k in range(3):
                    self.gyro_sum[k] -= d[k] * GYRO_LSB
                    self.accel_sum[k] += d[3 + k] * ACCEL_LSB
                self.count += 1
                self.last_read = time.monotonic()
        except OSError:
            self.errors += 1        # a glitch on the I2C bus: the next read goes on

    def _loop(self):
        next_t = time.monotonic()
        while self.running:
            self.read_once()
            next_t += self.period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def take(self):
        """Mean rates (rad/s) and accelerations (m/s^2) since the last call; None without a new sample."""
        with self.lock:
            n = self.count
            if n == 0:
                return None
            gyro = [s / n for s in self.gyro_sum]
            accel = [s / n for s in self.accel_sum]
            self.gyro_sum, self.accel_sum, self.count = [0.0] * 3, [0.0] * 3, 0
        return gyro, accel

    def close(self):
        self.running = False
        self.thread.join(0.5)
        self.bus.close()
