"""LSM6DS33 gyro + accelerometer (Pololu MinIMU-9 v5), read in a background thread.

Axes as published on /imu: the board sits upside down, rates are negated so that
z is positive CCW seen from the top; accelerations are the chip's own axes.
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


class Lsm6ds33:
    def __init__(self, bus, rate_hz=208.0):
        self.bus = SMBus(bus)
        self.period = 1.0 / rate_hz
        w = lambda reg, val: self.bus.write_byte_data(ADDRESS, reg, val)
        w(CTRL10_C, 0x38)       # gyro axes on
        w(CTRL2_G, 0x5C)        # gyro 208 Hz, 2000 dps
        w(CTRL1_XL, 0x58)       # accelerometer 208 Hz, 4 g
        w(CTRL3_C, 0x44)        # block data update, register auto-increment
        self.lock = threading.Lock()
        self.gyro_sum = [0.0, 0.0, 0.0]
        self.accel_sum = [0.0, 0.0, 0.0]
        self.count = 0
        self.last_read = None
        self.errors = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        next_t = time.monotonic()
        while self.running:
            try:
                raw = self.bus.read_i2c_block_data(ADDRESS, OUTX_L_G, 12)
                d = struct.unpack('<6h', bytes(raw))
                with self.lock:
                    for k in range(3):
                        self.gyro_sum[k] -= d[k] * GYRO_LSB
                        self.accel_sum[k] += d[3 + k] * ACCEL_LSB
                    self.count += 1
                    self.last_read = time.monotonic()
            except OSError:
                self.errors += 1        # a glitch on the I2C bus: the next read goes on
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
