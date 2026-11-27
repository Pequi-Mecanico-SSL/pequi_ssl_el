"""Motor CAN bus: the frames of stm32-motor-controller/PROTOCOL.md (firmware 9 and later)."""
import math
import socket
import struct

MAX_MOTORS = 4   # slots in the setpoint and feedforward frames
ID_ESTOP, ID_SETPOINT, ID_FEEDFORWARD, ID_CMD = 0x000, 0x080, 0x081, 0x100
ID_TLM, ID_DIAG, ID_HELLO, ID_STAT = 0x200, 0x280, 0x300, 0x480
(OP_STOP, OP_VEL, OP_ENABLE, OP_DISABLE, OP_CURLIM, OP_TELEM, OP_TERM, OP_PING,
 OP_WATCHDOG, OP_PARAM, OP_DIAG) = range(11)
PARAMS = {'vel_kp': 0, 'vel_ki': 1, 'accel_max': 2, 'ref_hz': 3, 'iq_limit': 4, 'obs_hz': 5,
          'cur_hz': 6, 'inertia_ff': 12, 'cog_gain': 15, 'flat': 16, 'drain': 17, 'i_cont': 20, 'heat_tau': 21, 'vbus_min': 22}
FLAG_ENABLED, FLAG_READY, FLAG_WD_TRIPPED, FLAG_FAULT, FLAG_LIMITED = 1, 2, 4, 8, 0x40
FAULTS = ('none', 'encoder', 'overcurrent', 'undervoltage', 'overvoltage',
          'not calibrated', 'current sense', 'angle (reversed torque: calibrate)')
POS_TO_RAD = 2 * math.pi / 4096
GUARD_FW = 10    # first firmware that lowers its Iq limit by itself when the windings get warm


def ntc_celsius(raw):
    r = 4700.0 * (255.5 / (min(max(raw, 1), 254) + 0.5) - 1.0)
    return 1.0 / (math.log(r / 10000.0) / 3425.0 + 1.0 / 298.15) - 273.15


class EscBus:
    def __init__(self, interface):
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                             b''.join(struct.pack('<II', i, 0x7FC) for i in (ID_TLM, ID_DIAG, ID_HELLO, ID_STAT)))
        self.sock.bind((interface,))
        self.sock.setblocking(False)
        self.send_errors = 0

    def close(self):
        self.sock.close()

    def send(self, can_id, data=b''):
        try:
            self.sock.send(struct.pack('<IB3x8s', can_id, len(data), data.ljust(8, b'\0')))
        except OSError:
            self.send_errors += 1

    def command(self, motor, op, args=b''):
        self.send(ID_CMD + motor, bytes([op]) + args)

    def set_param(self, motor, name, value):
        self.command(motor, OP_PARAM, struct.pack('<Bf', PARAMS[name], value))

    def _slots(self, can_id, values, scale):
        ints = [max(-32767, min(32767, round(v * scale))) for v in values]
        self.send(can_id, struct.pack('<4h', *(ints + [0] * (MAX_MOTORS - len(ints)))))

    def setpoints(self, speeds):
        self._slots(ID_SETPOINT, speeds, 100.0)

    def feedforward(self, currents):
        self._slots(ID_FEEDFORWARD, currents, 1000.0)

    def estop(self):
        self.send(ID_ESTOP)

    def poll(self):
        """Received frames, decoded:
        ('tlm', motor, speed rad/s, angle word, iq A, flags, fault)
        ('diag', motor, bus V, board C, iq reference A, id A, uq V)
        ('stat', motor, encoder parity errors, encoder frames rejected, longest control interrupt us,
                 error of the electrical zero in degrees (None: not measured yet), ud V)
        ('hello', motor, uid, firmware, init flags)"""
        out = []
        while True:
            try:
                raw = self.sock.recv(16)
            except (BlockingIOError, OSError):
                return out
            can_id, dlc, data = struct.unpack('<IB3x8s', raw)
            can_id &= socket.CAN_EFF_MASK
            base, motor = can_id & 0x7FC, can_id & 3
            if dlc < 8:
                continue
            if base == ID_TLM:
                vel, pos, iq, flags, fault = struct.unpack('<hHhBB', data)
                out.append(('tlm', motor, vel / 100.0, pos, iq / 1000.0, flags, fault))
            elif base == ID_DIAG:
                iq_ref, i_d, uq = struct.unpack('<hhh', data[:6])
                out.append(('diag', motor, data[6] * 0.2, ntc_celsius(data[7]), iq_ref / 1000.0, i_d / 1000.0, uq / 100.0))
            elif base == ID_STAT:
                parity, jumps, isr, angle, ud = struct.unpack('<HHBbh', data)
                out.append(('stat', motor, parity, jumps, isr / 4.0, None if angle == -128 else angle * 2, ud / 100.0))
            elif base == ID_HELLO:
                uid, fw, fl, mid, _ = struct.unpack('<IBBBB', data)
                out.append(('hello', mid, uid, fw, fl))
