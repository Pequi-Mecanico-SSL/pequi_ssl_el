"""Paired comparison: vision_filter with vs without the cmd_vel hint, same match."""
import math, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import Truth, arr, med, rms, wrap, ROBOTS

d = pickle.load(open(sys.argv[1], 'rb'))


def errors(rows, T):
    rows = rows[T.inside(rows[:, 0])]
    t = rows[:, 0]
    pos = np.hypot(rows[:, 1] - T.at(t, 'x'), rows[:, 2] - T.at(t, 'y'))
    hd = np.degrees(np.abs(wrap(rows[:, 3] - T.at(t, 'th'))))
    vel = np.hypot(rows[:, 4] - T.at(t, 'vxb'), rows[:, 5] - T.at(t, 'vyb'))
    return t, pos, hd, vel


print(f'{"robot":13s} {"hint ON  pos/heading/vel (median)":>36s} {"hint OFF pos/heading/vel":>30s} {"blocked*":>9s}')
for k in ROBOTS:
    T = Truth(d, k)
    on, off = arr(d, f'filt/{k}'), arr(d, f'filt_nohint/{k}')
    raw = arr(d, f'raw/{k}')
    t1, p1, h1, v1 = errors(on, T)
    t2, p2, h2, v2 = errors(off, T)
    blocked = np.zeros(len(t1), bool)
    cmd = arr(d, f'cmd/{k}')
    if k.startswith('blue') and len(cmd):
        i = np.clip(np.searchsorted(cmd[:, 0], t1) - 1, 0, len(cmd) - 1)
        cmd_speed = np.hypot(cmd[i, 1], cmd[i, 2])
        blocked = (cmd_speed > 0.3) & (T.at(t1, 'speed') < 0.1)
    print(f'{k:13s} {1000*np.median(p1):7.1f} mm {np.median(h1):5.2f} deg {np.median(v1):5.3f} m/s'
          f'   {1000*np.median(p2):7.1f} mm {np.median(h2):5.2f} deg {np.median(v2):5.3f} m/s'
          f'   {100*blocked.mean():5.1f}%')
    if blocked.any():
        b2 = np.interp(t2, t1, blocked.astype(float)) > 0.5
        print(f'{"":13s}   while blocked: hint ON {1000*np.median(p1[blocked]):.1f} mm / {np.median(v1[blocked]):.3f} m/s,'
              f' hint OFF {1000*np.median(p2[b2]):.1f} mm / {np.median(v2[b2]):.3f} m/s;'
              f'   otherwise: ON {1000*np.median(p1[~blocked]):.1f} mm, OFF {1000*np.median(p2[~b2]):.1f} mm')
print('* blocked = commanded faster than 0.3 m/s but really moving slower than 0.1 m/s (blue robots get the hint)')
