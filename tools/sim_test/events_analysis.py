"""robot_events detections vs simulator truth for blue robot0.

    python3 events_analysis.py <rec.pkl>

For every published event: the driver phase it fell in (scripted runs) and
what the truth shows around it: peak acceleration (+-0.1 s), distance to the
nearest robot, whether it was at a wall, commanded vs true speed.
Also lists truth moments a detector should have caught.
"""
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ROBOTS, Truth, arr


def context(d, T, others, t):
    m = (T.t > t - 0.1) & (T.t < t + 0.1)
    acc = float(T.acc[m].max()) if m.any() else float('nan')
    x, y = T.at(t, 'x'), T.at(t, 'y')
    near = min(((math.hypot(x - To.at(t, 'x'), y - To.at(t, 'y')), o) for o, To in others.items()))
    wall = abs(x) > 4.45 or abs(y) > 2.95
    cc = arr(d, 'cmd_corr')
    i = np.clip(np.searchsorted(cc[:, 0], t) - 1, 0, len(cc) - 1) if len(cc) else None
    cmd = math.hypot(cc[i, 1], cc[i, 2]) if i is not None else float('nan')
    return acc, near, wall, cmd, T.at(t, 'speed')


def phase_at(d, t):
    ph = sorted((p['t'], p['kind']) for _t, p in d.get('phase', []) if p['kind'] != 'skid_injected')
    name = '-'
    for tp, kind in ph:
        if tp <= t:
            name = kind
    return name


def main():
    d = pickle.load(open(sys.argv[1], 'rb'))
    T = Truth(d, 'blue/robot0')
    others = {o: Truth(d, o) for o in ROBOTS if o != 'blue/robot0'}
    t0 = T.t[0]
    inj = [p['t'] for _t, p in d.get('phase', []) if p['kind'] == 'skid_injected']
    if inj:
        print(f'skid injected at {inj[0] - t0:.2f} s')
    print(f'{"t":>7s} {"phase":11s} {"event":28s} | truth: {"peak acc":>8s} {"nearest robot":>20s} {"wall":>4s} {"cmd/true speed":>15s}')
    for _ta, e in d.get('events', []):
        t = e['t']
        acc, (dist, who), wall, cmd, spd = context(d, T, others, t)
        desc = e['event'] + (' ' + e['state'] if 'state' in e else '')
        if e['event'] == 'impact':
            desc += f" {e['kind'][:3]} " + (f"{e['accel']:.0f}m/s2 {e['direction_deg']:+.0f}deg" if e['kind'] == 'linear'
                                            else f"{e['ang_accel']:.0f}rad/s2")
        if e['event'] == 'skid':
            desc += f" w{e['wheel']} {e['candidates']}"
        print(f'{t - t0:7.2f} {phase_at(d, t):11s} {desc:28s} | {acc:8.1f} {who:>13s} {dist:5.2f}m '
              f'{"yes" if wall else "":>4s} {cmd:6.2f} / {spd:4.2f}')
    print('--- truth moments with |accel| > 8 m/s^2 (0.3 s apart) ---')
    last = -1e9
    for i in np.where(T.acc > 8.0)[0]:
        t = T.t[i]
        if t - last < 0.3:
            continue
        last = t
        acc, (dist, who), wall, cmd, spd = context(d, T, others, t)
        print(f'{t - t0:7.2f} {phase_at(d, t):11s} accel {acc:5.1f} m/s^2, nearest {who} {dist:.2f} m{", WALL" if wall else ""}')


if __name__ == '__main__':
    main()
