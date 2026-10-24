"""Heading drift of blue robot0 during the translation-only legs of
drive_straight.py, for several recorded runs.

    python3 straight_analysis.py label=file.pkl ...
"""
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import Truth, arr


def legs(d):
    """[(t_start, t_end, kind)] from the driver's phase marks."""
    ph = sorted((p['t'], p['kind']) for _t, p in d.get('phase', []))
    out = []
    for (t, kind), nxt in zip(ph, ph[1:] + [(None, None)]):
        out.append((t, nxt[0], kind))
    return out


def run(d):
    """Per translation leg: heading at the end vs the start (the heading the
    controller is asked to hold), and the largest swing during the leg."""
    T = Truth(d, 'blue/robot0')
    end_err, swing, t_start = [], [], []
    for t0, t1, kind in legs(d):
        if kind != 'goto' or t1 is None:
            continue
        tt = np.linspace(t0, t1, 100)
        dth = np.degrees(T.at(tt, 'th') - T.at(t0, 'th'))
        end_err.append(dth[-1])
        swing.append(np.max(np.abs(dth)))
        t_start.append(t0)
    return np.array(end_err), np.array(swing), np.array(t_start)


def main():
    rows = []
    for arg in sys.argv[1:]:
        label, f = arg.split('=')
        d = pickle.load(open(f, 'rb'))
        end, swing, ts = run(d)
        early = ts - ts[0] < 30
        rows.append((label, end))
        print(f'{label:24s} legs {len(end):2d} | heading error at leg end: mean |.| {np.mean(np.abs(end)):4.2f} deg, '
              f'max {np.max(np.abs(end)):4.2f} (first 30 s {np.mean(np.abs(end[early])):4.2f}, later '
              f'{np.mean(np.abs(end[~early])):4.2f}) | largest swing during a leg: mean {np.mean(swing):4.2f}, max {np.max(swing):4.2f}')
    for label, end in rows:
        print(f'  {label:22s} end errors: ' + ' '.join(f'{v:+.1f}' for v in end))


if __name__ == '__main__':
    main()
