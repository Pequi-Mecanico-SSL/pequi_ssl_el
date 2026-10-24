"""Analysis of a recorded scenario (pickle from recorder.py) against sim truth.

    python3 analyze.py <rec.pkl> [plot_prefix]
"""
import math
import os
import pickle
import sys

import numpy as np

COLORS = ('blue', 'yellow')
ROBOTS = [f'{c}/robot{i}' for c in COLORS for i in range(3)]


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def arr(d, k):
    return np.array(d.get(k, []), dtype=float)


def rms(e):
    e = np.asarray(e)
    return float(np.sqrt(np.mean(np.square(e)))) if e.size else float('nan')


def med(e):
    e = np.asarray(e)
    return float(np.median(np.abs(e))) if e.size else float('nan')


def robot_columns(k):
    c, i = k.split('/robot')
    j = 2 + 5 + 11 * (int(i) + (3 if c == 'yellow' else 0))
    return j


class Truth:
    """Sim truth for one robot (from /sim_harness/truth), interpolable."""

    def __init__(self, d, k):
        a = arr(d, 'truth')
        a = a[np.argsort(a[:, 1])]                 # by simulator step
        j = robot_columns(k)
        self.t = a[:, 0]
        self.x, self.y = a[:, j], a[:, j + 1]
        self.th = np.unwrap(np.radians(a[:, j + 2]))
        self.vxg, self.vyg, self.w = a[:, j + 3], a[:, j + 4], np.radians(a[:, j + 5])
        c, s = np.cos(self.th), np.sin(self.th)
        self.vxb, self.vyb = c * self.vxg + s * self.vyg, -s * self.vxg + c * self.vyg
        self.speed = np.hypot(self.vxg, self.vyg)
        dt = 0.016
        ax, ay = np.gradient(self.vxg, dt), np.gradient(self.vyg, dt)
        self.acc = np.hypot(ax, ay)

    def at(self, t, field):
        return np.interp(t, self.t, getattr(self, field))

    def inside(self, t):
        return (t >= self.t[0]) & (t <= self.t[-1])


def vision(d):
    print('\n--- vision: raw (connector) vs filtered (vision_filter), against sim truth ---')
    print(f'{"robot":13s} {"pos med raw/filt (mm)":>22s} {"pos RMS raw/filt (mm)":>22s} '
          f'{"heading med (deg)":>18s} {"vel med raw-diff/filt (m/s)":>28s} {"moving %":>8s}')
    agg = {k: [] for k in ('rp', 'fp', 'rv', 'fv', 'rth', 'fth', 'fp_mov', 'rp_mov', 'fv_mov', 'fv_still', 'fp_still', 'rp_still')}
    for k in ROBOTS:
        T = Truth(d, k)
        raw, filt = arr(d, f'raw/{k}'), arr(d, f'filt/{k}')
        if not len(raw) or not len(filt):
            continue
        raw = raw[T.inside(raw[:, 0])]
        filt = filt[T.inside(filt[:, 0])]
        rp = np.hypot(raw[:, 1] - T.at(raw[:, 0], 'x'), raw[:, 2] - T.at(raw[:, 0], 'y'))
        rth = wrap(raw[:, 3] - T.at(raw[:, 0], 'th'))
        fp = np.hypot(filt[:, 1] - T.at(filt[:, 0], 'x'), filt[:, 2] - T.at(filt[:, 0], 'y'))
        fth = wrap(filt[:, 3] - T.at(filt[:, 0], 'th'))
        fv = np.hypot(filt[:, 4] - T.at(filt[:, 0], 'vxb'), filt[:, 5] - T.at(filt[:, 0], 'vyb'))
        # naive velocity: difference consecutive raw poses (global frame)
        dt = np.diff(raw[:, 0])
        ok = dt > 1e-3
        vxr = np.diff(raw[:, 1])[ok] / dt[ok]
        vyr = np.diff(raw[:, 2])[ok] / dt[ok]
        tr = raw[1:, 0][ok]
        rv = np.hypot(vxr - T.at(tr, 'vxg'), vyr - T.at(tr, 'vyg'))
        moving_f = T.at(filt[:, 0], 'speed') > 0.1
        moving_r = T.at(raw[:, 0], 'speed') > 0.1
        print(f'{k:13s} {1000 * med(rp):9.1f} / {1000 * med(fp):5.1f}       '
              f'{1000 * rms(rp):8.1f} / {1000 * rms(fp):5.1f}      '
              f'{math.degrees(med(rth)):5.2f} / {math.degrees(med(fth)):5.2f}      '
              f'{med(rv):8.3f} / {med(fv):5.3f}            {100 * moving_f.mean():5.1f}')
        agg['rp'] += list(rp); agg['fp'] += list(fp); agg['rv'] += list(rv); agg['fv'] += list(fv)
        agg['rth'] += list(rth); agg['fth'] += list(fth)
        agg['fp_mov'] += list(fp[moving_f]); agg['rp_mov'] += list(rp[moving_r])
        agg['fp_still'] += list(fp[~moving_f]); agg['rp_still'] += list(rp[~moving_r])
        agg['fv_mov'] += list(fv[moving_f]); agg['fv_still'] += list(fv[~moving_f])
    print(f'{"ALL":13s} {1000 * med(agg["rp"]):9.1f} / {1000 * med(agg["fp"]):5.1f}       '
          f'{1000 * rms(agg["rp"]):8.1f} / {1000 * rms(agg["fp"]):5.1f}      '
          f'{math.degrees(med(agg["rth"])):5.2f} / {math.degrees(med(agg["fth"])):5.2f}      '
          f'{med(agg["rv"]):8.3f} / {med(agg["fv"]):5.3f}')
    print(f'moving robots (>0.1 m/s): pos median raw {1000 * med(agg["rp_mov"]):.1f} mm, '
          f'filtered {1000 * med(agg["fp_mov"]):.1f} mm; velocity median filtered {med(agg["fv_mov"]):.3f} m/s')
    print(f'still robots:             pos median raw {1000 * med(agg["rp_still"]):.1f} mm, '
          f'filtered {1000 * med(agg["fp_still"]):.1f} mm; velocity median filtered {med(agg["fv_still"]):.3f} m/s')
    return agg


def collisions(d, k, T):
    """Truth impact moments for robot k: big accelerations, with what was nearby."""
    others = {o: Truth(d, o) for o in ROBOTS if o != k}
    idx = np.where(T.acc > 8.0)[0]
    hits = []
    last = -1e9
    for i in idx:
        t = T.t[i]
        if t - last < 0.3:
            continue
        last = t
        near = []
        for o, To in others.items():
            dist = math.hypot(T.x[i] - To.at(t, 'x'), T.y[i] - To.at(t, 'y'))
            if dist < 0.25:
                near.append(f'{o} {dist:.2f}m')
        wall = abs(T.x[i]) > 4.5 or abs(T.y[i]) > 3.0
        hits.append((t, T.acc[i], near, wall, T.speed[max(0, i - 3)], T.x[i], T.y[i]))
    return hits


def stack(d, prefix=None):
    k = 'blue/robot0'
    T = Truth(d, k)
    odom = arr(d, 'odom')
    if not len(odom):
        return
    odom = odom[T.inside(odom[:, 0])]
    t0 = odom[0, 0]
    print('\n--- stack robot blue/robot0: odometry (gyro + wheels) against sim truth ---')
    # odometry frame starts at 0 where the node started: align with truth at t0
    th_off = T.at(t0, 'th') - odom[0, 3]
    c, s = math.cos(th_off), math.sin(th_off)
    ox = T.at(t0, 'x') + c * odom[:, 1] - s * odom[:, 2] - (c * odom[0, 1] - s * odom[0, 2])
    oy = T.at(t0, 'y') + s * odom[:, 1] + c * odom[:, 2] - (s * odom[0, 1] + c * odom[0, 2])
    th_unw = np.unwrap(odom[:, 3]) + th_off
    herr = np.degrees(th_unw - T.at(odom[:, 0], 'th'))
    perr = np.hypot(ox - T.at(odom[:, 0], 'x'), oy - T.at(odom[:, 0], 'y'))
    verr = np.hypot(odom[:, 4] - T.at(odom[:, 0], 'vxb'), odom[:, 5] - T.at(odom[:, 0], 'vyb'))
    werr = odom[:, 6] - T.at(odom[:, 0], 'w')
    dur = odom[-1, 0] - t0
    path = float(np.sum(np.hypot(np.diff(T.x), np.diff(T.y))))
    rot = float(np.sum(np.abs(np.diff(T.th))))
    print(f'duration {dur:.0f} s, true path {path:.1f} m, true total rotation {math.degrees(rot):.0f} deg')
    for frac in (0.25, 0.5, 0.75, 1.0):
        i = min(len(odom) - 1, int(frac * (len(odom) - 1)))
        print(f'  at {odom[i, 0] - t0:5.0f} s: heading error {herr[i]:+6.2f} deg, position error {perr[i]:.3f} m')
    print(f'  velocity error median {med(verr):.3f} m/s, RMS {rms(verr):.3f}; omega error median '
          f'{med(werr):.3f} rad/s, RMS {rms(werr):.3f}')
    bias, tb = arr(d, 'bias'), arr(d, 'true_bias')
    if len(bias) and len(tb):
        print(f'  gyro bias: final estimate {bias[-1, 1]:+.4f}, true {tb[-1, 1]:+.4f} rad/s')
    big_w = np.abs(werr) > 0.2
    if big_w.any():
        print(f'  omega error > 0.2 rad/s in {100 * big_w.mean():.1f}% of samples '
              f'(true |omega| there: median {med(T.at(odom[big_w, 0], "w")):.2f} rad/s)')

    print('\n--- stack robot blue/robot0: robot_events vs sim truth ---')
    hits = collisions(d, k, T)
    print(f'truth: {len(hits)} moments with |accel| > 8 m/s^2 (robot or wall contact):')
    for t, a, near, wall, v_before, x, y in hits:
        print(f'  t={t - t0:6.2f}s at ({x:+.2f}, {y:+.2f}) accel {a:5.1f} m/s^2, speed before {v_before:.2f} m/s, '
              f'{"WALL " if wall else ""}{", ".join(near) if near else ("" if wall else "(no robot within 0.25 m)")}')
    events = d.get('events', [])
    print(f'robot_events published {len(events)}:')
    for t, e in events:
        extra = {kk: v for kk, v in e.items() if kk not in ('event', 't', 'state')}
        print(f'  t={e["t"] - t0:6.2f}s {e["event"]:7s} {e.get("state", ""):5s} {extra}')
    if prefix:
        plot_stack(d, T, odom, t0, herr, werr, prefix)
    return herr, perr


def plot_stack(d, T, odom, t0, herr, werr, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    tt = odom[:, 0] - t0
    ax[0].plot(T.t - t0, T.vxb, 'k--', lw=1, label='truth vx (body)')
    ax[0].plot(tt, odom[:, 4], lw=0.8, label='odometry vx')
    ax[0].set_ylabel('m/s'); ax[0].legend(loc='upper right')
    ax[1].plot(T.t - t0, T.w, 'k--', lw=1, label='truth omega')
    ax[1].plot(tt, odom[:, 6], lw=0.8, label='odometry omega')
    ax[1].set_ylabel('rad/s'); ax[1].legend(loc='upper right')
    ax[2].plot(tt, herr, label='odometry heading error')
    ax[2].set_ylabel('deg'); ax[2].set_xlabel('s'); ax[2].legend(loc='upper left')
    for t, e in d.get('events', []):
        for a in ax:
            a.axvline(e['t'] - t0, color='r', lw=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(prefix + '_stack.png', dpi=100)


def match_summary(d):
    print('\n--- match overview ---')
    for k in ROBOTS:
        T = Truth(d, k)
        path = float(np.sum(np.hypot(np.diff(T.x), np.diff(T.y))))
        print(f'{k:13s} path {path:6.1f} m, max speed {T.speed.max():.2f} m/s, '
              f'moving {100 * (T.speed > 0.1).mean():4.0f}% of the time, x range [{T.x.min():+.2f}, {T.x.max():+.2f}]')
    b = arr(d, 'truth')[:, 2:4]
    if len(b):
        path = float(np.sum(np.hypot(np.diff(b[:, 0]), np.diff(b[:, 1]))))
        print(f'ball          path {path:6.1f} m, x range [{b[:, 0].min():+.2f}, {b[:, 0].max():+.2f}]')


def main():
    with open(sys.argv[1], 'rb') as f:
        d = pickle.load(f)
    prefix = sys.argv[2] if len(sys.argv) > 2 else None
    if any(k.startswith('cmd/') for k in d):
        match_summary(d)
    if any(k.startswith('filt/') for k in d):
        vision(d)
    stack(d, prefix)


if __name__ == '__main__':
    main()
