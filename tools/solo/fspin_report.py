"""Bend of field-frame moves while spinning (characterize.py fspin logs) -> the strategy latency with the straightest path.
usage: python3 -I tools/solo/fspin_report.py LOG.csv [LOG.csv ...]     (or a folder: every *_fspin_*.csv in it)
Per run (camera poses only): the command window = the last unbroken stretch of rows commanding the run's rate; the path from 0.4 s after the
start to the command off; bend = direction of the straight-line fit of that path minus the commanded direction (deg,
+ = counterclockwise), multiplied by the sign of the spin so that both spins count the same way; side = sideways
offset of the end of the window from the commanded line through the start of it (m, same sign rule).
Across runs: least-squares line of bend against latency, its zero = the latency to set.
"""
import csv, glob, math, os, sys


def load(p):
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            rows.append((float(r['t']), r['kind'], float(r['x']), float(r['y']), float(r['th']),
                         float(r['cmd_vx']), float(r['cmd_vy']), float(r['cmd_w'])))
    return rows


def args_of(p):
    """speed, dir_deg, rate, latency from the file name <hhmmss>_fspin_<v>_<dir>_<w>_<lat>[_...].csv"""
    a = os.path.basename(p)[:-4].split('_')
    i = a.index('fspin')
    return float(a[i + 1]), float(a[i + 2]), float(a[i + 3]), float(a[i + 4])


def one(p, skip=0.4):
    v, dirn, w, lat = args_of(p)
    R = load(p)
    # the fspin command: the last unbroken stretch of rows commanding exactly the run's rate (goto turns come before)
    on, run_ = [], []
    for r in R:
        if abs(r[7] - w) < 1e-4: run_.append(r[0])
        elif run_ and r[1] == 'r': on, run_ = run_, []
    on = run_ or on
    if not on: return None
    t0, t1 = min(on), max(on)
    P = [(r[0], r[2], r[3]) for r in R if r[1] == 'r' and t0 + skip <= r[0] <= t1]
    if len(P) < 10: return None
    ux, uy = math.cos(math.radians(dirn)), math.sin(math.radians(dirn))
    # straight-line fit: principal direction of the points, oriented along the motion
    mx = sum(x for _, x, _ in P) / len(P); my = sum(y for _, _, y in P) / len(P)
    sxx = sum((x - mx) ** 2 for _, x, _ in P); syy = sum((y - my) ** 2 for _, _, y in P)
    sxy = sum((x - mx) * (y - my) for _, x, y in P)
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    dx, dy = P[-1][1] - P[0][1], P[-1][2] - P[0][2]
    if math.cos(ang) * dx + math.sin(ang) * dy < 0: ang += math.pi
    sg = 1.0 if w > 0 else -1.0
    bend = sg * math.degrees((ang - math.radians(dirn) + math.pi) % (2 * math.pi) - math.pi)
    side = sg * (-dx * uy + dy * ux)
    along = dx * ux + dy * uy
    speed = along / max(P[-1][0] - P[0][0], 1e-3)
    return dict(file=os.path.basename(p), v=v, dir=dirn, w=w, latency=lat, bend_deg=round(bend, 1),
                side_m=round(side, 3), along_m=round(along, 2), speed=round(speed, 2), n=len(P))


def main(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, '*_fspin_*.csv'))) if os.path.isdir(p) else [p]
    res = [r for r in map(one, files) if r]
    for r in res:
        print(f"{r['file']}: v {r['v']} dir {r['dir']:.0f} w {r['w']:+.1f} latency {r['latency']:.3f} -> bend {r['bend_deg']:+.1f} deg, "
              f"side {r['side_m']:+.3f} m over {r['along_m']} m (speed {r['speed']} m/s, {r['n']} frames)")
    lats = sorted({r['latency'] for r in res})
    if len(lats) < 2:
        if res: print(f"DECIDE: only one latency value ({lats[0]}): mean bend {sum(r['bend_deg'] for r in res) / len(res):+.1f} deg; "
                      "run a second value to get the zero")
        return
    n = len(res); mx = sum(r['latency'] for r in res) / n; my = sum(r['bend_deg'] for r in res) / n
    b = sum((r['latency'] - mx) * (r['bend_deg'] - my) for r in res) / sum((r['latency'] - mx) ** 2 for r in res)
    a = my - b * mx
    resid = math.sqrt(sum((r['bend_deg'] - a - b * r['latency']) ** 2 for r in res) / max(n - 2, 1)) if n > 2 else float('nan')
    z = -a / b if abs(b) > 1e-9 else float('nan')
    print(f"fit: bend = {a:+.1f} {b:+.1f} x latency (deg; {b / 1000:+.2f} deg per ms), residual sd {resid:.1f} deg, {n} runs")
    zc = min(max(z, 0.0), 0.15)
    by = {L: [r['bend_deg'] for r in res if r['latency'] == L] for L in lats}
    best = min(lats, key=lambda L: abs(sum(by[L]) / len(by[L])))
    note = '' if lats[0] <= z <= lats[-1] else ' (outside the tried values: extrapolated)'
    print(f"DECIDE: set latency to {round(zc, 3)}{note}; smallest bend among the tried values: {best} "
          f"({sum(by[best]) / len(by[best]):+.1f} deg)")


if __name__ == '__main__':
    main(sys.argv[1:])
