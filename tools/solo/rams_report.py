"""Where a rammed ball goes: release direction and speed from characterize.py ram logs (camera only).
usage: python3 -I tools/solo/rams_report.py LOG.csv|FOLDER [...]          (a folder: every *_ram_*.csv in it)
Per ram: contact = first ball sample 2 cm past its resting spot along the ram line; robot heading = mean camera
heading over the 0.1 s before contact; robot speed = its speed along the line over the 0.1 s before contact; the
ball's release direction = straight-line fit of the ball from 0.05 to 0.45 s after contact (stopping at 0.8 m of
travel or 0.3 m from a wall); exit angle = that direction minus the robot heading (deg, + = counterclockwise); ball
speed = slope of the ball's travel over the same window. Rams with fewer than 6 ball samples in the window are left
out (printed as such), and so are rams whose heading is more than 8 deg off the ram line at contact. Groups by commanded speed (first argument of the ram).
Straight test rams release within a few degrees of the heading; in play the exit error grows with the commanded
sideways speed at contact, so these rams give the floor of the scatter and the ball's speed against the robot's.
"""
import csv, glob, math, os, sys

FX, FY = 2.26, 1.50


def load(p):
    R = []
    with open(p) as f:
        for r in csv.DictReader(f):
            R.append((float(r['t']), r['kind'], float(r['x']), float(r['y']), float(r['th']), float(r['cmd_vx'])))
    return R


def args_of(p):
    a = os.path.basename(p)[:-4].split('_')
    i = a.index('ram')
    v = float(a[i + 1]); d = float(a[i + 2]) if len(a) > i + 2 else 0.0
    return v, d


def fitdir(P):
    """direction of the straight-line fit through rows (t, kind, x, y, ...), oriented first -> last"""
    X = [p[2] for p in P]; Y = [p[3] for p in P]
    mx = sum(X) / len(X); my = sum(Y) / len(Y)
    sxx = sum((x - mx) ** 2 for x in X); syy = sum((y - my) ** 2 for y in Y); sxy = sum((x - mx) * (y - my) for x, y in zip(X, Y))
    a = 0.5 * math.atan2(2 * sxy, sxx - syy)
    if math.cos(a) * (X[-1] - X[0]) + math.sin(a) * (Y[-1] - Y[0]) < 0: a += math.pi
    return a


def wrapd(a): return (a + 180.0) % 360.0 - 180.0


def one(p):
    v_cmd, d = args_of(p)
    R = load(p)
    ux, uy = math.cos(math.radians(d)), math.sin(math.radians(d))
    B = [r for r in R if r[1] == 'b']; Rb = [r for r in R if r[1] == 'r']
    on = [r[0] for r in R if r[5] > 0.05]
    if not B or not on: return dict(file=os.path.basename(p), skip='no ball or no command')
    ts = min(on)
    rest = [b for b in B if ts - 1.0 <= b[0] <= ts] or B[:5]
    bx0 = sorted(b[2] for b in rest)[len(rest) // 2]; by0 = sorted(b[3] for b in rest)[len(rest) // 2]
    along = lambda b: (b[2] - bx0) * ux + (b[3] - by0) * uy
    hit = next((b[0] for b in B if b[0] >= ts and along(b) > 0.02), None)
    if hit is None: return dict(file=os.path.basename(p), skip='ball never moved 2 cm')
    pre = [r for r in Rb if hit - 0.1 <= r[0] <= hit]
    if len(pre) < 3: return dict(file=os.path.basename(p), skip='robot not seen at contact')
    sx = sum(math.sin(r[4]) for r in pre); cx = sum(math.cos(r[4]) for r in pre); head = math.atan2(sx, cx)
    v_rob = ((pre[-1][2] - pre[0][2]) * ux + (pre[-1][3] - pre[0][3]) * uy) / max(pre[-1][0] - pre[0][0], 1e-3)
    W = []
    for b in B:
        if b[0] < hit + 0.05: continue
        if b[0] > hit + 0.45: break
        if math.hypot(b[2] - bx0, b[3] - by0) > 0.8 or abs(b[2]) > FX - 0.3 or abs(b[3]) > FY - 0.3: break
        W.append(b)
    if len(W) < 6: return dict(file=os.path.basename(p), skip=f'{len(W)} ball samples after contact')
    a = fitdir(W)
    tr = [((b[2] - W[0][2]) * math.cos(a) + (b[3] - W[0][3]) * math.sin(a)) for b in W]
    tm = sum(b[0] for b in W) / len(W); xm = sum(tr) / len(tr)
    vb = sum((b[0] - tm) * (x - xm) for b, x in zip(W, tr)) / max(sum((b[0] - tm) ** 2 for b in W), 1e-9)
    return dict(file=os.path.basename(p), v_cmd=v_cmd, dir=d, v_rob=v_rob, head_vs_line=wrapd(math.degrees(head) - d),
                exit_vs_head=wrapd(math.degrees(a - head)), exit_vs_line=wrapd(math.degrees(a) - d), v_ball=vb, n=len(W))


def stats(xs):
    m = sum(xs) / len(xs); sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float('nan')
    return m, sd


def main(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, '*_ram_*.csv'))) if os.path.isdir(p) else [p]
    res = [one(p) for p in files]
    for r in res:                       # a ram that met the ball crooked says nothing about the release
        if 'skip' not in r and abs(r['head_vs_line']) > 8.0: r['bad'] = f"heading {r['head_vs_line']:+.1f} deg off the line at contact"
    ok = [r for r in res if 'skip' not in r and 'bad' not in r]
    for r in res:
        if 'skip' in r: print(f"{r['file']}: left out ({r['skip']})"); continue
        if 'bad' in r: print(f"{r['file']}: left out ({r['bad']}; exit {r['exit_vs_head']:+.1f} deg)"); continue
        print(f"{r['file']}: cmd {r['v_cmd']:.2f} robot {r['v_rob']:.2f} m/s, ball {r['v_ball']:.2f} m/s "
              f"(x{r['v_ball'] / max(r['v_rob'], 1e-3):.2f}), exit {r['exit_vs_head']:+.1f} deg off the heading "
              f"(heading {r['head_vs_line']:+.1f} off the line), {r['n']} samples")
    if not ok: print('DECIDE: no usable ram'); return
    for v in sorted({r['v_cmd'] for r in ok}) + [None]:
        g = [r for r in ok if v is None or r['v_cmd'] == v]
        if v is None and len({r['v_cmd'] for r in ok}) < 2: break
        m, sd = stats([r['exit_vs_head'] for r in g]); mb, sdb = stats([r['v_ball'] for r in g]); mr, _ = stats([r['v_rob'] for r in g])
        ratio = stats([r['v_ball'] / r['v_rob'] for r in g if r['v_rob'] > 0.1])[0]
        print(f"{'all' if v is None else f'cmd {v:.2f}'}: n {len(g)}, exit vs heading mean {m:+.1f} sd {sd:.1f} deg; "
              f"robot {mr:.2f} m/s, ball {mb:.2f} sd {sdb:.2f} m/s, ball/robot {ratio:.2f}")
    if len(ok) >= 3:
        xs = [r['v_rob'] for r in ok]; ys = [r['v_ball'] for r in ok]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 1e-6:
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            print(f"ball speed = {my - b * mx:+.2f} + {b:.2f} x robot speed (m/s, {len(ok)} rams)")
    m, sd = stats([r['exit_vs_head'] for r in ok])
    if len(ok) < 2 or sd != sd:
        print(f"DECIDE: one usable ram (exit {m:+.1f} deg): not enough for a mean and sd"); return
    print(f"DECIDE: release exit vs heading mean {m:+.1f} deg, sd {sd:.1f} deg (n {len(ok)}, straight rams: the floor of the "
          f"scatter in play); no strategy parameter turns the ram by a fixed angle (aim points only)")

if __name__ == '__main__':
    main(sys.argv[1:])
