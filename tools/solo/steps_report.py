"""Top speed and stopping from characterize.py step logs (camera only) -> keep speed_cap 1.5 or not.
usage: python3 -I tools/solo/steps_report.py LOG.csv|FOLDER [...] [--stop-ok M]   (a folder: every *_step_*.csv in it)
Per step: the command window = rows with a nonzero commanded speed; speed along the run direction (camera heading at the start + the body command's direction) from a 0.06 s
central difference of the camera track; steady = mean over the last 40 % of the window; v_cut = speed at the last
nonzero command (as sent by the laptop); stop = travel from there until the speed stays under 0.05 m/s; t_still the
time it took. Counted from the command sent by the laptop (a count from the moment the robot sees the zero gives
about v x 0.03-0.05 s less).
Reference, the same counting, measured on a carpet field: velocity_controller + kinematics from 1.5 m/s 0.51-0.54 m, from
1.0 0.24-0.27 m, from 0.5 0.08-0.09 m; drive node (2.55 A plan) from 1.5 0.44-0.48 m, from 1.0 0.22-0.23 m, from
0.5 0.06-0.09 m.
--stop-ok (default 0.55 m): the largest stop distance from 1.5 m/s that keeps speed_cap 1.5 (the old controller's,
with which the strategy has played at speed_cap 1.5).
"""
import csv, glob, math, os, sys

REF = {1.5: (0.51, 0.54), 1.0: (0.24, 0.27), 0.5: (0.08, 0.09)}
REF_NEW = {1.5: (0.44, 0.48), 1.0: (0.22, 0.23), 0.5: (0.06, 0.09)}
STOP_OK = 0.55       # m from 1.5 m/s (--stop-ok)


def one(p):
    a = os.path.basename(p)[:-4].split('_'); i = a.index('step')
    axis, sp = a[i + 1], float(a[i + 2])
    R = [(float(r['t']), r['kind'], float(r['x']), float(r['y']), float(r['cmd_vx']), float(r['cmd_vy']), float(r['th']))
         for r in csv.DictReader(open(p))]
    on = [r[0] for r in R if math.hypot(r[4], r[5]) > 1e-4]
    if not on: return None
    # the step's own command: the last unbroken nonzero stretch (the goto before it also moves)
    blocks, cur, last = [], [], None
    for r in R:
        nz = math.hypot(r[4], r[5]) > 1e-4
        if nz: cur.append(r[0])
        elif cur: blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    ts, te = blocks[-1][0], blocks[-1][-1]
    # run direction = camera heading at the start + direction of the body command (no field-frame assumption)
    r0 = next(r for r in R if r[0] >= ts and math.hypot(r[4], r[5]) > 1e-4)
    pre = [r for r in R if r[1] == 'r' and ts - 0.2 <= r[0] <= ts] or [next(r for r in R if r[1] == 'r' and r[0] >= ts)]
    th = math.atan2(sum(math.sin(r[6]) for r in pre), sum(math.cos(r[6]) for r in pre))
    d = th + math.atan2(r0[5], r0[4]); ux, uy = math.cos(d), math.sin(d)
    P = [(r[0], (r[2] * ux + r[3] * uy)) for r in R if r[1] == 'r' and ts - 0.2 <= r[0] <= te + 3.0]
    if len(P) < 20: return None
    V = []
    for k in range(len(P)):
        j0 = next(j for j in range(k, -1, -1) if P[k][0] - P[j][0] >= 0.03 or j == 0)
        j1 = next((j for j in range(k, len(P)) if P[j][0] - P[k][0] >= 0.03), len(P) - 1)
        if P[j1][0] - P[j0][0] > 0.03: V.append((P[k][0], (P[j1][1] - P[j0][1]) / (P[j1][0] - P[j0][0]), P[k][1]))
    win = [v for t, v, _ in V if ts + 0.6 * (te - ts) <= t <= te]
    steady = sum(win) / len(win) if win else float('nan')
    cut = next((v for v in V if v[0] >= te), None)
    if cut is None: return None
    after = [v for v in V if v[0] >= te]
    still = next((k for k in range(len(after)) if all(abs(x[1]) < 0.05 for x in after[k:k + 6])), None)
    if still is None: return dict(file=os.path.basename(p), axis=axis, cmd=sp, steady=steady, v_cut=cut[1], stop=float('nan'), t_still=float('nan'))
    return dict(file=os.path.basename(p), axis=axis, cmd=sp, steady=steady, v_cut=cut[1],
                stop=after[still][2] - cut[2], t_still=after[still][0] - te, run=te - ts)


def main(paths):
    global STOP_OK
    if '--stop-ok' in paths:
        i = paths.index('--stop-ok'); STOP_OK = float(paths[i + 1]); del paths[i:i + 2]
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, '*_step_*.csv'))) if os.path.isdir(p) else [p]
    res = [r for r in map(one, files) if r]
    for r in res:
        print(f"{r['file']}: {r['axis']} cmd {r['cmd']:.2f}: steady {r['steady']:.2f} m/s ({r['steady'] / r['cmd']:.2f} of cmd), "
              f"at cut {r['v_cut']:.2f}, stop {r['stop']:.3f} m in {r['t_still']:.2f} s")
    by = {}
    for r in res: by.setdefault(round(r['cmd'], 2), []).append(r)
    for c in sorted(by):
        g = [r for r in by[c] if r['stop'] == r['stop']]
        if not g: continue
        st = [r['stop'] for r in g]; ss = [r['steady'] for r in g]
        ref = REF.get(c); rn = REF_NEW.get(c)
        print(f"cmd {c:.2f}: n {len(g)}, steady {min(ss):.2f}-{max(ss):.2f} m/s, stop {min(st):.3f}-{max(st):.3f} m (mean {sum(st) / len(st):.3f})"
              + (f"; old controller {ref[0]:.2f}-{ref[1]:.2f}, drive node {rn[0]:.2f}-{rn[1]:.2f}" if ref else ''))
    top = by.get(1.5)
    if not top:
        print("DECIDE: no step at 1.5: speed_cap undecided"); return
    st = max(r['stop'] for r in top if r['stop'] == r['stop']); ss = sum(r['steady'] for r in top) / len(top)
    if st <= STOP_OK:
        print(f"DECIDE: keep speed_cap 1.5 (worst stop from 1.5: {st:.2f} m <= {STOP_OK}; steady {ss:.2f} m/s at a 1.5 command)")
    else:
        cap = 1.5 * math.sqrt(STOP_OK / st)
        print(f"DECIDE: speed_cap {math.floor(cap * 10) / 10:.1f} (worst stop from 1.5: {st:.2f} m > {STOP_OK}; stop grows ~v^2; steady {ss:.2f} m/s)")


if __name__ == '__main__':
    main(sys.argv[1:])
