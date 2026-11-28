"""Rebound of a hand-rolled ball off a robot, from the raw camera recording (tools/solo/record_match.sh). No ROS.
    python3 -I tools/solo/rebound_hand.py RAW.bin.gz [--robot blue 1] [--from HH:MM:SS --to HH:MM:SS] [--min-in 0.3] [--json OUT]
Reads the vision packets (port 10006) of the recording, takes the most confident ball of every camera frame (camera
times t_capture) and, with --robot, that robot's pose (robot off: its tag is still seen). A contact is a time when the
ball centre comes within 0.13 m of the robot centre while approaching at least --min-in m/s, and its velocity relative to
the robot turns by more than 60 deg. Per contact (speeds relative to the robot, m/s):
  v_in     line fit over the 0.10 s before the contact          v_out3   fit over the first 3 frames after it
  v_out04  fit over 0.35-0.45 s after it (0.4 s later)          e3, e04  v_out3 / v_in, v_out04 / v_in
  where    bearing of the contact in the robot's body frame (0 = front pocket, +-90 = sides, 180 = back) and the name
Without --robot every sharp turn of the ball (over 60 deg, ball faster than --min-in before it) is listed, no bearing.
--from/--to: local clock times (the recording's own time stamps are Unix times; local = this computer's time zone)."""
import gzip, json, math, struct, sys, time

HDR = struct.Struct("<dHI")
def read_raw(path):
    with gzip.open(path, "rb") as f:
        try:
            while True:
                h = f.read(HDR.size)
                if len(h) < HDR.size: return
                t, port, n = HDR.unpack(h); d = f.read(n)
                if len(d) < n: return
                yield t, port, d
        except (EOFError, OSError): return

def fields(buf):
    """protobuf wire format -> [(field, wire_type, value)]; value is int, float (fixed64/32 as raw bytes) or bytes"""
    i, out = 0, []
    while i < len(buf):
        key, i = varint(buf, i); fn, wt = key >> 3, key & 7
        if wt == 0: v, i = varint(buf, i)
        elif wt == 1: v = buf[i:i + 8]; i += 8
        elif wt == 2: ln, i = varint(buf, i); v = buf[i:i + ln]; i += ln
        elif wt == 5: v = buf[i:i + 4]; i += 4
        else: raise ValueError("wire type %d" % wt)
        out.append((fn, wt, v))
    return out
def varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7F) << s; s += 7
        if not c & 0x80: return r, i
f32 = lambda v: struct.unpack("<f", v)[0]
f64 = lambda v: struct.unpack("<d", v)[0]

def frames(path, colour=None, rid=None, t_from=None, t_to=None):
    """-> [(t_capture, ball (x, y) or None, robot (x, y, th) or None)] in metres, sorted by camera time"""
    out = []
    for t, port, d in read_raw(path):
        if port != 10006 or (t_from and t < t_from) or (t_to and t > t_to): continue
        det = next((v for fn, wt, v in fields(d) if fn == 1 and wt == 2), None)
        if det is None: continue
        tc, best, rob = None, None, None
        for fn, wt, v in fields(det):
            if fn == 2 and wt == 1: tc = f64(v)
            elif fn == 5 and wt == 2:
                b = {k: x for k, _, x in fields(v)}
                if 3 in b and 4 in b:
                    c = f32(b[1]) if 1 in b else 0.0
                    if best is None or c > best[0]: best = (c, f32(b[3]) / 1000, f32(b[4]) / 1000)
            elif colour and fn == (6 if colour == "yellow" else 7) and wt == 2:
                r = {k: x for k, _, x in fields(v)}
                if r.get(2, 0) == rid and 3 in r and 5 in r: rob = (f32(r[3]) / 1000, f32(r[4]) / 1000, f32(r[5]))
        if tc is not None: out.append((tc, best[1:] if best else None, rob))
    out.sort(key=lambda r: r[0])
    return out

def fit(pts, t0, t1):
    """velocity (vx, vy) and sample count of a line fit over t0..t1 of [(t, x, y)]"""
    w = [p for p in pts if t0 <= p[0] <= t1]
    if len(w) < 3: return None, len(w)
    mt = sum(p[0] for p in w) / len(w); den = sum((p[0] - mt) ** 2 for p in w)
    if den <= 0: return None, len(w)
    sl = lambda k: sum((p[0] - mt) * p[k] for p in w) / den
    return (sl(1), sl(2)), len(w)

def first_n(pts, t0, n):
    w = [p for p in pts if p[0] > t0][:n]
    return fit(w, w[0][0], w[-1][0]) if len(w) >= 3 else (None, len(w))

def contacts(F, min_in=0.3, use_robot=True):
    ball = [(t, b[0], b[1]) for t, b, r in F if b]
    rob = [(t, r[0], r[1], r[2]) for t, b, r in F if r]
    out, last = [], -9.0
    def rpose(t):
        if not rob: return None
        k = min(range(len(rob)), key=lambda i: abs(rob[i][0] - t))
        return rob[k] if abs(rob[k][0] - t) < 0.1 else None
    for i in range(1, len(ball) - 1):
        t = ball[i][0]
        if t - last < 0.3: continue
        if use_robot:
            r = rpose(t)
            if r is None or math.hypot(ball[i][1] - r[1], ball[i][2] - r[2]) > 0.13: continue
        vb, _ = fit(ball, t - 0.10, t - 0.01); va, _ = fit(ball, t + 0.01, t + 0.06)
        if vb is None or va is None: continue
        vr = fit([(p[0], p[1], p[2]) for p in rob], t - 0.10, t - 0.01)[0] if use_robot else (0.0, 0.0)
        vr = vr or (0.0, 0.0)
        rb = (vb[0] - vr[0], vb[1] - vr[1]); ra = (va[0] - vr[0], va[1] - vr[1])
        sin_, sout = math.hypot(*rb), math.hypot(*ra)
        if sin_ < min_in: continue
        turn = math.degrees(math.acos(max(-1, min(1, (rb[0] * ra[0] + rb[1] * ra[1]) / (sin_ * sout))))) if sout > 1e-6 else 180.0
        if turn < 60: continue
        # refine: the contact is the sample of closest approach (robot) or of the sharpest turn, within +-0.05 s
        win = [p for p in ball if abs(p[0] - t) <= 0.05]
        if use_robot:
            r = rpose(t); tc = min(win, key=lambda p: math.hypot(p[1] - r[1], p[2] - r[2]))[0]
        else: tc = t
        v_in, n_in = fit(ball, tc - 0.10, tc - 0.005); v3, n3 = first_n(ball, tc, 3); v04, n04 = fit(ball, tc + 0.35, tc + 0.45)
        rel = lambda v: math.hypot(v[0] - vr[0], v[1] - vr[1]) if v else None
        row = dict(t=round(tc, 3), clock=time.strftime("%H:%M:%S", time.localtime(tc)), v_in=round(rel(v_in), 3) if v_in else None,
                   v_out3=round(rel(v3), 3) if v3 else None, v_out04=round(rel(v04), 3) if v04 else None, n_in=n_in, n_out3=n3, n_out04=n04)
        if row["v_in"]:
            row["e3"] = round(row["v_out3"] / row["v_in"], 3) if row["v_out3"] is not None else None
            row["e04"] = round(row["v_out04"] / row["v_in"], 3) if row["v_out04"] is not None else None
        if use_robot:
            r = rpose(tc); bp = next(p for p in ball if p[0] == tc)
            br = math.degrees(math.atan2(bp[2] - r[2], bp[1] - r[1]) - r[3]); br = (br + 180) % 360 - 180
            row.update(where=round(br), part="pocket/front" if abs(br) <= 25 else "lip land" if abs(br) <= 45 else
                       "side" if abs(br) <= 135 else "back", robot_moving=round(math.hypot(*vr), 2),
                       d_min=round(math.hypot(bp[1] - r[1], bp[2] - r[2]), 3))
        out.append(row); last = tc
    return out

def clock(s, day_of):
    """HH:MM:SS on the local day of the Unix time day_of -> Unix time"""
    lt = time.localtime(day_of); h, m, sec = (int(x) for x in s.split(":"))
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, sec, 0, 0, -1))

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: sys.exit(__doc__)
    opt = lambda k, n=1: (a[a.index(k) + 1:a.index(k) + 1 + n]) if k in a else None
    path = a[0]; rb = opt("--robot", 2); mi = float((opt("--min-in") or [0.3])[0]); js = opt("--json")
    t_first = next(read_raw(path))[0]
    tf = clock(opt("--from")[0], t_first) if opt("--from") else None; tt = clock(opt("--to")[0], t_first) if opt("--to") else None
    F = frames(path, rb[0] if rb else None, int(rb[1]) if rb else None, tf, tt)
    nb = sum(1 for f in F if f[1]); nr = sum(1 for f in F if f[2])
    print(f"{len(F)} frames, ball in {nb}, robot in {nr}" + ("" if rb else " (no --robot: sharp turns of the ball only)"))
    C = contacts(F, mi, bool(rb))
    for c in C: print(json.dumps(c))
    if js: open(js[0], "a").write("".join(json.dumps(dict(c, file=path.split('/')[-1])) + "\n" for c in C))
    for part in sorted({c.get("part", "-") for c in C}):
        g = [c for c in C if c.get("part", "-") == part and c.get("e3") is not None]
        if g: print(f"{part:13s} n={len(g)}  e3 {sum(c['e3'] for c in g) / len(g):.2f} (range {min(c['e3'] for c in g):.2f}-{max(c['e3'] for c in g):.2f})"
                    + (f"  e04 {sum(c['e04'] for c in g if c['e04'] is not None) / max(1, sum(1 for c in g if c['e04'] is not None)):.2f}"))
