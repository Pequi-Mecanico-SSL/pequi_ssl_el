"""Vision coverage map: where on and around the field the overhead camera actually detects robots.

    V="python3 tools/solo/vision_map.py"                       # numpy; protobuf for add/live/replay; matplotlib for --png
    $V add $SOLO_DATA/raw/raw_*.bin.gz [--skip WORD]          # seed / extend from recordings (record_raw.py format);
                                                               #   --skip: leave out files whose name contains WORD
    $V live [--iface ADDR] [--every 5]                         # passive multicast listener, rewrites the map; Ctrl-C stops
                                                               #   --iface: address of the interface to join on
                                                               #   (default 0.0.0.0, the kernel's choice)
    $V show [--png FILE]                                       # picture + numbers per region
    $V query X Y                                               # rate and evidence at a point (metres, vision frame)
    $V reset                                                   # moves the old map aside (map.npz.bak-<time>)
    $V replay FILE                                             # test: a recording through the live code path
  every command takes [--map PATH] (default $SOLO_DATA/vision_map/map.npz, SOLO_DATA default: log/solo in this
  repository; map.json next to it; the folder is created when the map is first written).
  Adding a recording again (e.g. the one still being written) only adds the part not yet in the map.
  The vision messages come from this repository's ssl_vision_connector package (installed, after sourcing
  tools/solo/env.sh, or else from src/ssl_vision_connector next to this file; protobuf >= 5.28 for the generated
  modules). SOLO_VISION_PB_DIR names a folder with other generated modules (messages_robocup_ssl_wrapper_pb2 or
  ssl_vision_wrapper_pb2) to use instead.

Grid: cells of 0.10 m over x -3.1..3.1, y -2.4..2.4 (the 4.52 x 3.0 m field plus 0.84 m / 0.9 m beyond every line),
in ssl-vision coordinates (metres, field centre at 0; no side flip). Per cell: E = frames in which a robot was expected
there, H = frames in which it was detected there; rate = H / E, evidence = E (frames, about 62 per second).

Counting rule ("expected"):
  Every robot (colour + id, both colours) is a track. One detection per track per frame (the one nearest the track when
  ssl-vision reports duplicates). A detection that jumps away from the track and is not confirmed by >= 2 of the next
  detections within 0.5 s is dropped as a ghost; a confirmed jump starts a new segment.
  1. Inside a segment, a detection is a hit (E+1, H+1) at its cell. Two consecutive detections belong to the same
     segment if they are <= 1.0 s apart and the move is possible (<= 0.15 + 3 m/s * dt), or <= 5 s apart and less
     than 0.25 m apart (a robot standing in a blind spot; in the recordings this was set on, such losses in a corner
     lasted <= 2.3 s, while losses of 10-20 s of a robot standing mid-field came from people standing over it, not
     from the camera). Every frame between them is a miss (E+1) at the position
     interpolated in time between the two detections.
  2. When a track is lost for longer (or reappears somewhere else, or never), only a bounded number of the missing
     frames are misses: after the last detection, up to 0.5 s of frames if the robot was moving (> 0.15 m/s over its
     last 0.25 s), placed along its last velocity (capped at 2 m/s, so at most 1 m ahead), or up to 0.1 s at its last
     position if it was standing still; symmetrically, before a reappearance, up to 0.5 s (moving) / 0.1 s (still)
     back along its first velocity. The rest of the gap counts nowhere: a robot lifted off by hand adds at most 0.1 s
     of misses where it stood, and a robot driven into a blind zone paints the 0.5 s of path it was heading along.
     Segments shorter than 5 detections get no end attribution, and nothing is attributed before the first frame of a
     recording, or after the last one unless the track was already lost for > 1 s.
  3. Stationary cap: samples are grouped into visits (a track's stay in one cell; a visit ends after 2 s away from
     the cell, and lasts at most 2 minutes). A visit adds at most 3 s of evidence (about 186 frames) to its cell; a
     longer visit is scaled down, keeping its own hit rate. A parked robot adds at most 3 s per 2 minutes to its cell.
     (The 2-minute limit matches the live listener's 2-minute counting chunks, so live and recordings count alike.)
  Data from a time already in the map (a recording added twice, a growing recording added again, or a recording of a
  period the live listener already counted) is skipped: map.json keeps the covered time intervals.

Import (numpy only; tolerant of a file being rewritten, which is atomic via os.replace anyway):
    from vision_map import load
    m = load()                       # or load(path); m.reload() refreshes when the file changed
    m.rate(x, y)                     # detection rate in [0, 1], nan without evidence
    m.evidence(x, y)                 # expected frames at that cell (0 outside the grid)
    m.nearest_good(x, y, min_rate=0.9, min_evidence=30)   # -> (x, y): nearest cell centre with rate >= min_rate,
          # evidence >= min_evidence and the 3x3 neighbourhood also >= min_rate; cells inside the field first.
          # With no such cell, the point clamped into the field shrunk by 0.3 m.
"""
import json, math, os, sys, time

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DEFAULT_MAP = os.path.join(os.environ.get("SOLO_DATA", os.path.join(REPO, "log", "solo")), "vision_map", "map.npz")
GEN = os.environ.get("SOLO_VISION_PB_DIR")          # optional folder of generated vision modules (see the docstring)
CELL = 0.10
X0, X1, Y0, Y1 = -3.1, 3.1, -2.4, 2.4
NX, NY = int(round((X1 - X0) / CELL)), int(round((Y1 - Y0) / CELL))
FIELD = dict(length=4.52, width=3.0, goal_width=0.8, goal_depth=0.18, boundary=0.5,
             penalty_depth=0.4975, penalty_width=1.35)          # this field's geometry packets (metres)

# counting rule parameters (see the docstring)
T_GAP, T_STILL_GAP, D_STILL = 1.0, 5.0, 0.25
V_MAX_GAP = 3.0
T_END_MOVE, T_END_STILL, V_MOVING, V_EXTRAP_MAX, T_VEL = 0.5, 0.1, 0.15, 2.0, 0.25
MIN_SEG = 5
T_LOST_AT_END = 1.0
VISIT_BREAK, VISIT_CAP_S, VISIT_MAX = 2.0, 3.0, 120.0


# ------------------------------------------------------------------ the importable part (numpy only)
class VisionMap:
    def __init__(self, path=DEFAULT_MAP):
        self.path = path if path.endswith(".npz") else os.path.join(path, "map.npz")
        self.E = np.zeros((NX, NY)); self.H = np.zeros((NX, NY)); self.T = np.zeros((NX, NY))
        self.x0, self.y0, self.cell, self.nx, self.ny = X0, Y0, CELL, NX, NY
        self.field = dict(FIELD); self.mtime = None; self.ok = False
        self.reload()

    def reload(self):
        """re-read the file if it changed; on any read error keep the previous data (returns True if loaded)"""
        for attempt in range(5):
            try:
                mt = os.stat(self.path).st_mtime_ns
                if mt == self.mtime:
                    return True
                with np.load(self.path, allow_pickle=False) as z:
                    E, H, T = z["expected"].astype(float), z["detected"].astype(float), z["t_last"].astype(float)
                    g = z["grid"]; f = z["field"]
                self.E, self.H, self.T = E, H, T
                self.x0, self.y0, self.cell = float(g[0]), float(g[1]), float(g[2])
                self.nx, self.ny = E.shape
                self.field.update(dict(zip(("length", "width", "goal_width", "goal_depth", "boundary", "penalty_depth", "penalty_width"),
                                           map(float, f))))
                self.mtime = mt; self.ok = True
                return True
            except FileNotFoundError:
                return False
            except Exception:
                time.sleep(0.02 * (attempt + 1))
        return False

    def _idx(self, x, y):
        i, j = int(math.floor((x - self.x0) / self.cell)), int(math.floor((y - self.y0) / self.cell))
        return (i, j) if 0 <= i < self.nx and 0 <= j < self.ny else None

    def centre(self, i, j):
        return self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell

    def rate(self, x, y):
        k = self._idx(x, y)
        if k is None or self.E[k] < 1e-9:
            return float("nan")
        return float(min(1.0, self.H[k] / self.E[k]))

    def evidence(self, x, y):
        k = self._idx(x, y)
        return 0.0 if k is None else float(self.E[k])

    def rate_grid(self, min_evidence=1.0):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(self.E >= max(min_evidence, 1e-9), np.minimum(self.H / np.maximum(self.E, 1e-9), 1.0), np.nan)
        return r

    def pooled_rate_grid(self):
        """3x3 evidence-weighted rate"""
        E = np.pad(self.E, 1); H = np.pad(self.H, 1)
        sE = sum(E[1 + a:1 + a + self.nx, 1 + b:1 + b + self.ny] for a in (-1, 0, 1) for b in (-1, 0, 1))
        sH = sum(H[1 + a:1 + a + self.nx, 1 + b:1 + b + self.ny] for a in (-1, 0, 1) for b in (-1, 0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sE > 1e-9, np.minimum(sH / np.maximum(sE, 1e-9), 1.0), np.nan)

    def inside_mask(self, shrink=0.0):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell; ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        L, W = self.field["length"] / 2 - shrink, self.field["width"] / 2 - shrink
        return (np.abs(xs)[:, None] <= L) & (np.abs(ys)[None, :] <= W)

    def nearest_good(self, x, y, min_rate=0.9, min_evidence=30):
        r = self.rate_grid(min_evidence); p = self.pooled_rate_grid()
        good = (r >= min_rate) & (p >= min_rate)
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell; ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        d2 = (xs[:, None] - x) ** 2 + (ys[None, :] - y) ** 2
        for mask in (good & self.inside_mask(), good):
            if mask.any():
                i, j = np.unravel_index(np.argmin(np.where(mask, d2, np.inf)), d2.shape)
                return round(float(xs[i]), 3), round(float(ys[j]), 3)
        L, W = self.field["length"] / 2 - 0.3, self.field["width"] / 2 - 0.3
        return round(float(min(max(x, -L), L)), 3), round(float(min(max(y, -W), W)), 3)


def load(path=DEFAULT_MAP):
    return VisionMap(path)


# ------------------------------------------------------------------ counting
def _cons(a, b):
    dt = b[1] - a[1]; d = math.hypot(b[2] - a[2], b[3] - a[3])
    return (dt <= T_GAP and d <= 0.15 + V_MAX_GAP * dt) or (dt <= T_STILL_GAP and d <= D_STILL)


def _vel(seg, at_end):
    """velocity over the first / last T_VEL seconds of a segment of (fi, t, x, y)"""
    if at_end:
        b = seg[-1]; a = next((d for d in seg if d[1] >= b[1] - T_VEL), seg[0])
    else:
        a = seg[0]; b = next((d for d in reversed(seg) if d[1] <= a[1] + T_VEL), seg[-1])
    dt = b[1] - a[1]
    if dt < 0.08:
        return 0.0, 0.0
    vx, vy = (b[2] - a[2]) / dt, (b[3] - a[3]) / dt
    s = math.hypot(vx, vy)
    if s > V_EXTRAP_MAX:
        vx, vy = vx * V_EXTRAP_MAX / s, vy * V_EXTRAP_MAX / s
    return vx, vy


def track_samples(ft, dets):
    """ft: frame times; dets: [(fi, [(x, y, conf), ...])] sorted by fi -> [(t, x, y, hit)] in time order"""
    nfr = len(ft)
    # one detection per frame, nearest to the track
    pts = []; last = None
    for fi, cands in dets:
        if len(cands) == 1 or last is None:
            c = max(cands, key=lambda c: c[2])
        else:
            c = min(cands, key=lambda c: (c[0] - last[0]) ** 2 + (c[1] - last[1]) ** 2)
        pts.append((fi, ft[fi], c[0], c[1])); last = c
    if not pts:
        return []
    # segments, ghosts dropped
    segs = []; cur = [pts[0]]
    for i in range(1, len(pts)):
        d = pts[i]; lp = cur[-1]
        if _cons(lp, d):
            cur.append(d); continue
        if d[1] - lp[1] > T_STILL_GAP:
            segs.append(cur); cur = [d]; continue
        look = [e for e in pts[i + 1:i + 7] if e[1] <= d[1] + 0.5]
        supp = sum(1 for e in look if _cons(d, e) and not _cons(lp, e))
        if supp >= 2:
            segs.append(cur); cur = [d]
        # else: a ghost, dropped
    segs.append(cur)
    fps_dt = float(np.median(np.diff(ft))) if nfr > 2 else 1 / 62
    out = []
    taken_fwd = 0
    for k, seg in enumerate(segs):
        long_enough = len(seg) >= MIN_SEG
        # misses before the start (reappearance), shared gap with the previous segment's forward misses
        if k > 0 and long_enough:
            fa, fb = segs[k - 1][-1][0] + 1, seg[0][0]
            n = fb - fa - taken_fwd
            if n > 0:
                vx, vy = _vel(seg, at_end=False)
                kmax = int(round((T_END_MOVE if math.hypot(vx, vy) > V_MOVING else T_END_STILL) / fps_dt))
                m = min(kmax, n)
                t0, x0, y0 = seg[0][1], seg[0][2], seg[0][3]
                for fi in range(fb - m, fb):
                    dt = min(t0 - ft[fi], T_END_MOVE)
                    out.append((ft[fi], x0 - vx * dt, y0 - vy * dt, 0))
        # inside the segment
        for a, b in zip(seg, seg[1:]):
            out.append((a[1], a[2], a[3], 1))
            span = b[1] - a[1]
            for fi in range(a[0] + 1, b[0]):
                al = (ft[fi] - a[1]) / span if span > 0 else 0.0
                out.append((ft[fi], a[2] + al * (b[2] - a[2]), a[3] + al * (b[3] - a[3]), 0))
        out.append((seg[-1][1], seg[-1][2], seg[-1][3], 1))
        # misses after the end
        taken_fwd = 0
        if long_enough:
            fa = seg[-1][0] + 1
            if k + 1 < len(segs):
                fb = segs[k + 1][0][0]; n = fb - fa
                budget = (n + 1) // 2 if len(segs[k + 1]) >= MIN_SEG else n
            else:
                fb = nfr; n = fb - fa; budget = n
                if n <= 0 or ft[-1] - seg[-1][1] < T_LOST_AT_END:
                    n = budget = 0
            if n > 0:
                vx, vy = _vel(seg, at_end=True)
                kmax = int(round((T_END_MOVE if math.hypot(vx, vy) > V_MOVING else T_END_STILL) / fps_dt))
                m = min(kmax, budget)
                t0, x0, y0 = seg[-1][1], seg[-1][2], seg[-1][3]
                for fi in range(fa, fa + m):
                    dt = min(ft[fi] - t0, T_END_MOVE)
                    out.append((ft[fi], x0 + vx * dt, y0 + vy * dt, 0))
                taken_fwd = m
    return out


def count(frames, skip=(), fps=62.0):
    """frames: [(t, {key: [(x, y, conf), ...]})] in time order -> (E, H, T) arrays; samples whose time falls in one of
    the `skip` intervals [t0, t1] are not counted"""
    E = np.zeros((NX, NY)); H = np.zeros((NX, NY)); T = np.zeros((NX, NY))
    if not frames:
        return E, H, T
    ft = [f[0] for f in frames]
    tracks = {}
    for fi, (t, d) in enumerate(frames):
        for key, cands in d.items():
            tracks.setdefault(key, []).append((fi, cands))
    cap = VISIT_CAP_S * fps
    skip = sorted(skip)

    def skipped(t):
        return any(a <= t <= b for a, b in skip)

    for key, dets in tracks.items():
        visits = {}                      # cell -> [last_t, e, h]

        def close(c, v):
            w = min(1.0, cap / v[1]) if v[1] > 0 else 0.0
            E[c] += v[1] * w; H[c] += v[2] * w; T[c] = max(T[c], v[0])

        for t, x, y, hit in track_samples(ft, dets):
            if skip and skipped(t):
                continue
            i, j = int(math.floor((x - X0) / CELL)), int(math.floor((y - Y0) / CELL))
            if not (0 <= i < NX and 0 <= j < NY):
                continue
            c = (i, j); v = visits.get(c)
            if v is not None and (t - v[0] > VISIT_BREAK or t - v[3] > VISIT_MAX):
                close(c, v); v = None
            if v is None:
                v = visits[c] = [t, 0, 0, t]
            v[0] = t; v[1] += 1; v[2] += hit
        for c, v in visits.items():
            close(c, v)
    return E, H, T


# ------------------------------------------------------------------ decoding
def _pb():
    """the generated SSL_WrapperPacket module: SOLO_VISION_PB_DIR if set, else this repository's ssl_vision_connector"""
    if GEN:
        if GEN not in sys.path:
            sys.path.insert(0, GEN)
        try:
            import messages_robocup_ssl_wrapper_pb2 as vispb
        except ImportError:
            import ssl_vision_wrapper_pb2 as vispb
        return vispb
    try:
        from ssl_vision_connector.messages import messages_robocup_ssl_wrapper_pb2 as vispb
    except ImportError:
        sys.path.append(os.path.join(REPO, "src", "ssl_vision_connector"))
        from ssl_vision_connector.messages import messages_robocup_ssl_wrapper_pb2 as vispb
    return vispb


def read_raw(path):
    """record_raw.py format; stops quietly at a truncated end (a file still being written)"""
    import gzip, struct
    HDR = struct.Struct("<dHI")
    with gzip.open(path, "rb") as f:
        try:
            while True:
                h = f.read(HDR.size)
                if len(h) < HDR.size:
                    return
                t, port, n = HDR.unpack(h)
                d = f.read(n)
                if len(d) < n:
                    return
                yield t, port, d
        except (EOFError, OSError, Exception):
            return


def decode(w, t, frames, geo):
    """append (t, {key: [(x, y, conf)]}) for a detection wrapper; store geometry in geo"""
    if w.HasField("geometry"):
        f = w.geometry.field
        g = dict(length=f.field_length / 1000, width=f.field_width / 1000, goal_width=f.goal_width / 1000,
                 goal_depth=f.goal_depth / 1000, boundary=f.boundary_width / 1000)
        for l in f.field_lines:
            if l.name == "LeftPenaltyStretch":
                g["penalty_depth"] = round(f.field_length / 2000 - abs(l.p1.x) / 1000, 4)
                g["penalty_width"] = abs(l.p1.y - l.p2.y) / 1000
        if g["length"] > 0:
            geo.update(g)
    if not w.HasField("detection"):
        return
    det = w.detection; d = {}
    for col, rs in (("b", det.robots_blue), ("y", det.robots_yellow)):
        for r in rs:
            d.setdefault((col, r.robot_id), []).append((r.x / 1000, r.y / 1000, r.confidence))
    frames.append((t, d))


# ------------------------------------------------------------------ the map file
def _json_path(p):
    return p[:-4] + ".json"


def read_store(path):
    E = np.zeros((NX, NY)); H = np.zeros((NX, NY)); T = np.zeros((NX, NY))
    meta = dict(grid=dict(x0=X0, y0=Y0, cell=CELL, nx=NX, ny=NY), field=dict(FIELD), covered=[], sources={}, created=time.time())
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            E, H, T = z["expected"].astype(float), z["detected"].astype(float), z["t_last"].astype(float)
    if os.path.exists(_json_path(path)):
        meta.update(json.load(open(_json_path(path))))
    return E, H, T, meta


def write_store(path, E, H, T, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta["updated"] = time.time(); meta["updated_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    f = meta["field"]
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, expected=E.astype(np.float32), detected=H.astype(np.float32), t_last=T,
                        grid=np.array([X0, Y0, CELL]), field=np.array([f[k] for k in ("length", "width", "goal_width", "goal_depth",
                                                                                     "boundary", "penalty_depth", "penalty_width")]))
    os.replace(tmp, path)
    tj = _json_path(path) + f".tmp{os.getpid()}"
    json.dump(meta, open(tj, "w"), indent=1)
    os.replace(tj, _json_path(path))


class Lock:
    def __init__(self, path):
        self.p = path + ".lock"

    def __enter__(self):
        import fcntl
        os.makedirs(os.path.dirname(self.p), exist_ok=True)
        self.f = open(self.p, "w"); fcntl.flock(self.f, fcntl.LOCK_EX); return self

    def __exit__(self, *a):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()


def merge_intervals(iv):
    iv = sorted([list(a) for a in iv]); out = []
    for a in iv:
        if out and a[0] <= out[-1][1] + 0.5:
            out[-1][1] = max(out[-1][1], a[1])
        else:
            out.append(a)
    return out


# ------------------------------------------------------------------ commands
def cmd_add(path, files, skip=None):
    vispb = _pb()
    for fpath in files:
        if skip and skip in os.path.basename(fpath):
            print(f"skip {fpath}: its name contains '{skip}' (--skip)"); continue
        t_start = time.time(); frames = []; geo = {}
        for t, port, d in read_raw(fpath):
            if port != 10006:
                continue
            w = vispb.SSL_WrapperPacket()
            try:
                w.ParseFromString(d)
            except Exception:
                continue
            decode(w, t, frames, geo)
        if not frames:
            print(f"{fpath}: no detection frames"); continue
        with Lock(path):
            E, H, T, meta = read_store(path)
            e, h, tl = count(frames, skip=meta["covered"])
            E += e; H += h; T = np.maximum(T, tl)
            if geo:
                meta["field"].update(geo)
            iv = [frames[0][0], frames[-1][0]]
            meta["covered"] = merge_intervals(meta["covered"] + [iv])
            meta["sources"][os.path.basename(fpath)] = dict(t0=iv[0], t1=iv[1], frames=len(frames), added=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                            new_expected=round(float(e.sum())), new_detected=round(float(h.sum())))
            write_store(path, E, H, T, meta)
        print(f"{os.path.basename(fpath)}: {len(frames)} frames {time.strftime('%H:%M:%S', time.localtime(iv[0]))}-"
              f"{time.strftime('%H:%M:%S', time.localtime(iv[1]))}, added {e.sum():.0f} expected / {h.sum():.0f} detected frames "
              f"({time.time() - t_start:.1f} s)")


class Live:
    """the live map update: frames accumulate in memory; every write recounts the open 2-minute chunk (with 10 s of
    context before it) and swaps this process's previous contribution for the new one in the map file, under a lock,
    so `add` can run at the same time; a chunk is committed for good once 10 s past its end"""
    CHUNK, OVER = 120.0, 10.0

    def __init__(self, path, label):
        self.path, self.label = path, label
        self.frames = []; self.geo = {}; self.chunk0 = None; self.t0 = None; self.P = None

    def push(self, t, w):
        n0 = len(self.frames); decode(w, t, self.frames, self.geo)
        if len(self.frames) > n0 and self.chunk0 is None:
            self.chunk0 = self.t0 = t

    def write(self, now, final=False):
        commit = None
        if self.chunk0 is not None and (final or now - self.chunk0 > self.CHUNK + self.OVER):
            end = now + 1e-3 if final else self.chunk0 + self.CHUNK
            commit = count([f for f in self.frames if f[0] >= self.chunk0 - self.OVER],
                           skip=[[-1e18, self.chunk0 - 1e-6], [end, 1e18]])
            self.chunk0 = end; self.frames = [f for f in self.frames if f[0] >= self.chunk0 - self.OVER]
        z = np.zeros((NX, NY))
        cur = (z, z, z) if final or self.chunk0 is None else count(self.frames, skip=[[-1e18, self.chunk0 - 1e-6]])
        with Lock(self.path):
            E, H, T, meta = read_store(self.path)
            if self.P is not None:
                E -= self.P[0]; H -= self.P[1]
            for part in (commit, cur):
                if part is not None:
                    E += part[0]; H += part[1]; T = np.maximum(T, part[2])
            E = np.maximum(E, 0); H = np.minimum(np.maximum(H, 0), E)
            self.P = (cur[0], cur[1])
            if self.geo:
                meta["field"].update(self.geo)
            if self.t0 is not None and self.frames:
                meta["covered"] = merge_intervals(meta["covered"] + [[self.t0, self.frames[-1][0]]])
                meta["sources"][f"{self.label}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(self.t0))}"] = dict(
                    t0=self.t0, t1=self.frames[-1][0])
            write_store(self.path, E, H, T, meta)
        return E, H


def cmd_live(path, iface, every):
    import select, signal, socket
    vispb = _pb()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except Exception:
        pass
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 21)
    s.bind(("224.5.23.2", 10006))                      # only this group's packets
    joined = []

    def join():
        try:
            s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton("224.5.23.2") + socket.inet_aton(iface))
            joined.append(1); print(f"{time.strftime('%H:%M:%S')} joined 224.5.23.2 on {iface}", flush=True)
        except OSError as e:
            print(f"{time.strftime('%H:%M:%S')} cannot join on {iface} yet ({e}); retrying every 5 s", flush=True)
    join()
    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1)); signal.signal(signal.SIGINT, lambda *a: stop.append(1))
    print(f"{time.strftime('%H:%M:%S')} listening on 224.5.23.2:10006 via {iface} (receive only); map {path}, every {every:.0f} s; Ctrl-C stops", flush=True)
    L = Live(path, "live"); n_pk = n_rb = 0; t_write = time.time(); t_print = 0.0
    while not stop:
        r, _, _ = select.select([s], [], [], 0.5)
        if r:
            d = s.recv(65535); n_pk += 1
            w = vispb.SSL_WrapperPacket()
            try:
                w.ParseFromString(d)
            except Exception:
                continue
            if w.HasField("detection"):
                n_rb += len(w.detection.robots_blue) + len(w.detection.robots_yellow)
            L.push(time.time(), w)
        if time.time() - t_write > every:
            if not joined:
                join()
            E, H = L.write(time.time()); t_write = time.time()
            if time.time() - t_print > 60:
                print(f"{time.strftime('%H:%M:%S')} {n_pk} packets, {n_rb} robot detections so far; map evidence {E.sum() / 62:.0f} s in total", flush=True)
                t_print = time.time()
    L.write(time.time(), final=True)
    print(f"{time.strftime('%H:%M:%S')} stopped after {n_pk} packets ({n_rb} robot detections); map written", flush=True)


def cmd_replay(path, fpath, every):
    """test of the live path: feeds a recording through Live with its recorded times"""
    vispb = _pb(); L = Live(path, "replay"); t_write = None; n = 0; t = 0.0
    for t, port, d in read_raw(fpath):
        if port != 10006:
            continue
        w = vispb.SSL_WrapperPacket()
        try:
            w.ParseFromString(d)
        except Exception:
            continue
        L.push(t, w); n += 1
        if t_write is None:
            t_write = t
        if t - t_write > every:
            L.write(t); t_write = t
    E, H = L.write(t, final=True)
    print(f"replayed {n} packets of {fpath}: map now {E.sum():.0f} expected / {H.sum():.0f} detected frames")


def regions(m, min_e=6):
    """name -> (rate, evidence seconds, share of cells without evidence) for the reporting regions"""
    xs = m.x0 + (np.arange(m.nx) + 0.5) * m.cell; ys = m.y0 + (np.arange(m.ny) + 0.5) * m.cell
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    L, W = m.field["length"] / 2, m.field["width"] / 2
    inside = (np.abs(X) <= L) & (np.abs(Y) <= W)
    band, out = 0.25, 0.6
    R = {"inside the field": inside,
         "inner 0.25 m along +x (right goal line)": inside & (X > L - band),
         "inner 0.25 m along -x (left goal line)": inside & (X < -L + band),
         "inner 0.25 m along +y (top touch line)": inside & (Y > W - band),
         "inner 0.25 m along -y (bottom touch line)": inside & (Y < -W + band),
         "0-0.6 m beyond +x": (X > L) & (X <= L + out) & (np.abs(Y) <= W + out),
         "0-0.6 m beyond -x": (X < -L) & (X >= -L - out) & (np.abs(Y) <= W + out),
         "0-0.6 m beyond +y": (Y > W) & (Y <= W + out) & (np.abs(X) <= L + out),
         "0-0.6 m beyond -y": (Y < -W) & (Y >= -W - out) & (np.abs(X) <= L + out)}
    res = {}
    for k, msk in R.items():
        e = m.E[msk].sum(); h = m.H[msk].sum()
        res[k] = (h / e if e > 0 else float("nan"), e / 62.0, float((m.E[msk] < min_e).mean()), int(msk.sum()))
    return res


def cmd_show(path, png):
    m = load(path)
    if not m.ok:
        sys.exit(f"no map at {path}")
    meta = read_store(path)[3]
    print(f"map {path}: updated {meta.get('updated_local')}, sources: {', '.join(meta['sources'])}")
    print(f"{'region':44s} {'rate':>6s} {'evidence':>9s} {'no-evidence cells':>18s}")
    for k, (r, es, empty, n) in regions(m).items():
        print(f"{k:44s} {100 * r:5.1f}% {es:8.0f}s {100 * empty:7.0f}% of {n:4d}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    f = m.field; L, W = f["length"] / 2, f["width"] / 2
    ext = (m.x0, m.x0 + m.nx * m.cell, m.y0, m.y0 + m.ny * m.cell)
    fig, axs = plt.subplots(1, 2, figsize=(19, 7.6), constrained_layout=True)
    rate = m.rate_grid(min_evidence=6)
    ev = np.where(m.E >= 1, m.E / 62.0, np.nan)
    cmap = plt.get_cmap("RdYlGn").copy(); cmap.set_bad("white")
    im0 = axs[0].imshow(rate.T, origin="lower", extent=ext, cmap=cmap, vmin=0.3, vmax=1.0, interpolation="nearest")
    c2 = plt.get_cmap("Blues").copy(); c2.set_bad("white")
    im1 = axs[1].imshow(ev.T, origin="lower", extent=ext, cmap=c2, norm=LogNorm(vmin=0.1, vmax=max(1.0, np.nanmax(ev) if np.isfinite(ev).any() else 1)),
                        interpolation="nearest")
    for ax in axs:
        ax.add_patch(Rectangle((-L, -W), 2 * L, 2 * W, fill=False, lw=2, ec="black"))
        ax.plot([0, 0], [-W, W], "k-", lw=1)
        ax.add_patch(plt.Circle((0, 0), 0.5, fill=False, lw=1, ec="black"))
        for s in (-1, 1):
            pd, pw = f.get("penalty_depth", 0.5), f.get("penalty_width", 1.35)
            ax.add_patch(Rectangle((s * L - (pd if s > 0 else 0), -pw / 2), pd, pw, fill=False, lw=1.5, ec="black"))
            gd, gw = f["goal_depth"], f["goal_width"]
            ax.add_patch(Rectangle((s * L if s > 0 else -L - gd, -gw / 2), gd, gw, fill=False, lw=3, ec="purple"))
        b = f.get("boundary", 0.5)
        ax.add_patch(Rectangle((-L - b, -W - b), 2 * (L + b), 2 * (W + b), fill=False, lw=1, ls="--", ec="gray"))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        ax.set_xticks(np.arange(-3, 3.01, 0.5)); ax.set_yticks(np.arange(-2, 2.01, 0.5)); ax.grid(alpha=0.25)
        ax.set_xlabel("x (m, vision frame)"); ax.set_ylabel("y (m)")
    rg = regions(m)
    r_in = rg["inside the field"]
    axs[0].set_title(f"Detection rate per 0.1 m cell (blank: < 0.1 s of evidence)\ninside the field {100 * r_in[0]:.0f} %, "
                     f"{100 * r_in[2]:.0f} % of field cells without evidence", fontsize=13)
    axs[1].set_title(f"Evidence: seconds a robot was expected in the cell (log)\nupdated {meta.get('updated_local')}", fontsize=13)
    fig.colorbar(im0, ax=axs[0], shrink=0.85, label="detected / expected frames")
    fig.colorbar(im1, ax=axs[1], shrink=0.85, label="seconds of evidence")
    png = png or os.path.join(os.path.dirname(path), "map.png")
    fig.savefig(png, dpi=90); print("wrote", png)


def cmd_query(path, x, y):
    m = load(path)
    r, e = m.rate(x, y), m.evidence(x, y)
    p = m.pooled_rate_grid(); k = m._idx(x, y)
    print(f"({x:+.2f}, {y:+.2f}): rate {100 * r:.1f} % over {e:.0f} expected frames ({e / 62:.1f} s); 3x3 pooled "
          f"{100 * (p[k] if k else float('nan')):.1f} %; nearest good (rate >= 0.9, >= 30 frames): {m.nearest_good(x, y)}")


def main():
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    path = opt("--map", DEFAULT_MAP)
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
    if not pos:
        sys.exit(__doc__)
    c = pos[0]
    if c == "add":
        cmd_add(path, pos[1:], opt("--skip", None))
    elif c == "live":
        cmd_live(path, opt("--iface", "0.0.0.0"), float(opt("--every", 5)))
    elif c == "replay":
        cmd_replay(path, pos[1], float(opt("--every", 5)))
    elif c == "show":
        cmd_show(path, opt("--png", None))
    elif c == "query":
        cmd_query(path, float(pos[1]), float(pos[2]))
    elif c == "reset":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for p in (path, _json_path(path)):
            if os.path.exists(p):
                os.replace(p, p + ".bak-" + stamp); print("moved aside", p + ".bak-" + stamp)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
