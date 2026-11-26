"""Record the raw vision and referee multicast packets, exactly as received, for replay later.
    python3 tools/solo/record_raw.py [--out DIR] [--tag NAME]
--out defaults to $SOLO_DATA/raw (SOLO_DATA default: log/solo in this repository; created if missing).
Listens on every interface that is up (and notices interfaces that appear later), so it does not matter which cable or
Wi-Fi carries the field network. Stops cleanly on Ctrl-C / SIGTERM.
File: DIR/raw_<date>_<time>[_NAME].bin.gz, a gzip stream of records  <d t_unix> <H port> <I length> <payload>  (little
endian); port 10006 = SSL_WrapperPacket (vision), 10003 = Referee. A new file every 30 minutes; flushed every 10 s."""
import gzip, os, select, signal, socket, struct, subprocess, sys, time

HDR = struct.Struct("<dHI")
GROUPS = (("224.5.23.2", 10006), ("224.5.23.1", 10003))


def read_raw(path):
    """-> yields (t_unix, port, payload bytes); stops quietly at a truncated end"""
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
        except (EOFError, OSError):
            return


def interface_ips():
    out = subprocess.run(["ip", "-4", "-br", "addr"], capture_output=True, text=True).stdout.splitlines()
    return sorted(l.split()[2].split("/")[0] for l in out
                  if len(l.split()) > 2 and not l.startswith(("lo", "docker", "br-", "tailscale", "wg", "veth")))


if __name__ == "__main__":
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    out = arg("--out", os.path.join(os.environ.get("SOLO_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "log", "solo")), "raw"))
    os.makedirs(out, exist_ok=True)
    tag = arg("--tag", "")
    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    signal.signal(signal.SIGINT, lambda *a: stop.append(1))
    socks = {}
    for g, p in GROUPS:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 21)
        s.bind(("", p)); socks[s] = (g, p)
    joined = set()

    def join():
        for ip in interface_ips():
            if ip in joined:
                continue
            for s, (g, p) in socks.items():
                try:
                    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(g) + socket.inet_aton(ip))
                except OSError:
                    pass
            joined.add(ip)
            print(f"{time.strftime('%H:%M:%S')} listening on {ip}", flush=True)

    f = None; t_file = t_flush = t_print = t_join = 0.0; n = {10006: 0, 10003: 0}; last = {10006: 0.0, 10003: 0.0}
    while not stop:
        now = time.time()
        if now - t_join > 5:
            join(); t_join = now
        if f is None or now - t_file > 1800:
            if f: f.close()
            path = f"{out}/raw_{time.strftime('%Y%m%d_%H%M%S')}{'_' + tag if tag else ''}.bin.gz"
            f = gzip.open(path, "wb", compresslevel=5); t_file = now
            print(f"{time.strftime('%H:%M:%S')} recording to {path}", flush=True)
        r, _, _ = select.select(list(socks), [], [], 1.0)
        for s in r:
            d, _ = s.recvfrom(65535); p = socks[s][1]
            f.write(HDR.pack(time.time(), p, len(d))); f.write(d); n[p] += 1; last[p] = time.time()
        if now - t_flush > 10:
            f.flush(); t_flush = now
        if now - t_print > 60:
            age = lambda p: "never" if not last[p] else f"{now - last[p]:.0f} s ago"
            print(f"{time.strftime('%H:%M:%S')} vision {n[10006]} (last {age(10006)}), referee {n[10003]} (last {age(10003)})", flush=True)
            t_print = now
    if f: f.close()
    print(f"{time.strftime('%H:%M:%S')} stopped: vision {n[10006]}, referee {n[10003]} packets", flush=True)
