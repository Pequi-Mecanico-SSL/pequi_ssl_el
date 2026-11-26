"""Is the field network there? Listens 5 s for vision and referee multicast on the given interface addresses.
    python3 tools/solo/listen.py $(ip -4 -br addr | awk '{print $3}' | cut -d/ -f1)
The league's standard groups are joined on the given addresses. Additional groups (e.g. a simulator on its own
addresses) come from SOLO_EXTRA_GROUPS="name=addr:port,..." (empty by default) and are listed separately: joined on
the given addresses and, if SOLO_EXTRA_IFACE names an interface, on its address too, with the interface each packet
came in on."""
import os, socket, struct, subprocess, sys, time, select
STANDARD = [("224.5.23.2", 10006, "vision"), ("224.5.23.2", 10005, "vision-legacy"), ("224.5.23.2", 10020, "sim vision (grSim)"),
            ("224.5.23.1", 10003, "referee"), ("224.5.23.2", 10010, "tracker")]
EXTRA = [(a.strip().rsplit(":", 1)[0], int(a.strip().rsplit(":", 1)[1]), n.strip())
         for n, a in (e.split("=", 1) for e in os.environ.get("SOLO_EXTRA_GROUPS", "").split(",") if e.strip())]
EXTRA_IFACE = os.environ.get("SOLO_EXTRA_IFACE", "")
IP_PKTINFO = getattr(socket, "IP_PKTINFO", 8)
ifaces = sys.argv[1:]
extra_if = [l.split()[2].split("/")[0] for l in subprocess.run(["ip", "-4", "-br", "addr", "show", EXTRA_IFACE], capture_output=True,
                                                                 text=True).stdout.splitlines() if len(l.split()) > 2] if EXTRA_IFACE else []
socks = []
for g, p, name in STANDARD + EXTRA:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except Exception: pass
    s.setsockopt(socket.IPPROTO_IP, IP_PKTINFO, 1)
    s.bind(("", p))
    for ip in ifaces + (extra_if if (g, p, name) in EXTRA else []):
        try: s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(g) + socket.inet_aton(ip))
        except OSError as e: print("join failed", g, ip, e)
    socks.append((s, g, p, name))
cnt = {}; t0 = time.time()
while time.time() - t0 < 5:
    r, _, _ = select.select([s for s, *_ in socks], [], [], 0.5)
    for s in r:
        d, anc, _, a = s.recvmsg(65535, 256)
        info = next((x for lv, ty, x in anc if lv == socket.IPPROTO_IP and ty == IP_PKTINFO), None)
        ifn, dst = (socket.if_indextoname(struct.unpack("i", info[:4])[0]), socket.inet_ntoa(info[8:12])) if info else ("?", None)
        k = [(g, p, n) for ss, g, p, n in socks if ss is s][0]
        if dst and dst != k[0]:
            continue                                   # bound to the port only: a packet for another group on this port
        c = cnt.setdefault(k, [0, 0, set(), set()]); c[0] += 1; c[1] += len(d); c[2].add(a[0]); c[3].add(ifn)
for title, groups in [("standard league addresses", STANDARD)] + ([("additional groups (SOLO_EXTRA_GROUPS" + (", " + EXTRA_IFACE + " " + extra_if[0] if extra_if else ", no SOLO_EXTRA_IFACE address" if EXTRA_IFACE else "") + ")", EXTRA)] if EXTRA else []):
    print(title)
    for s, g, p, n in socks:
        if (g, p, n) not in [(x[0], x[1], x[2]) for x in groups]: continue
        c = cnt.get((g, p, n))
        print(f"  {n:20s} {g}:{p}  " + (f"{c[0] / 5:.0f} packets/s, {c[1] / max(c[0], 1):.0f} bytes each, from {sorted(c[2])} on {sorted(c[3])}" if c else "nothing in 5 s"))
