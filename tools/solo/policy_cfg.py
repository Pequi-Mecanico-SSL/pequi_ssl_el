#!/usr/bin/env python3
"""The strategy's parameters, live: read and change them in a running policy server (policy_server.py) by UDP, with
no restart and no recompile (each parameter is a traced value of the compiled program).
    python3 tools/solo/policy_cfg.py params [--port 47001]                 the values in force, the profile, the report
    python3 tools/solo/policy_cfg.py set key=value [key=value ...] [--port 47001]
    python3 tools/solo/policy_cfg.py profile [key ...] [--port 47001]      back to the start values (all, or these keys)
The server refuses a set whole unless every key is a parameter of its bot and every value a finite number. The
values in force last until the server is started again (run_duo.sh start / run_rung.sh): a restart of the strategy
node keeps them. Also used by the field page (hq.py)."""
import json, math, os, socket, sys

PORT = int(os.environ.get("SOLO_POLICY_PORT", "47001"))


def num(v):
    """a finite JSON number (not a bool, not a string) as float, else None"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def parse_sets(items):
    """["key=value", ...] -> ({key: float}, "") or (None, why); keys are names, values finite numbers"""
    out = {}
    for it in items:
        k, eq, v = it.partition("=")
        k = k.strip()
        if not eq or not k or not (k.replace("_", "").isalnum() and k.isascii()):
            return None, f"not key=value: {it!r}"
        try:
            f = float(v)
        except ValueError:
            return None, f"not a number: {it!r}"
        if not math.isfinite(f):
            return None, f"not a finite number: {it!r}"
        out[k] = f
    if not out:
        return None, "nothing to set"
    return out, ""


def check_request(b):
    """the body of the page's POST /params -> (cfg request, "") or (None, why):
    {"set": {key: number, ...}} or {"profile": true} or {"profile": [key, ...]}"""
    if not isinstance(b, dict) or len(b) != 1:
        return None, "bad request: one of set, profile"
    if "set" in b:
        s = b["set"]
        if not isinstance(s, dict) or not s or len(s) > 64:
            return None, "bad request: set must be {key: number, ...}"
        vals = {}
        for k, v in s.items():
            f = num(v)
            if not isinstance(k, str) or f is None:
                return None, f"bad request: {k!r} must be a finite number"
            vals[k] = f
        return {"cfg": "set", "set": vals}, ""
    if "profile" in b:
        p = b["profile"]
        if p is True:
            return {"cfg": "profile"}, ""
        if isinstance(p, list) and p and all(isinstance(k, str) for k in p):
            return {"cfg": "profile", "keys": p}, ""
        return None, "bad request: profile must be true or a list of keys"
    return None, "bad request: one of set, profile"


def ask(req, port=PORT, timeout=0.5):
    """send one cfg request to the policy server on 127.0.0.1:port -> its reply, or {"error": why} with no reply"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    seq = int.from_bytes(os.urandom(4), "little")
    try:
        s.sendto(json.dumps({**req, "seq": seq}).encode(), ("127.0.0.1", port))
        while True:
            r = json.loads(s.recv(65535))
            if r.get("seq") == seq:
                if "now" not in r:                             # an older server answers with a robot command
                    return {"error": "the policy server does not take live settings (an older version)", "noreply": False}
                return r
    except (socket.timeout, ConnectionRefusedError):
        return {"error": f"no answer from a policy server on port {port} (not started, or still compiling)", "noreply": True}
    except (OSError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}", "noreply": True}
    finally:
        s.close()


MODES = {0: "cautious", 1: "free", 2: "pick"}


def report_text(rep):
    """the strategy's memory report in a line: duo_att's mode and their fastest speed seen, then the rest"""
    parts = []
    if "mode" in rep:
        parts.append("mode " + MODES.get(int(round(rep["mode"])), str(rep["mode"])))
    if "vo_m" in rep:
        parts.append(f"their top speed {rep['vo_m']:.2f} m/s")
    if "vb_m" in rep:
        parts.append(f"their ball top {rep['vb_m']:.2f} m/s")
    for k in sorted(rep):
        if k not in ("mode", "vo_m", "vb_m"):
            parts.append(f"{k} {rep[k]:g}")
    return ", ".join(parts)


def table(r):
    """the reply as text lines: key, profile, in force (* changed)"""
    if "now" not in r:
        return r.get("error", "no reply")
    lines = [f"{r['bot']}: {len(r['changed'])} changed, {r['compiles']} compiled variants"]
    for k, v in r["now"].items():
        p = r["profile"].get(k, r.get("default", {}).get(k))
        lines.append(f"  {'*' if k in r['changed'] else ' '} {k:14s} profile {p:<10g} now {v:g}")
    lines.append("  report: " + report_text(r.get("report", {})))
    if r.get("error"):
        lines.append("REFUSED: " + r["error"])
    return "\n".join(lines)


def main(argv):
    port = PORT
    if "--port" in argv:
        i = argv.index("--port"); port = int(argv[i + 1]); del argv[i:i + 2]
    if not argv or argv[0] not in ("params", "set", "profile"):
        print(__doc__); return 2
    if argv[0] == "params":
        r = ask({"cfg": "get"}, port)
    elif argv[0] == "set":
        vals, why = parse_sets(argv[1:])
        if vals is None:
            print("REFUSED: " + why); return 1
        r = ask({"cfg": "set", "set": vals}, port)
    else:
        r = ask({"cfg": "profile", **({"keys": argv[1:]} if argv[1:] else {})}, port)
    print(table(r))
    return 1 if r.get("error") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
