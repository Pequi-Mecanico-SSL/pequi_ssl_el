"""Tests of the live parameters: tools/solo/policy_cfg.py (checks, the UDP client) and the field page's PARAMS
(GET/POST /params in hq.py, without ROS) against a fake policy server on a local UDP port.
    uv run --with pytest --with numpy pytest -q tools/solo/tests/test_policy_cfg.py
With POLICY_LIVE=<strategy simulator checkout> (its .venv with JAX) also against a real policy_server.py process on
port 47131: a set takes effect on the next tick, the compiled program is not compiled again, the robots keep getting
commands, a node's reset keeps the values (about two minutes: the server compiles first)."""
import importlib.util, json, math, os, socket, subprocess, sys, threading, time
import pytest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import policy_cfg  # noqa: E402
from test_hq import World, stubs, post, get  # noqa: E402
HQ = os.path.join(os.path.dirname(HERE), "hq.py")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


class FakeServer:
    """answers cfg requests on a UDP port as policy_server.py does (profile, values in force, refusals)"""
    def __init__(self, profile, keys, bot="duo_att"):
        self.profile, self.now, self.keys, self.bot = dict(profile), dict(profile), set(keys) | set(profile), bot
        self.report = {"mode": 2.0, "vo_m": 0.4}
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.s.bind(("127.0.0.1", 0)); self.port = self.s.getsockname()[1]
        self.got = []; threading.Thread(target=self.loop, daemon=True).start()

    def view(self, err=None):
        ch = sorted(k for k in self.now if self.now[k] != self.profile.get(k, 0.0))
        r = dict(bot=self.bot, profile=self.profile, now={k: self.now[k] for k in sorted(set(self.profile) | set(ch))}, changed=ch,
                 default={k: 0.0 for k in ch if k not in self.profile}, report=dict(self.report), keys=sorted(self.keys), compiles=4)
        if err: r["error"] = err
        return r

    def loop(self):
        while True:
            d, a = self.s.recvfrom(65535); q = json.loads(d); self.got.append(q)
            if q.get("cfg") == "set":
                bad = [k for k in q["set"] if k not in self.keys]
                r = self.view(f"not parameters: {bad}") if bad else (self.now.update(q["set"]) or self.view())
            elif q.get("cfg") == "profile":
                for k in q.get("keys", list(self.now)):
                    self.now[k] = self.profile.get(k, 0.0)
                r = self.view()
            else:
                r = self.view()
            r["seq"] = q.get("seq"); self.s.sendto(json.dumps(r).encode(), a)


# ---- the pure checks ------------------------------------------------------------------------------------------------
def test_parse_sets():
    assert policy_cfg.parse_sets(["a_nfv=1.3", "a_nf=1"]) == ({"a_nfv": 1.3, "a_nf": 1.0}, "")
    for bad in (["a_nfv"], ["=1"], ["a_nfv=x"], ["a_nfv=nan"], ["a_nfv=inf"], ["a nfv=1"], ["a;b=1"], []):
        assert policy_cfg.parse_sets(bad)[0] is None, bad


def test_check_request():
    ok = lambda b: policy_cfg.check_request(b)[0]
    assert ok({"set": {"a_nfv": 1.3}}) == {"cfg": "set", "set": {"a_nfv": 1.3}}
    assert ok({"set": {"a_nf": 1}}) == {"cfg": "set", "set": {"a_nf": 1.0}}
    assert ok({"profile": True}) == {"cfg": "profile"}
    assert ok({"profile": ["a_nfv"]}) == {"cfg": "profile", "keys": ["a_nfv"]}
    for bad in ({"set": {"a_nfv": "1.3"}}, {"set": {"a_nfv": True}}, {"set": {"a_nfv": float("nan")}}, {"set": {"a_nfv": float("inf")}},
                {"set": {}}, {"set": [1]}, {"profile": False}, {"profile": []}, {"profile": [1]}, {"set": {"a": 1}, "profile": True},
                {}, [], "x", {"get": True}):
        assert ok(bad) is None, bad


def test_report_text_and_table():
    assert policy_cfg.report_text({"mode": 0.0, "vo_m": 1.034}) == "mode cautious, their top speed 1.03 m/s"
    assert policy_cfg.report_text({"mode": 1.0}) == "mode free"
    assert policy_cfg.report_text({"mode": 2.0, "est_lvl": 0.5}) == "mode pick, est_lvl 0.5"
    t = policy_cfg.table(dict(bot="duo_att", profile={"a_nfv": 0.9}, now={"a_nfv": 1.3, "a_hold": 2.5}, changed=["a_hold", "a_nfv"],
                              default={"a_hold": 0.0}, report={"mode": 2.0}, compiles=4))
    assert "* a_nfv" in t and "profile 0.9" in t and "now 1.3" in t and "* a_hold" in t and "mode pick" in t
    assert policy_cfg.table({"error": "no answer"}) == "no answer"


# ---- the UDP client ---------------------------------------------------------------------------------------------------
def test_ask_set_and_profile():
    f = FakeServer({"a_nfv": 0.9, "a_nf": 2.0}, ["a_hold"])
    r = policy_cfg.ask({"cfg": "set", "set": {"a_nfv": 1.3}}, f.port)
    assert r["now"]["a_nfv"] == 1.3 and r["changed"] == ["a_nfv"]
    r = policy_cfg.ask({"cfg": "profile"}, f.port)
    assert r["now"]["a_nfv"] == 0.9 and r["changed"] == []
    assert all("seq" in q for q in f.got)


def test_ask_no_server_and_old_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    t0 = time.monotonic(); r = policy_cfg.ask({"cfg": "get"}, port, timeout=0.2)    # bound, never answers: compiling
    assert r["noreply"] and "no answer" in r["error"] and time.monotonic() - t0 < 1.0
    def old():                                                                    # an older server: a robot command back
        for _ in range(2):                                                        # (the first, unanswered request too)
            d, a = s.recvfrom(65535); q = json.loads(d)
            s.sendto(json.dumps(dict(v=[0.0, 0.0], w=0.0, mode="error: KeyError('us')", seq=q["seq"])).encode(), a)
    threading.Thread(target=old, daemon=True).start()
    r = policy_cfg.ask({"cfg": "get"}, port)
    assert "older version" in r["error"]


def test_cli(tmp_path):
    f = FakeServer({"a_nfv": 0.9}, ["a_hold"])
    run = lambda *a: subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "policy_cfg.py"), *a, "--port", str(f.port)],
                                    capture_output=True, text=True)
    p = run("set", "a_nfv=1.3", "a_hold=2.5"); assert p.returncode == 0 and "now 1.3" in p.stdout and f.now["a_hold"] == 2.5
    p = run("set", "a_nfv=abc"); assert p.returncode == 1 and "REFUSED" in p.stdout and f.now["a_nfv"] == 1.3
    p = run("set", "zz=1"); assert p.returncode == 1 and "REFUSED" in p.stdout
    p = run("profile", "a_nfv"); assert p.returncode == 0 and f.now["a_nfv"] == 0.9 and f.now["a_hold"] == 2.5
    p = run("params"); assert p.returncode == 0 and "* a_hold" in p.stdout


# ---- the page --------------------------------------------------------------------------------------------------------
@pytest.fixture
def hqp(tmp_path, monkeypatch):
    f = FakeServer({"a_nfv": 0.9, "a_nf": 2.0}, ["a_hold"])
    W = World()
    for k, v in stubs(W).items(): monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(sys, "argv", ["hq.py", "blue"])
    monkeypatch.setenv("SOLO_DATA", str(tmp_path)); monkeypatch.setenv("SOLO_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("SOLO_POLICY_PORTS", str(f.port))
    spec = importlib.util.spec_from_file_location("hq_under_test_p", HQ)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.now = lambda: W.T; m.others_running = lambda: []; m.BUSY[0] = []
    m.F = f
    for _ in range(50):                                          # the page's reader has asked the server once
        if m.PV[f.port] is not None: break
        time.sleep(0.05)
    return m


def page_events(m):
    return [e["text"] for e in m.EVENTS if "params" in e["text"]]


def test_page_get_params(hqp):
    code, body = get(hqp, "/params")
    d = json.loads(body)
    assert code == 200 and d[0]["port"] == hqp.F.port and d[0]["now"] == {"a_nf": 2.0, "a_nfv": 0.9}


def test_page_set_needs_unlock(hqp):
    p = hqp.F.port
    assert post(hqp, "/params", {"port": p, "set": {"a_nfv": 1.3}}) == 423 and hqp.F.now["a_nfv"] == 0.9
    assert post(hqp, "/lock", {"on": False}) == 200
    assert post(hqp, "/params", {"port": p, "set": {"a_nfv": 1.3}}) == 200 and hqp.F.now["a_nfv"] == 1.3
    assert any("a_nfv 0.9 -> 1.3" in t for t in page_events(hqp)), page_events(hqp)
    assert post(hqp, "/params", {"port": p, "profile": ["a_nfv"]}) == 200 and hqp.F.now["a_nfv"] == 0.9
    assert any("a_nfv 1.3 -> 0.9" in t for t in page_events(hqp))


def test_page_refusals(hqp):
    p = hqp.F.port
    assert post(hqp, "/lock", {"on": False}) == 200
    for body in ({"set": {"a_nfv": 1}}, {"port": p + 1, "set": {"a_nfv": 1}}, {"port": True, "set": {"a_nfv": 1}},
                 {"port": p, "set": {"a_nfv": "1"}}, {"port": p, "set": {"a_nfv": None}}, {"port": p}, {"port": p, "profile": False}):
        assert post(hqp, "/params", body) == 400, body
    assert post(hqp, "/params", None, raw=b'{"port": %d, "set": {"a_nfv": NaN}}' % p) == 400
    assert post(hqp, "/params", {"port": p, "set": {"zz": 1}}) == 409                 # the server refuses: logged
    assert hqp.F.now == {"a_nfv": 0.9, "a_nf": 2.0}
    assert any("REFUSED" in t for t in page_events(hqp))


def test_page_logs_mode_change(hqp):
    p = hqp.F.port
    r = policy_cfg.ask({"cfg": "get"}, p); hqp.params_seen(p, r)
    r["report"]["mode"] = 0.0; hqp.params_seen(p, r)
    assert any("mode cautious" in e["text"] for e in hqp.EVENTS if e["kind"] == "strategy")


# ---- a real policy server ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("POLICY_LIVE"), reason="POLICY_LIVE=<strategy simulator checkout> not set")
def test_real_server_live_set():
    sim = os.environ["POLICY_LIVE"]; port = 47131
    prof = json.load(open(os.path.join(REPO, "tools/solo/profiles/duo_att.json")))
    env = dict(os.environ, PYTHONPATH=f"{sim}/src:{sim}/ssl-rules/src", JAX_PLATFORMS="cpu")
    p = subprocess.Popen([f"{sim}/.venv/bin/python", os.path.join(REPO, "src/solo_strategy/solo_strategy/policy_server.py"), "--bot", "duo_att",
                          "--port", str(port), "--keep", "vo_m,vb_m,om", "--params", json.dumps(prof)], cwd=sim, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in p.stdout:
            if "listening" in line: break
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(10)
        def ask(q):
            s.sendto(json.dumps(q).encode(), ("127.0.0.1", port)); return json.loads(s.recv(65535))
        def tick(t, vx, reset=False):
            return ask(dict(t=t, state=3, ours=False, cmd_t=0.0, still_since=0.0, last_team=0, kick_pos=[0, 0], reset=reset,
                            us=[[-2.0, 0.0, 0.0, 0, 0, 1], [-0.8, 0.3, 0.0, 0, 0, 1]], opp=[[1.0, 0.5, 3.1, vx, 0.0, 1]], ball=[0.3, 0.0, 0.0, 0.0]))
        g = ask({"cfg": "get"}); c0 = g["compiles"]
        assert g["now"] == {k: float(v) for k, v in prof.items()} and g["changed"] == []
        for k in range(60): r = tick(k / 30, 0.6)                      # their robot at 0.6 m/s: under a_nfv 0.9
        assert not r["cautious"] and r["their_v"] > 0.3
        r = ask({"cfg": "set", "set": {"a_nfv": 0.2}})                 # now 0.6 m/s is fast: cautious on the next tick
        assert r["now"]["a_nfv"] == 0.2 and r["changed"] == ["a_nfv"]
        t1 = time.perf_counter(); r = tick(2.0, 0.6); first = (time.perf_counter() - t1) * 1000
        assert r["cautious"] and first < 100 and any(abs(x) > 1e-6 for v in r["v"] for x in v)
        assert ask({"cfg": "get"})["compiles"] == c0
        ms = sorted(ask({"cfg": "get"})["ms"] for _ in range(20))      # a cfg request delays the next tick by this
        print("cfg get server ms: median", ms[10], "max", ms[-1])
        assert ms[10] < 2.0
        for bad in (float("nan"), float("inf"), 1e39, "x", True):
            r = ask({"cfg": "set", "set": {"a_nfv": bad}})
            assert "error" in r and r["now"]["a_nfv"] == 0.2
        assert "error" in ask({"cfg": "set", "set": {"d_role": 0}})
        tick(3.0, 0.0, reset=True)                                     # a node restart: the values stay
        assert ask({"cfg": "get"})["now"]["a_nfv"] == 0.2
        r = ask({"cfg": "profile"}); assert r["changed"] == [] and r["now"]["a_nfv"] == float(prof["a_nfv"])
        assert not tick(3.1, 0.6)["cautious"]
    finally:
        p.terminate(); p.wait(10)
