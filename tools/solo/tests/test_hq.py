"""Tests of the ball placement in tools/solo/hq.py without ROS: rclpy and the message packages are stubs, the HTTP
server is not started, the 50 Hz loop runs against the offline simulation (ballsim.py) on a simulated clock.
    uv run --with pytest pytest -q tools/solo/tests/test_hq.py"""
import importlib.util, io, json, math, os, random, sys, types
import pytest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import ballsim  # noqa: E402
HQ = os.path.join(os.path.dirname(HERE), "hq.py")


class World:
    """the simulated clock, the loop's exit condition and what the stubs saw"""
    def __init__(self):
        self.T = 1000.0; self.ok = lambda: False; self.spin = lambda: None; self.sent = []


def stubs(W):
    def mod(name, **kw):
        m = types.ModuleType(name); m.__dict__.update(kw); return m
    class Pub:
        def publish(self, m): W.sent.append((W.T, m.linear.x, m.linear.y, m.angular.z))
    class Node:
        def __init__(self, *a): pass
        def create_publisher(self, *a): return Pub()
        def create_subscription(self, *a): return None
    class V: x = y = z = 0.0
    class Twist:
        def __init__(self): self.linear, self.angular = V(), V()
    class Msg: pass
    def spin_once(node, timeout_sec=0.0):
        W.T += 0.02; W.spin()
    return {"rclpy": mod("rclpy", init=lambda: None, ok=lambda: W.ok(), spin_once=spin_once),
            "rclpy.node": mod("rclpy.node", Node=Node), "rclpy.qos": mod("rclpy.qos", qos_profile_sensor_data=None),
            "geometry_msgs": mod("geometry_msgs"), "geometry_msgs.msg": mod("geometry_msgs.msg", Pose2D=Msg, Twist=Twist),
            "std_msgs": mod("std_msgs"), "std_msgs.msg": mod("std_msgs.msg", String=Msg),
            "pequi_msgs": mod("pequi_msgs"), "pequi_msgs.msg": mod("pequi_msgs.msg", GameState=Msg),
            "referee_connector": None}                     # the game controller listener returns at once


@pytest.fixture
def hq(tmp_path, monkeypatch):
    W = World()
    for k, v in stubs(W).items(): monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(sys, "argv", ["hq.py", "blue"])
    monkeypatch.setenv("SOLO_DATA", str(tmp_path)); monkeypatch.setenv("SOLO_RUN_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("hq_under_test", HQ)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)      # main() is not run: no port, no loop
    m.now = lambda: W.T; m.others_running = lambda: list(W.busy); W.busy = []; m.BUSY[0] = []
    m.W = W
    return m


PAGE_HEADERS = {"Host": "127.0.0.1:8700", "Content-Type": "application/json"}      # what the page's fetch sends


def post(hq, path, body, headers=None, raw=None):
    """one POST through the handler as the page sends it -> the response code"""
    raw = json.dumps(body).encode() if raw is None else raw; h = hq.H.__new__(hq.H); got = []
    hd = dict(PAGE_HEADERS if headers is None else headers, **{"Content-Length": str(len(raw))})
    h.path, h.headers, h.rfile = path, hd, io.BytesIO(raw)
    h.out = lambda body, ctype="application/json", code=200: got.append(code)
    h.do_POST()
    return got[0]


def get(hq, path, headers=None):
    """one GET through the handler -> (code, body)"""
    got = []; h = hq.H.__new__(hq.H); h.path = path; h.headers = dict(PAGE_HEADERS if headers is None else headers)
    h.out = lambda body, ctype="application/json", code=200: got.append((code, body)); h.do_GET()
    return got[0]


def state(hq):
    code, body = get(hq, "/state"); assert code == 200
    return json.loads(body, parse_constant=lambda c: pytest.fail("not JSON: " + c))


def balls(hq):
    return [e["text"] for e in hq.EVENTS if e["kind"] == "ball"]


def hook_sim(hq, robot, ball, seed=1, page=True):
    """the loop's spin_once steps the simulation: vision at 60 Hz into S, the page polling, commands into the robot"""
    W = hq.W; sim = ballsim.Sim(robot, ball, random.Random(seed)); vis = [sim.t]; W.page = page; W.robot_seen = True
    def spin():
        while sim.t < W.T - 1000.0:
            sim.step()
            if sim.t >= vis[0]:
                vis[0] += 1 / 60.0; r, b = sim.seen(sim.t)
                if r and W.robot_seen: hq.S["us"], hq.S["us_t"] = r, W.T
                if b: hq.S["ball"], hq.S["ball_t"] = b, W.T
        if W.page: hq.S["poll"] = W.T
        if W.sent: sim.command(W.sent[-1][1:])
    W.spin = spin
    for _ in range(30): W.T += 0.02; spin()                  # vision has seen everything once
    return sim


def run(hq, secs, until=None):
    t_end = hq.W.T + secs
    hq.W.ok = lambda: hq.W.T < t_end and not (until and until())
    hq.control()


# ---- the pure functions ------------------------------------------------------------------------------------------
def test_place_request_validation(hq):
    ok = lambda b: hq.place_request(b)[0]
    assert ok({"x": 0.5, "y": -0.2}) == (0.5, -0.2, 0.15)
    assert ok({"x": 0.5, "y": -0.2, "precise": True}) == (0.5, -0.2, 0.05)
    assert ok({"x": 0.5, "y": -0.2, "precise": False}) == (0.5, -0.2, 0.15)
    assert ok({"x": 1, "y": 0, "tol": 0.08}) == (1.0, 0.0, 0.08)
    for b in ({"x": float("nan"), "y": 0}, {"x": 0, "y": float("inf")}, {"x": 0, "y": 0, "tol": float("nan")}, {"y": 0},
              {"x": "a", "y": 0}, {"x": None, "y": 0}, {"x": 0, "y": 0, "tol": 0.0}, {"x": 0, "y": 0, "tol": 2.0}, [1, 2],
              {"x": "1", "y": 0}, {"x": True, "y": 0}, {"x": 0, "y": 0, "tol": "0.1"}, {"x": 0, "y": 0, "precise": "false"},
              {"x": 0, "y": 0, "precise": 1}, "x"):
        assert ok(b) is None and hq.place_request(b)[1].startswith("bad request"), b


def test_small_helpers(hq):
    assert hq.place_speed(0.05) == 0.1 and hq.place_speed(0.6) == 0.6 and hq.place_speed(1.5) == 1.0
    m = hq.PLACE_MARGIN; ex = hq.LX + 0.10 + m
    assert hq.in_place_box(0, 0) and hq.in_place_box(ex - 1e-6, 0) and not hq.in_place_box(ex + 0.01, 0)
    assert not hq.in_place_box(0, -(hq.LY + 0.10 + m) - 0.01)
    pl = hq.BallPlacer((0, 0)); P = dict(pl=pl)
    assert hq.loop_mode(None, None) == "idle" and hq.loop_mode((0, 0, 0), None) == "goto"
    assert hq.loop_mode(None, P) == "place" and hq.loop_mode((0, 0, 0), P) == "place"
    pl.stop_with(0.0, "stopped"); assert hq.loop_mode(None, P) == "idle" and hq.loop_mode((0, 0, 0), P) == "goto"
    assert hq.new_ball((1, 2), 5.0, 4.0) == ((1, 2), 5.0) and hq.new_ball((1, 2), 5.0, 5.0) == (None, 5.0)
    assert hq.new_ball(None, 5.0, 4.0) == (None, 4.0)
    assert hq.phase_line("", "waiting: ball moving") is None and hq.phase_line("a", "a") is None
    assert hq.phase_line("ball rolling", "lining up for a carry push of 0.60 m") == "lining up for a carry push of 0.60 m"
    assert hq.clean({"a": (1.0, float("nan")), "b": [float("inf")], "c": True}) == {"a": [1.0, None], "b": [None], "c": True}
    assert "6 cm" in hq.result_text(dict(ok=True, ball=[0.5, 0.1], error=0.06, time=12.0, pushes=2))


def test_place_tick_rules(hq):
    pl = hq.BallPlacer((0.5, 0.0)); us = (0.0, -0.5, 0.0)
    assert hq.place_tick(pl, 0.0, us, 0.5, (0.0, 0.0), 0.1, []) == (0.0, 0.0, 0.0) and not pl.finished   # stale pose
    for kw, word in ((dict(busy=["solo_node"]), "solo_node is running"), (dict(poll=1.6), "page stopped asking"),
                     (dict(us=(hq.LX + 0.5, 0.0, 0.0)), "left the area"), (dict(age=2.1), "our robot not seen"),
                     (dict(us=None), "our robot not seen")):
        pl = hq.BallPlacer((0.5, 0.0)); a = dict(t=0.0, us=us, age=0.0, ball=(0.0, 0.0), poll=0.1, busy=[]); a.update(kw)
        assert hq.place_tick(pl, **a) == (0.0, 0.0, 0.0) and pl.finished and word in pl.status, word


# ---- the server ------------------------------------------------------------------------------------------------
def test_place_is_refused_like_goto(hq):
    hq.S["us"], hq.S["us_t"] = (0.0, 0.0, 0.0), hq.W.T
    post(hq, "/place", {"x": 0.5, "y": 0.0})                       # locked at start
    assert hq.S["place"] is None and hq.S["note"] == "locked: unlock first"
    post(hq, "/lock", {"on": False})
    post(hq, "/place", {"x": float("nan"), "y": 0.0}); assert hq.S["place"] is None and "bad request" in hq.S["note"]
    hq.W.busy = ["series.py"]; post(hq, "/place", {"x": 0.5, "y": 0.0})
    assert hq.S["place"] is None and "series.py is running" in hq.S["note"]
    hq.W.busy = []; hq.S["us_t"] = hq.W.T - 1.0; post(hq, "/place", {"x": 0.5, "y": 0.0})
    assert hq.S["place"] is None and "not seen" in hq.S["note"]
    hq.S["us_t"] = hq.W.T; BP = hq.BallPlacer; hq.BallPlacer = None; hq.PLACE_ERR = "place_ball.py not loaded (x)"
    post(hq, "/place", {"x": 0.5, "y": 0.0}); assert hq.S["place"] is None and "ball mode off" in hq.S["note"]
    s = state(hq); assert s["place_ok"] is False and "not loaded" in s["place_err"]
    hq.BallPlacer = BP; post(hq, "/place", {"x": 0.5, "y": 0.0, "precise": True})
    P = hq.S["place"]; assert P and P["tol"] == 0.05 and not P["pl"].finished and balls(hq)[-1].startswith("placement to (+0.50, +0.00)")
    s = state(hq)["place"]; assert s["active"] and s["target"] == [0.5, 0.0] and s["tol"] == 0.05 and s["status"]


def test_goto_place_stop_lock_replace_each_other(hq):
    hq.S["us"], hq.S["us_t"] = (0.0, 0.0, 0.0), hq.W.T; post(hq, "/lock", {"on": False})
    post(hq, "/goto", {"x": 1.0, "y": 0.0}); assert hq.S["target"] is not None
    post(hq, "/place", {"x": 0.5, "y": 0.0}); assert hq.S["target"] is None and hq.S["place"]
    pl1 = hq.S["place"]["pl"]; post(hq, "/place", {"x": -0.5, "y": 0.0})
    assert pl1.finished and "replaced by a new placement" in pl1.status and hq.S["place"]["pl"] is not pl1
    pl2 = hq.S["place"]["pl"]; post(hq, "/goto", {"x": 1.0, "y": 0.0})
    assert pl2.finished and "replaced by a drive target" in pl2.status and hq.S["target"] is not None
    post(hq, "/place", {"x": 0.5, "y": 0.0}); pl3 = hq.S["place"]["pl"]; post(hq, "/stop", {})
    assert pl3.finished and pl3.status == "stopped" and hq.S["note"].startswith("placement ended: stopped")
    post(hq, "/place", {"x": 0.5, "y": 0.0}); pl4 = hq.S["place"]["pl"]; post(hq, "/lock", {"on": True})
    assert pl4.finished and "locked" in pl4.status
    assert sum(t.startswith("placement ended") for t in balls(hq)) == 4          # one result line each
    assert state(hq)["place"]["result"]["status"] == pl4.status


# ---- the 50 Hz loop with the simulation --------------------------------------------------------------------------
def start(hq, x, y, **kw):
    post(hq, "/lock", {"on": False}); post(hq, "/place", dict(x=x, y=y, **kw)); assert hq.S["place"], hq.S["note"]
    pl = hq.S["place"]["pl"]; out = []; step = pl.step
    def spy(t, r, b):
        c = step(t, r, b); out.append((t, r is not None) + tuple(c[:3])); return c
    pl.step = spy
    return pl, out


def test_a_simulated_placement_through_the_loop(hq):
    sim = hook_sim(hq, (-0.6, -0.4, 0.0), (0.0, 0.0), seed=4)
    pl, out = start(hq, 0.6, 0.3, precise=True)
    run(hq, 70.0, until=lambda: pl.finished and hq.W.T > pl.t_end + 1.0)
    r = hq.S["place"]["result"]
    assert pl.status == "done" and r["ok"] and r["error"] <= 0.05, r
    assert math.hypot(sim.bx - 0.6, sim.by - 0.3) < 0.06
    sent = {round(t, 3): c for t, *c in hq.W.sent}
    for t, fresh, vx, vy, w in out:                       # published as the placer gave it (no ramp, no frame change)
        assert sent[round(t, 3)] == ([vx, vy, w] if fresh else [0.0, 0.0, 0.0])
    assert any(abs(c[0]) > 0.05 for c in sent.values())
    tail = [c for t, *c in hq.W.sent if t > pl.t_end]; assert tail and all(c == [0.0, 0.0, 0.0] for c in tail)
    L = balls(hq); print("\n  " + "\n  ".join(L))
    assert L[0].startswith("placement to") and L[-1].startswith("ball placed") and 2 <= len(L) <= 20
    assert not any(x.startswith("waiting") for x in L)
    s = state(hq)["place"]; assert not s["active"] and s["text"] == L[-1] and s["result"]["pushes"] >= 1


def test_dead_mans_switch_pose_age_busy_box(hq):
    hook_sim(hq, (-0.6, -0.4, 0.0), (0.0, 0.0), seed=5)
    pl, out = start(hq, 1.0, 0.5); run(hq, 3.0)
    assert not pl.finished and any(abs(c[1]) > 0.05 for c in hq.W.sent)
    hq.W.robot_seen = False; n = len(hq.W.sent); run(hq, 0.6)              # no fresh pose: zeros
    assert all(c[1:] == (0.0, 0.0, 0.0) for c in hq.W.sent[n + 12:]) and not pl.finished
    hq.W.robot_seen = True; run(hq, 1.0)
    hq.W.page = False; t0 = hq.W.T; run(hq, 3.0)                           # the page stops polling
    assert pl.finished and "page stopped asking" in pl.status and pl.t_end - t0 <= 1.6
    assert all(c[1:] == (0.0, 0.0, 0.0) for c in hq.W.sent if c[0] > pl.t_end)
    assert balls(hq)[-1].startswith("placement ended: failed: page stopped asking")
    hq.W.page = True; pl, out = start(hq, -1.2, -0.6); run(hq, 2.0); assert not pl.finished
    hq.W.busy = ["solo_node"]; t0 = hq.W.T; run(hq, 1.5)                   # the strategy node appears
    assert pl.finished and "solo_node is running" in pl.status and pl.t_end - t0 <= 1.05
    hq.W.busy = []; pl, out = start(hq, -1.2, -0.6); run(hq, 0.5); assert not pl.finished
    hq.S["us"] = (hq.LX + 0.5, 0.0, 0.0); hq.W.robot_seen = False; run(hq, 0.1)  # outside the target box
    assert pl.finished and "left the area" in pl.status


def test_goto_still_drives_and_the_dead_mans_switch_drops_it(hq):
    hook_sim(hq, (-0.6, -0.4, 0.0), (1.5, 1.0), seed=6)
    post(hq, "/lock", {"on": False}); post(hq, "/goto", {"x": 0.4, "y": -0.4}); run(hq, 6.0, until=lambda: hq.S["target"] is None)
    assert hq.S["note"] == "arrived" and math.hypot(hq.S["us"][0] - 0.4, hq.S["us"][1] + 0.4) < 0.06
    post(hq, "/goto", {"x": -0.6, "y": -0.4}); run(hq, 0.5); hq.W.page = False; run(hq, 2.0)
    assert hq.S["target"] is None and hq.S["note"] == "page stopped asking: target dropped"


def test_without_place_ball_the_page_still_works(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "place_ball", None)                  # the import fails
    W = World()
    for k, v in stubs(W).items(): monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(sys, "argv", ["hq.py", "yellow"]); monkeypatch.setenv("SOLO_DATA", str(tmp_path))
    spec = importlib.util.spec_from_file_location("hq_no_placer", HQ); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); m.now = lambda: W.T; m.others_running = lambda: []; m.BUSY[0] = []; m.W = W
    assert m.BallPlacer is None and "not loaded" in m.PLACE_ERR
    s = state(m); assert s["place_ok"] is False and s["place"] is None
    m.S["us"], m.S["us_t"] = (0.0, 0.0, 0.0), W.T; post(m, "/lock", {"on": False})
    post(m, "/place", {"x": 0.5, "y": 0.0}); assert m.S["place"] is None and m.S["note"].startswith("ball mode off")
    post(m, "/goto", {"x": 0.5, "y": 0.0}); assert m.S["target"] is not None


HARNESS = r"""
const POSTS=[],els={};let STATE=null;
const ctx=new Proxy({},{get:(t,k)=>k in t?t[k]:()=>{},set:(t,k,v)=>{t[k]=v;return true}});
function el(id){return els[id]||(els[id]={id,style:{},dataset:{},textContent:'',value:'rough',offsetHeight:0,title:'',className:'',
  appendChild(){},prepend(){},getContext(){return ctx}})}
globalThis.document={getElementById:el,activeElement:null,createElement:()=>el('n'+Math.random())};
globalThis.innerWidth=1600;globalThis.innerHeight=900;globalThis.addEventListener=()=>{};globalThis.setInterval=()=>{};
globalThis.fetch=async(u,o)=>{if(o)POSTS.push([u,JSON.parse(o.body)]);return {json:async()=>u.startsWith('/events')?[]:STATE}};
const fs=require('fs');const page=fs.readFileSync(process.argv[2],'utf8');const states=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
eval(page+';globalThis.T={poll,pm,ptol,c,plEl}');
(async()=>{const out={};
 for(const [i,s] of states.entries()){STATE=s;await T.poll();out['pl'+i]=T.plEl.textContent;out['pm'+i]=T.pm.textContent}
 STATE=states[0];await T.poll();T.pm.onclick();out.mode=T.pm.textContent;T.ptol.value='precise';
 T.c.onmousedown({offsetX:900,offsetY:300});T.c.onmouseup({offsetX:900,offsetY:300});
 T.pm.onclick();T.c.onmousedown({offsetX:500,offsetY:300});T.c.onmouseup({offsetX:500,offsetY:300});
 out.posts=POSTS;console.log(JSON.stringify(out))})();
"""


def test_page_script_parses_and_runs(hq, tmp_path):
    import re, shutil, subprocess
    node = shutil.which("node")
    if not node: pytest.skip("no node")
    js = re.search(r"<script>(.*)</script>", hq.PAGE, re.S).group(1); (tmp_path / "page.js").write_text(js)
    chk = subprocess.run([node, "--check", str(tmp_path / "page.js")], capture_output=True, text=True)
    assert chk.returncode == 0, chk.stderr
    hook_sim(hq, (-0.6, -0.4, 0.0), (0.0, 0.0), seed=4)
    pl, out = start(hq, 0.6, 0.3); hq.S["lock"] = False
    run(hq, 6.0, until=lambda: pl.state == "push"); s1 = state(hq)          # a placement under way
    assert s1["place"]["active"] and "stand" in s1["place"]["points"]
    run(hq, 60.0, until=lambda: pl.finished); s2 = state(hq)                 # and finished
    s3 = dict(s2, place_ok=False, place_err="place_ball.py not loaded (test)", place=None)
    (tmp_path / "states.json").write_text(json.dumps([s1, s2, s3])); (tmp_path / "h.js").write_text(HARNESS)
    r = subprocess.run([node, str(tmp_path / "h.js"), str(tmp_path / "page.js"), str(tmp_path / "states.json")],
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout.strip().splitlines()[-1]); print("\n ", o)
    assert o["pl0"].startswith("placing the ball at (0.60, 0.30), tol 15 cm, 0.60 m/s:") and "push" in o["pl0"]
    assert o["pl1"].startswith("last placement: ball placed") and o["pl2"].startswith("ball mode off") and o["pm2"] == "ball mode off"
    assert o["mode"] == "click = place ball"
    (u1, b1), (u2, b2) = o["posts"][-2:]
    assert u1 == "/place" and b1["precise"] is True and set(b1) == {"x", "y", "precise"} and u2 == "/goto" and "th" not in b2


def test_a_placer_error_ends_the_placement_not_the_loop(hq):
    hook_sim(hq, (-0.6, -0.4, 0.0), (0.0, 0.0), seed=8)
    pl, out = start(hq, 0.6, 0.3); run(hq, 1.0)
    def boom(*a): raise ZeroDivisionError("x")
    pl.step = boom; run(hq, 1.0)
    assert pl.finished and "placer error" in pl.status and balls(hq)[-1].startswith("placement ended: failed: placer error")
    after = [c for c in hq.W.sent if c[0] >= pl.t_end]; assert len(after) >= 20 and all(c[1:] == (0.0, 0.0, 0.0) for c in after)
    post(hq, "/goto", {"x": 0.0, "y": -0.4}); run(hq, 1.0); assert any(abs(c[1]) + abs(c[2]) > 0.05 for c in hq.W.sent[-40:])


# ---- the request checks on every endpoint ----------------------------------------------------------------------------
def test_check_functions(hq):
    assert hq.check_goto({"x": 1, "y": -0.5}) == (True, 1.0, -0.5, None)
    assert hq.check_goto({"x": 1, "y": -0.5, "th": 0.3}) == (True, 1.0, -0.5, 0.3)
    for b in ({"x": float("nan"), "y": 0}, {"x": float("inf"), "y": 0}, {"x": 0}, {"x": "1", "y": 0}, {"x": True, "y": 0},
              {"x": None, "y": 0}, {"x": 0, "y": 0, "th": float("-inf")}, [1, 2], "x"):
        assert hq.check_goto(b) == (False, None, None, None), b
    assert hq.check_speed({"v": 0.9}) == (True, 0.9)
    for b in ({"v": float("nan")}, {"v": "1"}, {"v": False}, {}, [0.5]):
        assert hq.check_speed(b)[0] is False, b
    assert hq.check_flag({"on": False}, "on", True) == (True, False) and hq.check_flag({}, "on", True) == (True, True)
    assert hq.check_flag({}, "on") == (False, None)
    for v in ("false", 0, 1, None, [True]):
        assert hq.check_flag({"on": v}, "on", True) == (False, None), v
    assert hq.host_ok("127.0.0.1:8700", 8700, "127.0.0.1") and hq.host_ok("LOCALHOST:8700", 8700, "127.0.0.1")
    assert not hq.host_ok("evil.example", 8700, "127.0.0.1") and not hq.host_ok(None, 8700, "127.0.0.1")
    assert not hq.host_ok("127.0.0.1:8701", 8700, "127.0.0.1") and hq.host_ok("198.51.100.7:8700", 8700, "198.51.100.7")


def snapshot(hq):
    P = hq.S["place"]
    return (hq.S["target"], hq.S["speed"], hq.S["lock"], P and P["pl"], P and P["pl"].finished)


def test_every_post_needs_json_type_and_host(hq):
    hq.S["us"], hq.S["us_t"] = (0.0, 0.0, 0.0), hq.W.T; assert post(hq, "/lock", {"on": False}) == 200
    assert post(hq, "/goto", {"x": 1.0, "y": 0.0}) == 200 and hq.S["target"] is not None
    before = snapshot(hq)
    for path, body in (("/stop", {}), ("/speed", {"v": 0.9}), ("/goto", {"x": -1.0, "y": 0.0}), ("/place", {"x": 0.5, "y": 0.0}),
                       ("/lock", {"on": True}), ("/strategy", {"on": False})):
        for hd, code in (({"Host": "127.0.0.1:8700", "Content-Type": "text/plain"}, 415),
                         ({"Host": "127.0.0.1:8700", "Content-Type": "application/x-www-form-urlencoded"}, 415),
                         ({"Host": "127.0.0.1:8700"}, 415),
                         ({"Host": "evil.example", "Content-Type": "application/json"}, 403),
                         ({"Content-Type": "application/json"}, 403)):
            assert post(hq, path, body, headers=hd) == code, (path, hd)
            assert snapshot(hq) == before, (path, hd)
    assert post(hq, "/nothing", {}) == 404 and snapshot(hq) == before
    assert post(hq, "/goto", None, raw=b"{bad") == 400 and post(hq, "/goto", None, raw=b"\xff\xfe") == 400
    assert post(hq, "/goto", None, raw=b'{"x": 1, "y": 0, "pad": "' + b"a" * 5000 + b'"}') == 400
    assert snapshot(hq) == before
    assert get(hq, "/state", headers={"Host": "evil.example"})[0] == 403
    assert get(hq, "/events?since=0", headers={"Host": "evil.example"})[0] == 403
    assert get(hq, "/", headers={})[0] == 403
    assert post(hq, "/goto", {"x": 1.0, "y": 0.0}, headers={"Host": "localhost:8700", "Content-Type": "application/json; charset=utf-8"}) == 200


def test_numbers_and_flags_are_checked_on_every_endpoint(hq):
    hq.S["us"], hq.S["us_t"] = (0.0, 0.0, 0.0), hq.W.T; post(hq, "/lock", {"on": False})
    post(hq, "/goto", {"x": 1.0, "y": 0.0}); before = snapshot(hq)
    nan, inf = float("nan"), float("inf")
    for path, body in (("/speed", {"v": nan}), ("/speed", {"v": inf}), ("/speed", {"v": "1"}), ("/speed", {}),
                       ("/goto", {"x": nan, "y": 0}), ("/goto", {"x": 0, "y": -inf}), ("/goto", {"x": 0, "y": 0, "th": nan}),
                       ("/goto", {"x": "1", "y": 0}), ("/goto", {"x": True, "y": 0}), ("/goto", {"y": 0}),
                       ("/place", {"x": nan, "y": 0}), ("/place", {"x": 0, "y": 0, "tol": inf}), ("/place", {"x": "0.5", "y": 0}),
                       ("/place", {"x": 0, "y": 0, "precise": "true"}), ("/place", {"x": 0, "y": 0, "tol": 0.001}),
                       ("/lock", {"on": "false"}), ("/lock", {"on": 0}), ("/lock", {"on": None}),
                       ("/strategy", {"on": "false"}), ("/strategy", {"on": 1}), ("/strategy", {})):
        assert post(hq, path, body) == 400, (path, body)
        assert snapshot(hq) == before, (path, body)
    assert post(hq, "/speed", {"v": 9}) == 200 and hq.S["speed"] == hq.VMAX
    assert post(hq, "/speed", {"v": 0.0}) == 200 and hq.S["speed"] == 0.1


def test_the_lock_refuses_what_it_should(hq, monkeypatch):
    calls = []
    monkeypatch.setattr(hq, "strategy_on", lambda: calls.append("on") or (True, ""))
    monkeypatch.setattr(hq, "strategy_off", lambda: calls.append("off") or (True, ""))
    hq.S["us"], hq.S["us_t"] = (0.0, 0.0, 0.0), hq.W.T; assert hq.S["lock"] is True       # locked at start
    sp = hq.S["speed"]
    for path, body in (("/goto", {"x": 1.0, "y": 0.0}), ("/place", {"x": 0.5, "y": 0.0}), ("/speed", {"v": 1.2}),
                       ("/strategy", {"on": True}), ("/strategy", {"on": False})):
        assert post(hq, path, body) == 423, path
    assert hq.S["target"] is None and hq.S["place"] is None and hq.S["speed"] == sp and calls == []
    assert post(hq, "/stop", {}) == 200 and hq.S["note"] == "stopped"             # STOP works while locked
    assert post(hq, "/lock", {}) == 200 and hq.S["lock"] is True                     # no flag: stays locked
    assert post(hq, "/lock", {"on": False}) == 200 and hq.S["lock"] is False
    assert post(hq, "/strategy", {"on": False}) == 200 and calls == ["off"]
    post(hq, "/goto", {"x": 1.0, "y": 0.0}); assert hq.S["target"] is not None
    assert post(hq, "/lock", {"on": True}) == 200 and hq.S["lock"] and hq.S["target"] is None


def node_files(tmp_path):
    """an executable named python3 and a file named solo_node (never run: Popen is replaced)"""
    (tmp_path / "bin").mkdir(exist_ok=True); (tmp_path / "lib").mkdir(exist_ok=True)
    py = tmp_path / "bin" / "python3"; py.write_text("#!/bin/sh\n"); py.chmod(0o755)
    node = tmp_path / "lib" / "solo_node"; node.write_text("x"); other = tmp_path / "lib" / "other"; other.write_text("x")
    return str(py), str(node), str(other)


def test_strategy_on_runs_only_the_saved_node_command(hq, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(hq, "strategy_procs", lambda: [])
    monkeypatch.setattr(hq.subprocess, "Popen", lambda argv, **k: started.append(argv) or types.SimpleNamespace(pid=1))
    py, node, other = node_files(tmp_path)
    hostile = (["sh", "-c", "echo x"], [py], [py, other, "--ros-args"], " ".join([py, node]), [1, 2], {"a": 1},
               [py, "-c__import__('os').system('x')#/solo_node"], ["python3", "-c<code>#/solo_node"],
               ["python3", node, "--ros-args"], [py, node, "--ros-args", "-p", "evil:=1"], [py, node, "--ros-args", "-c", "x"])
    for argv in hostile:
        open(hq.CMDF, "w").write(json.dumps(argv)); ok, why = hq.strategy_on(); assert ok is False and why, argv
    assert started == []
    open(hq.CMDF, "w").write("{not json"); assert hq.strategy_on()[0] is False and started == []
    good = [py, node, "--ros-args", "-p", "team:=blue"]
    open(hq.CMDF, "w").write(json.dumps(good)); assert hq.strategy_on() == (True, "") and started == [good]


def test_a_refused_command_line_is_noted_and_logged(hq, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(hq, "strategy_procs", lambda: [])
    monkeypatch.setattr(hq.subprocess, "Popen", lambda argv, **k: started.append(argv) or types.SimpleNamespace(pid=1))
    open(hq.CMDF, "w").write(json.dumps(["python3", "-c<code>#/solo_node"]))
    post(hq, "/lock", {"on": False}); assert post(hq, "/strategy", {"on": True}) == 200
    assert started == [] and "saved command line refused" in hq.S["note"]
    assert any(e["text"] == hq.S["note"] for e in hq.EVENTS)


def test_strategy_on_is_refused_while_another_commander_runs(hq, monkeypatch):
    calls = []
    monkeypatch.setattr(hq, "strategy_on", lambda: calls.append("on") or (True, ""))
    monkeypatch.setattr(hq, "strategy_off", lambda: calls.append("off") or (True, ""))
    post(hq, "/lock", {"on": False})
    for other in (["characterize.py"], ["goto.py"], ["solo_node", "nudge.py"]):
        hq.W.busy = other; n = len(hq.EVENTS)
        assert post(hq, "/strategy", {"on": True}) == 409 and calls == [], other
        assert "strategy not started" in hq.S["note"] and other[-1] in hq.S["note"] and hq.EVENTS[n:]
    hq.W.busy = ["solo_node"]; assert post(hq, "/strategy", {"on": True}) == 200 and calls == ["on"]   # the node itself
    hq.W.busy = ["goto.py"]; assert post(hq, "/strategy", {"on": False}) == 200 and calls == ["on", "off"]  # off: always


def test_strategy_off_stops_even_when_the_line_cannot_be_saved(hq, monkeypatch, tmp_path):
    node = ["/usr/bin/python3", "/ws/lib/solo_strategy/solo_node", "--ros-args"]
    table = [(4242, node)]; killed = []
    monkeypatch.setattr(hq.procs, "process_table", lambda: list(table))
    def kill(pid, sig): killed.append((pid, sig)); table.clear()
    monkeypatch.setattr(hq.os, "kill", kill)
    monkeypatch.setattr(hq, "RUN", str(tmp_path / "new" / "run")); monkeypatch.setattr(hq, "CMDF", str(tmp_path / "new" / "run" / "solo.cmd"))
    assert hq.strategy_off() == (True, "") and json.load(open(hq.CMDF)) == node and killed == [(4242, hq.signal.SIGINT)]
    table.append((4243, node)); killed.clear(); (tmp_path / "file").write_text("")
    monkeypatch.setattr(hq, "RUN", str(tmp_path / "file" / "run")); monkeypatch.setattr(hq, "CMDF", str(tmp_path / "file" / "run" / "solo.cmd"))
    ok, why = hq.strategy_off(); assert ok and "not saved" in why and killed == [(4243, hq.signal.SIGINT)]
    def boom(): raise RuntimeError("x")
    monkeypatch.setattr(hq, "strategy_off", boom); post(hq, "/lock", {"on": False})
    assert post(hq, "/strategy", {"on": False}) == 200 and "strategy switch failed" in hq.S["note"]


def test_a_page_placement_keeps_the_carry_under_max_speed(hq, monkeypatch):
    monkeypatch.setattr(hq, "VMAX", 0.3); hq.S["us"], hq.S["us_t"] = (0.0, 0.0, 0.0), hq.W.T
    post(hq, "/lock", {"on": False}); post(hq, "/place", {"x": 0.5, "y": 0.0})
    assert hq.S["place"]["pl"].carry == 0.3


def test_events_since_is_never_negative(hq):
    for i in range(5): hq.event("page", f"e{i}")
    code, body = get(hq, "/events?since=-2"); assert code == 200 and [e["text"] for e in json.loads(body)] == [f"e{i}" for i in range(5)]
    code, body = get(hq, "/events?since=3"); assert [e["text"] for e in json.loads(body)] == ["e3", "e4"]


def test_a_target_is_dropped_after_2_s_unseen(hq):
    hook_sim(hq, (-0.6, -0.4, 0.0), (1.5, 1.0), seed=6)
    post(hq, "/lock", {"on": False}); post(hq, "/goto", {"x": 1.5, "y": 0.4}); run(hq, 0.5); assert hq.S["target"] is not None
    hq.W.robot_seen = False; run(hq, 1.8); assert hq.S["target"] is not None          # blind for under 2 s: kept, zeros
    run(hq, 0.6); assert hq.S["target"] is None and hq.S["note"] == "target dropped: our robot not seen"
    hq.W.robot_seen = True; run(hq, 0.5); assert hq.S["target"] is None                # it does not resume


def test_a_placement_ends_after_2_s_unseen(hq):
    hook_sim(hq, (-0.6, -0.4, 0.0), (0.0, 0.0), seed=5)
    pl, out = start(hq, 1.0, 0.5); run(hq, 1.0); assert not pl.finished
    hq.W.robot_seen = False; t0 = hq.W.T; run(hq, 3.0)
    assert pl.finished and "our robot not seen" in pl.status and 1.9 <= pl.t_end - t0 <= 2.2
    assert all(c[1:] == (0.0, 0.0, 0.0) for c in hq.W.sent if c[0] > pl.t_end)


def test_only_the_pages_own_polls_feed_the_dead_mans_switch(hq):
    assert "fetch('/state?page=1')" in hq.PAGE and "fetch('/state')" not in hq.PAGE
    hq.S["poll"] = 0.0
    assert get(hq, "/state")[0] == 200 and hq.S["poll"] == 0.0                       # another reader: no refresh
    assert get(hq, "/state?x=1&page=10")[0] == 200 and hq.S["poll"] == 0.0
    assert get(hq, "/state?page=1")[0] == 200 and hq.S["poll"] == hq.W.T              # the page
    assert get(hq, "/state?x=1&page=1")[0] == 200


def test_a_reader_polling_state_does_not_keep_a_target_alive(hq):
    hook_sim(hq, (-0.6, -0.4, 0.0), (1.5, 1.0), seed=6, page=False)
    hq.S["poll"] = hq.W.T; post(hq, "/lock", {"on": False}); post(hq, "/goto", {"x": 1.5, "y": 0.4})
    assert hq.S["target"] is not None
    spin = hq.W.spin; n = [0]
    def spin_and_read():                                                             # a supervisor at 4 Hz, no browser
        spin(); n[0] += 1
        if n[0] % 12 == 0: assert state(hq)["target"] is not None or hq.S["target"] is None
    hq.W.spin = spin_and_read; run(hq, 2.5)
    assert hq.S["target"] is None and hq.S["note"] == "page stopped asking: target dropped" and n[0] > 100
