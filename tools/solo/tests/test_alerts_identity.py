"""Tests of tools/solo/auto_session.py: alerts never delay a stop, our robot told apart from a twin when its tag drops
out, the wait before the drive back, failed alerts retried and a dead alert channel, the alert rate, cameras not on
the list and the merge allowance. Fakes for the field (no robot, ROS or network); the stop test starts a local
`sleep` child.
    uv run --with pytest pytest -q tools/solo/tests/test_alerts_identity.py"""
import os, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import auto_session as A  # noqa: E402
from test_home_area import SPIN, alerts  # noqa: E402
from test_home_safety import make_cams, OURS, TWIN  # noqa: E402


def gone(pid):
    """the process has ended (a zombie or reaped)"""
    try:
        with open(f"/proc/{pid}/stat") as f: return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError: return True


# ---- 1. stopping never waits for an alert
class BlockingNotifier:
    """an alert command that hangs (an unreachable alert service)"""
    def __init__(self, block): self.block, self.sent = block, []
    def send(self, msg, title):
        self.sent.append((time.time(), msg)); time.sleep(self.block); return "failed: error timeout; error timeout"

def test_the_step_stops_within_half_a_second_of_a_mismatch_while_the_alert_command_blocks():
    st = SPIN(label="spin")
    s, run, note, world, clk = make_cams([st], lambda t: {0: [OURS]}, budget=60, alert_thread=True)
    assert s.preflight() is None and s.sender.threaded
    s.notifier = BlockingNotifier(3.0); s.runner = A.Runner(echo=False)
    w = s.watch_for(st); k = [0]; t_mm = [None]; t_dead = [None]
    def watch():
        k[0] += 1
        if k[0] == 5: world.set(robot=(-0.5, 0.0, 0.0)); t_mm[0] = time.time()     # the connector follows another robot
        return w()
    def monitor():
        while t_dead[0] is None:
            pid = s.runner.child_pid() if s.runner.p is not None else None
            if t_mm[0] is not None and (pid is None or gone(pid)): t_dead[0] = time.time()
            time.sleep(0.01)
    th = threading.Thread(target=monitor, daemon=True); th.start()
    r = s.runner.run(["sleep", "30"], dict(os.environ), 60, watch=watch); th.join(5)
    assert r["stopped"].startswith("not ours") and s.not_ours
    assert t_dead[0] - t_mm[0] <= 0.5 and r["t1"] - t_mm[0] < 1.5
    for _ in range(100):
        if s.notifier.sent: break
        time.sleep(0.02)
    assert s.notifier.sent and s.notifier.sent[0][1].startswith("the robot the tools are driving is not ours")


# ---- 2. a twin must not become ours when our robot's tag drops out
def swap_session(back=None, back_at=(-1.0, 0.0), budget=120):
    """a twin (our colour and id) enters at 2 s; our robot's tag is not detected from 20 s; the connector then follows
    the twin. back: from then the twin has left and one robot of our colour and id is seen at back_at"""
    T = {}
    def frames(t):
        dt = t - T["0"]
        if back is not None and dt >= back:
            world.set(robot=(back_at[0], back_at[1], 0.0)); return {0: [("blue", 3, back_at[0], back_at[1])]}
        rs = [] if dt >= 20 else [OURS]
        rs = rs + ([TWIN] if dt >= 2 else [])
        if dt >= 20: world.set(robot=(TWIN[2], TWIN[3], 0.0))
        return {0: rs}
    s, run, note, world, clk = make_cams([SPIN(label="a"), SPIN(label="b")], frames, budget=budget, switch_after=0)
    T["0"] = clk.t; starts = []; go = run.run
    run.run = lambda *a, **k: (starts.append(clk.t - T["0"]), go(*a, **k))[1]
    s.run()
    return s, run, note, starts

def test_a_twin_left_alone_after_our_tag_drops_out_holds_without_clearing_and_says_which_robot_is_ours_was_lost():
    s, run, note, starts = swap_session()
    lost = [m for m in alerts(note) if m.startswith("which robot is ours was lost")]
    assert len(starts) == 1 and s.done == [] and not s.not_ours           # nothing ran after the twin came in
    assert lost and "after two robots were seen as ours, the one left is blue 3 at (-0.60, +0.40)" in lost[0]
    assert lost[0].endswith("; the session holds") and s.ours_lost is not None

def test_which_robot_is_ours_clears_only_when_ours_is_seen_again_at_its_old_place():
    s, run, note, starts = swap_session(back=60, back_at=(-0.6, 0.0))      # one robot, 0.4 m from where ours stood
    assert len(starts) == 1 and s.done == [] and s.ours_lost is not None
    s, run, note, starts = swap_session(back=60)                           # ours at its old place
    assert s.ours_lost is None and len(starts) == 3 and starts[1] >= 60 + 2 * s.clear_s - 0.3 and s.done == [1, 2]

def test_our_robot_seen_continuously_while_a_step_moves_it_is_not_lost():
    script = [dict(match="goto.py", line="arrived: robot at (-0.40, +0.20) after 3.0 s", dur=3.0)]
    s, run, note, world, clk = make_cams([dict(kind="goto", x=-1.6, y=0.5, label="g")], lambda t: {0: [("blue", 3, *world.robot[:2])]},
                                         budget=60, switch_after=0)
    s.run()
    assert s.ours_lost is None and s.done == [1] and alerts(note) == []


# ---- 3. the wait before the drive back names its reason and where our robot stands
OUT = ("blue", 3, 0.5, 0.0)                                                # outside the home area (x > -0.10)

def test_the_wait_before_the_drive_back_alerts_with_its_reason_and_where_our_robot_stands():
    cases = {"no vision": (lambda t, on: {} if on else {0: [OUT]}, 10, "no vision for 10.0 s (camera 0)"),
             "a twin": (lambda t, on: {0: [OUT] + ([("blue", 3, -1.5, 0.5)] if on else [])}, 10,
                        "two robots are seen as blue 3: at (-1.50, +0.50) and (+0.50, +0.00), the tools cannot tell which is ours"),
             "a robot inside": (lambda t, on: {0: [OUT] + ([("yellow", 2, -1.8, 1.0)] if on else [])}, 10,
                                "another robot (yellow 2) entered our half at (-1.80, +1.00)"),
             "on the way": (lambda t, on: {0: [OUT, ("yellow", 1, 0.15, 0.3)]}, 60,
                            "another robot (yellow 1 at (+0.15, +0.30)) blocks the way back")}
    for name, (f, after, why) in cases.items():
        T = [1e18]
        s, run, note, world, clk = make_cams([SPIN()], lambda t, f=f: f(t, t >= T[0]), robot=(0.5, 0.0, 0.0), budget=400, switch_after=0)
        assert s.preflight() is None, name
        t0 = clk.t; T[0] = clk.t + 0.05
        assert s.drive_home(s.look(), "return_home") is False and run.calls == [], name
        a = [m for m in alerts(note)]
        assert a[0] == f"{why}; our robot at (+0.50, +0.00), outside our half: the session waits", (name, a[0])
        rec_t = [r for r in s.alerts if r == a[0]]
        assert rec_t, name
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OUT, ("blue", 3, -1.5, 0.5)]}, robot=(0.5, 0.0, 0.0), budget=40, switch_after=0)
    s.ensure_home_clear()                                                  # a hold alert while our robot is outside
    assert alerts(note)[0].endswith("which is ours, our robot at (+0.50, +0.00), outside our half; the session holds")

def test_the_drive_back_alert_waits_the_information_delay_or_fetch_wait():
    for frames, after in ((lambda t: {0: [OUT, ("yellow", 2, -1.8, 1.0)]}, 10), (lambda t: {0: [OUT, ("yellow", 1, 0.15, 0.3)]}, 60)):
        s, run, note, world, clk = make_cams([SPIN()], frames, robot=(0.5, 0.0, 0.0), budget=200, switch_after=0)
        t = []; log = s.log
        s.log = lambda kind, **k: (t.append(clk.t) if kind == "alert" and "outside our half" in k.get("message", "") else None, log(kind, **k))[1]
        s.ensure_home_clear = lambda inside_only=False: True             # the hold before the wait is tested elsewhere
        t0 = clk.t; s.drive_home(s.look(), "return_home")
        assert t and after - 0.1 <= t[0] - t0 <= after + 0.6


# ---- 4. a failed alert does not count as sent
def test_failed_alerts_are_retried_in_order_and_motion_holds_once_none_was_delivered_for_alert_dead():
    NET = {"up": True}; T = {"in": 1e18, "out": 1e18}
    class Note:
        def __init__(self): self.tries = []
        def send(self, msg, title):
            self.tries.append((round(clk.t - t0, 1), NET["up"], msg)); return "rc 0" if NET["up"] else "failed: error timeout; error timeout"
    frames = lambda t: {0: [OURS] + ([("yellow", 1, -0.5, -1.0)] if T["in"] <= t < T["out"] else [])}
    s, run, note, world, clk = make_cams([SPIN(label=f"s{i}") for i in range(300)], frames, budget=900, switch_after=0)
    s.notifier = Note(); t0 = clk.t; orig = s.preflight; starts = []; go = run.run
    run.run = lambda *a, **k: (starts.append(clk.t - t0), go(*a, **k))[1]
    def pf():
        r = orig(); NET["up"] = False; T["in"] = clk.t + 3.0; T["out"] = clk.t + 30.0; return r     # the network drops after the start
    s.preflight = pf
    tl = [(700.0, lambda: NET.update(up=True))]
    sl = s.sleep
    def sleep(dt):
        while tl and clk.t - t0 >= tl[0][0]: tl.pop(0)[1]()
        sl(dt)
    s.sleep = sleep
    rec = []; log = s.log
    s.log = lambda kind, **k: (rec.append((round(clk.t - t0, 1), kind)), log(kind, **k))[1]
    s.run()
    down = [t for t, up, m in s.notifier.tries if not up]
    assert [round(b - a) for a, b in zip(down, down[1:])][:4] == [15, 30, 60, 60]
    assert sum(1 for t, k in rec if k == "notify_failed") == len(down)
    dead = [t for t, k in rec if k == "alert_channel_dead"]
    assert dead and 300 <= dead[0] - down[0] <= 301
    assert not [t for t in starts if dead[0] + 1 < t < 700] and [t for t in starts if 30 < t < dead[0]]   # held while dead
    assert [t for t in starts if t > 700]                                                             # and resumed
    up = [m for t, ok, m in s.notifier.tries if ok and t >= 700]
    assert up[0].startswith("no alert could be delivered from ") and "the session held all motion (robot standing)" in up[0]
    assert any(m.startswith("1 earlier alert was not delivered since ") for m in up)                # the intruder alert, too old
    assert up[-1].split(": ", 1)[-1].startswith("session ended")


# ---- 5. the alert rate
class F:
    """a scenario from t0 on; before it our robot alone"""
    def __init__(self, f): self.f, self.t0 = f, None
    def __call__(self, t): return {0: [OURS]} if self.t0 is None or t < self.t0 else self.f(t - self.t0)

FLOOD = {"one robot in 6 s / out 6 s": (lambda t: {0: [OURS] + [("yellow", 1, -0.5, -1.0) if t % 12 < 6 else ("yellow", 1, 1.0, -1.0)]}, 0.5),
         "six robots in phase 6 s / 6 s": (lambda t: {0: [OURS] + [("yellow", k, -0.5, -1.2 + 0.4 * k) if t % 12 < 6 else ("yellow", k, 1.0, -1.2 + 0.4 * k)
                                                                  for k in range(6)]}, 0.5),
         "a twin in 6 s / out 6 s": (lambda t: {0: [OURS] + ([("blue", 3, 1.0, 0.0)] if t % 12 < 6 else [])}, 0.5),
         "camera silent 6 s of every 12 s": (lambda t: {} if t % 12 < 6 else {0: [OURS]}, 11.0)}    # 0.5 s stale + info_alert

def test_alert_totals_over_600_s_stay_low_and_the_first_alert_of_a_problem_goes_out_at_once():
    for name, (f, first) in FLOOD.items():
        fr = F(f)
        s, run, note, world, clk = make_cams([SPIN(label=f"s{i}") for i in range(200)], fr, budget=600, switch_after=0)
        t = []; log = s.log
        s.log = lambda kind, t=t, log=log, **k: (t.append(clk.t) if kind == "alert" else None, log(kind, **k))[1]
        fr.t0 = clk.t + 2.0; s.run()
        assert len(note.sent) <= 15, (name, len(note.sent))
        assert t[1] - fr.t0 <= first, (name, t[1] - fr.t0)               # t[0]: the start alert


# ---- 6. cameras not on the list, the merge allowance
def test_a_hold_for_a_camera_not_on_the_list_says_so():
    late = [1e18]
    s, run, note, world, clk = make_cams([SPIN(label="a")], lambda t: {0: [OURS], **({2: []} if late[0] <= t < late[0] + 2 else {})},
                                         cameras=(0,), budget=60, switch_after=0)
    late[0] = clk.t + 2.0; s.run()
    assert any("(camera 2, not on the camera list)" in m for m in alerts(note))

def test_a_sighting_its_camera_no_longer_shows_merges_within_0_8_m_at_most():
    det = lambda d, a: [("blue", 3, -1.0, 0.0, 0.0, 0, False), ("blue", 3, -1.0 + d, 0.0, a, 1, True)]
    assert len(A.merge_dets(det(0.5, 0.2))) == 1 and len(A.merge_dets(det(0.75, 0.45))) == 1
    assert len(A.merge_dets(det(1.0, 0.45))) == 2 and len(A.merge_dets(det(0.85, 0.45))) == 2
