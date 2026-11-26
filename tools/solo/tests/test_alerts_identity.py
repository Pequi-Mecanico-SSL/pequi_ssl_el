"""Tests of tools/solo/auto_session.py: alerts never delay a stop, our robot told apart from a twin when its tag drops
out, the wait before the drive back, failed alerts retried and a dead alert channel, the alert rate, cameras not on
the list and the merge allowance. Fakes for the field (no robot, ROS or network); the stop test starts a local
`sleep` child.
    uv run --with pytest pytest -q tools/solo/tests/test_alerts_identity.py"""
import math, os, sys, threading, time
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
def swap_session(back=None, back_at=(-1.0, 0.0), budget=120, others=()):
    """a twin (our colour and id) enters at 2 s; our robot's tag is not detected from 20 s; the connector then follows
    the twin. back: from then the twin has left and one robot of our colour and id is seen at back_at (with others)"""
    T = {}
    def frames(t):
        dt = t - T["0"]
        if back is not None and dt >= back:
            world.set(robot=(back_at[0], back_at[1], 0.0)); return {0: [("blue", 3, back_at[0], back_at[1]), *others]}
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
    assert lost[0].endswith("; the session holds: put the robot back at (-1.00, +0.00) or restart the session; the session will try to recognise it by a short turn on the spot") and s.ours_lost is not None

def test_which_robot_is_ours_clears_only_when_ours_is_seen_again_at_its_old_place():
    s, run, note, starts = swap_session(back=60, back_at=(-0.6, 0.0))      # one robot, 0.4 m from where ours stood
    assert len(starts) == 1 and s.done == [] and s.ours_lost is not None
    s, run, note, starts = swap_session(back=60)                           # ours at its old place
    assert s.ours_lost is None and len(starts) == 3 and starts[1] >= 60 + 2 * s.clear_s - 0.3 and s.done == [1, 2]

def test_our_robot_seen_again_at_its_place_after_the_lost_alert_is_alerted_with_the_session_resuming():
    s, run, note, starts = swap_session(back=60)
    lost = [i for i, m in enumerate(alerts(note)) if m.startswith("which robot is ours was lost")]
    back = [i for i, m in enumerate(alerts(note)) if m == "our robot seen again at its place (-1.00, +0.00): the session resumes"]
    assert lost and back and lost[0] < back[0] and s.ours_lost is None and s.done == [1, 2] and not note.quiet

def test_our_robot_seen_continuously_while_a_step_moves_it_is_not_lost():
    script = [dict(match="goto.py", line="arrived: robot at (-0.40, +0.20) after 3.0 s", dur=3.0)]
    s, run, note, world, clk = make_cams([dict(kind="goto", x=-1.6, y=0.5, label="g")], lambda t: {0: [("blue", 3, *world.robot[:2])]},
                                         budget=60, switch_after=0)
    s.run()
    assert s.ours_lost is None and s.done == [1] and alerts(note) == []



# ---- 2b. ordinary dropouts and stops at speed on a clean field: our robot is not lost
REACT, DECEL = 0.15, 3.0                    # s, m/s^2: the simulated robot after a stop (reaction delay, then braking)
START, GOAL = (-2.0, -1.2), (-0.5, 1.2)

def clean_session(speed, drop=None, others=None, react=REACT, decel=DECEL, steps=None, script=None, budget=300, page_sees=False):
    """a goto from START to GOAL at speed (m/s), then a spin. drop (at, dur): from `at` s our robot's tag is not
    detected for dur s (the connector does not see it either: the page keeps its last position; page_sees: the page's
    position stays fresh, as from a connector that still tracks it). others(dt) -> robots
    seen besides ours. -> (session, runner, notifier, clock, step starts, [(distance from the kept position, allowed)])"""
    D = {}
    def frames(t):
        dt = t - D["t0"]; rs = []
        if drop is None or not (drop[0] <= dt < drop[0] + drop[1]): rs.append(("blue", 3, *world.robot[:3]))
        return {0: rs + (others(dt) if others else [])}
    s, run, note, world, clk = make_cams(steps or [dict(kind="goto", x=GOAL[0], y=GOAL[1], label="g"), SPIN(label="after")], frames,
                                         robot=(*START, 0.0), budget=budget, switch_after=0)
    dist = math.hypot(GOAL[0] - START[0], GOAL[1] - START[1])
    run.SPEED, run.REACT, run.DECEL = speed, react, decel
    run.script = script if script is not None else [dict(match="goto.py", line="arrived: robot at (-0.50, +1.20)", dur=dist / speed,
                                                         world={"robot": (*GOAL, 0.0)})]
    snap0 = world.snapshot; F = {}
    def snap():
        d = snap0()
        if drop is not None and drop[0] <= clk.t - D["t0"] < drop[0] + drop[1] and not page_sees:
            F.setdefault("p", (d["robot"], clk.t)); d["robot"] = F["p"][0]; d["robot_age"] = round(clk.t - F["p"][1], 2)
        else: F.clear()
        return d
    world.snapshot = snap
    checks = []; tr = s.track_us
    def track(mine, us, amb):
        if s.ours_lost is None and us is not None and len(mine) == 1 and amb is None and s.us_last is not None:
            checks.append((math.hypot(us[2] - s.us_last["x"], us[3] - s.us_last["y"]), s.id_bound(clk.t)))
        return tr(mine, us, amb)
    s.track_us = track
    D["t0"] = clk.t; starts = []; go = run.run
    run.run = lambda *a, **k: (starts.append(round(clk.t - D["t0"], 2)), go(*a, **k))[1]
    s.run()
    return s, run, note, clk, starts, checks

PROBES = lambda run: [c for c in run.cmds("characterize.py") if " spin +1.50 0.70 " in c or " spin -1.50 0.70 " in c]

def test_a_dropout_or_a_stop_at_speed_on_a_clean_field_keeps_our_robot_or_proves_it_by_a_turn_and_the_session_goes_on():
    cases = {"0.3 s at 1.5 m/s": (1.5, (1.0, 0.3), None), "0.45 s at 1.0 m/s": (1.0, (1.0, 0.45), None),
             "0.8 s at 0.4 m/s": (0.4, (1.0, 0.8), "(-1.61, -0.58)"),   # reappears 0.33 m away: lost, then the probe
             "1.0 s at 0.5 m/s": (0.5, (1.0, 1.0), "(-1.51, -0.42)"),   # 0.42 m away
             "1.0 s at 1.0 m/s": (1.0, (1.0, 1.0), "(-0.98, +0.43)")}   # 0.92 m away
    for name, (v, drop, probed) in cases.items():
        s, run, note, clk, starts, checks = clean_session(v, drop)
        assert s.ours_lost is None and not s.not_ours and s.done == [1, 2], (name, s.done, run.stops)
        lost = [m for m in alerts(note) if "which robot is ours" in m]
        assert all(d <= lim for d, lim in checks), name
        found = [m for m in alerts(note) if m.startswith("our robot recognised")]
        if probed:
            assert len(lost) == 1 and len(PROBES(run)) == 2 and not note.quiet, name
            assert found == [f"our robot recognised by its turn at {probed}: the session resumes"], name
        else: assert not lost and not PROBES(run) and not note.quiet and not found, name
        if drop[1] > A.VISION_FRESH:                        # longer than the raw view keeps a detection: stopped, then repeated
            assert run.stops and run.stops[0][1].startswith("no information: our robot (blue 3) not seen") and len(run.cmds("goto.py")) == 2, name
        else: assert run.stops == [] and len(starts) == 2, name

def test_our_robot_sliding_after_an_intruder_stops_a_fast_drive_is_not_lost_and_the_session_goes_on():
    for react, decel in ((0.0, 5.0), (0.1, 4.0), (REACT, DECEL)):     # slides of 0.23, 0.43 and 0.60 m from 1.5 m/s
        other = lambda dt: [("yellow", 1, -1.9, 1.2)] if 1.0 <= dt < 4.0 else []
        s, run, note, clk, starts, checks = clean_session(1.5, others=other, react=react, decel=decel)
        assert run.stops and run.stops[0][1].startswith("another robot: yellow1"), (react, decel)
        assert s.ours_lost is None and not s.not_ours and s.done == [1, 2] and len(starts) == 3, (react, decel, run.stops)
        assert not [m for m in alerts(note) if "which robot is ours" in m] and all(d <= lim for d, lim in checks)

def test_a_3_s_dropout_during_a_strategy_step_ends_with_our_robot_proved_by_its_turn():
    st = dict(kind="strategy", on=True, seconds=6, side="negative", label="play")
    s0 = {}
    def drive(s, clk, world, v=1.0):                     # the node drives our robot along +y while it runs
        sl = s.sleep
        def sleep(dt):
            was = s.strat.running(); t = clk.t; sl(dt)
            if was and s.strat.running():
                x, y, th = world.robot; world.set(robot=(x, min(1.2, y + v * (clk.t - t)), th))
            elif was and not s.strat.running(): world.coast((0.0, v), REACT, DECEL)
        s.sleep = sleep
    D = {}
    def frames(t):
        dt = t - D["t0"]; return {0: [] if 1.0 <= dt < 4.0 else [("blue", 3, *world.robot[:3])]}
    s, run, note, world, clk = make_cams([st, SPIN(label="after")], frames, robot=(-1.6, -1.2, 0.0), budget=200, switch_after=0)
    snap0 = world.snapshot; F = {}
    def snap():
        d = snap0()
        if 1.0 <= clk.t - D["t0"] < 4.0:
            F.setdefault("p", (d["robot"], clk.t)); d["robot"] = F["p"][0]; d["robot_age"] = round(clk.t - F["p"][1], 2)
        else: F.clear()
        return d
    world.snapshot = snap; drive(s, clk, world); D["t0"] = clk.t
    s.run()
    assert s.ours_lost is None and not s.not_ours and 2 in s.done
    lost = [m for m in alerts(note) if "which robot is ours" in m]          # reappeared 0.6 m away: proved by its turn
    assert len(lost) == 1 and "ours unseen for 3.2 s" in lost[0] and len(PROBES(run)) == 2
    assert not note.quiet and "our robot recognised by its turn at (-1.60, +0.40): the session resumes" in alerts(note)

def test_our_robot_standing_its_tag_dropping_out_and_a_robot_of_our_colour_and_id_seen_0_4_m_away_is_lost():
    D = {}
    def frames(t):
        dt = t - D["t0"]
        if dt >= 12: world.set(robot=(-0.6, 0.0, 0.0)); return {0: [("blue", 3, -0.6, 0.0), ("yellow", 2, -1.8, 1.0)]}
        return {0: [OURS] + ([("yellow", 2, -1.8, 1.0)] if dt >= 2 else [])}
    s, run, note, world, clk = make_cams([SPIN(label="a"), SPIN(label="b")], frames, budget=60, switch_after=0)
    D["t0"] = clk.t; s.run()
    lost = [m for m in alerts(note) if m.startswith("which robot is ours was lost")]
    assert s.ours_lost is not None and s.done == [] and lost and "0.40 m from where ours last stood" in lost[0]
    assert lost[0].endswith("put the robot back at (-1.00, +0.00) or restart the session; the session will try to recognise it by a short turn on the spot")

def test_our_robot_carried_1_m_away_while_the_session_holds_is_lost():
    D = {}
    def frames(t):
        dt = t - D["t0"]; y2 = [("yellow", 2, -1.8, 1.0)] if dt >= 2 else []
        if 10 <= dt < 12: return {0: y2}                                         # lifted: not seen
        if dt >= 12: world.set(robot=(-1.0, -1.0, 0.0)); return {0: [("blue", 3, -1.0, -1.0)] + y2}
        return {0: [OURS] + y2}
    s, run, note, world, clk = make_cams([SPIN(label="a"), SPIN(label="b")], frames, budget=60, switch_after=0)
    D["t0"] = clk.t; s.run()
    assert s.ours_lost is not None and s.done == [] and any(m.startswith("which robot is ours was lost") for m in alerts(note))

def test_a_fresh_page_position_is_not_compared_with_a_raw_detection_its_camera_no_longer_shows():
    s, run, note, clk, starts, checks = clean_session(1.5, (1.0, 0.4), page_sees=True)      # 0.4 s: up to 0.6 m apart
    assert not s.not_ours and s.ours_lost is None and s.done == [1, 2] and run.stops == []
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OURS]}, budget=60)
    q = s.snap(); world.set(robot=(-0.6, 0.0, 0.0)); q = s.snap()                           # both fresh, 0.4 m apart
    assert s.mismatch(q) and s.mismatch(q).startswith("the tools' robot at (-0.60, +0.00) is 0.40 m from blue 3")

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

def test_the_drive_back_alert_waits_the_information_delay_or_fetch_wait_and_a_robot_inside_our_half_goes_out_at_once():
    for frames, after in ((lambda t: {0: [OUT, ("yellow", 2, -1.8, 1.0)]}, 0), (lambda t: {0: [OUT, ("yellow", 1, 0.15, 0.3)]}, 60),
                          (lambda t: {} , 10)):
        s, run, note, world, clk = make_cams([SPIN()], frames, robot=(0.5, 0.0, 0.0), budget=200, switch_after=0)
        t = []; log = s.log
        s.log = lambda kind, **k: (t.append(clk.t) if kind == "alert" and "outside our half" in k.get("message", "") else None, log(kind, **k))[1]
        s.ensure_home_clear = lambda inside_only=False: True             # the hold before the wait is tested elsewhere
        t0 = clk.t; s.drive_home(s.look(), "return_home")
        assert t and max(0, after - 0.1) <= t[0] - t0 <= after + 0.6, after


def test_a_robot_entering_our_half_while_our_robot_waits_outside_is_alerted_at_once():
    T = {}
    def frames(t):
        rs = [OUT, ("yellow", 1, 0.15, 0.3)]                               # blocks the way back the whole time
        if t - T.get("0", t) >= 70: rs.append(("yellow", 4, -1.0, 0.5))           # a new robot enters our half at 70 s and stays
        return {0: rs}
    s, run, note, world, clk = make_cams([SPIN()], frames, robot=(0.5, 0.0, 0.0), budget=600, switch_after=0)
    assert s.preflight() is None
    T["0"] = clk.t; rec = []; log = s.log
    s.log = lambda kind, **k: (rec.append((clk.t - T["0"], k.get("message", ""))) if kind == "alert" else None, log(kind, **k))[1]
    s.ensure_home_clear = lambda inside_only=False: True
    assert s.drive_home(s.look(), "return_home") is False
    y4 = [(t, m) for t, m in rec if "(yellow 4) entered our half" in m]
    assert rec[0][1].startswith("another robot (yellow 1 at (+0.15, +0.30)) blocks the way back") and 59.9 <= rec[0][0] <= 60.6
    assert y4 and 70 <= y4[0][0] <= 70.6 and y4[0][1] == ("another robot (yellow 4) entered our half at (-1.00, +0.50); our robot at "
                                                        "(+0.50, +0.00), outside our half: the session waits")

def test_which_robot_is_ours_lost_while_our_robot_waits_outside_is_alerted_at_once():
    T = {}
    def frames(t):
        if t - T.get("0", t) >= 30: world.set(robot=(0.9, 0.0, 0.0)); return {0: [("blue", 3, 0.9, 0.0), ("yellow", 1, 0.15, 0.3)]}
        if t - T.get("0", t) >= 28: return {0: [("yellow", 1, 0.15, 0.3)]}
        return {0: [OUT, ("yellow", 1, 0.15, 0.3)]}
    s, run, note, world, clk = make_cams([SPIN()], frames, robot=(0.5, 0.0, 0.0), budget=200, switch_after=0)
    assert s.preflight() is None
    T["0"] = clk.t; rec = []; log = s.log
    s.log = lambda kind, **k: (rec.append((clk.t - T["0"], k.get("message", ""))) if kind == "alert" else None, log(kind, **k))[1]
    s.ensure_home_clear = lambda inside_only=False: True
    s.drive_home(s.look(), "return_home")
    lost = [(t, m) for t, m in rec if m.startswith("which robot is ours was lost")]
    assert lost and 30 <= lost[0][0] <= 30.6 and lost[0][1].endswith(       # this wait runs no probe: none promised
        "; our robot at (+0.50, +0.00), outside our half: the session waits; put the robot back at (+0.50, +0.00) or restart the session")

def test_session_ended_says_it_ended_holding_since_when_for_what_and_where_our_robot_stands():
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OUT, ("yellow", 2, -1.8, 1.0)]}, robot=(0.5, 0.0, 0.0), budget=60, switch_after=0)
    t0 = clk.t; s.run()
    end = [m for m in note.sent if m.startswith("session ended")][-1]
    since = A.time.strftime("%H:%M", A.time.localtime(t0))
    assert (f"; ended while holding since {since}: another robot (yellow 2) entered our half at (-1.80, +1.00)"
            ", our robot at (+0.50, +0.00), outside our half") in end
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OURS]}, budget=60, switch_after=0)
    s.run()
    assert "holding" not in [m for m in note.sent if m.startswith("session ended")][-1]

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


def test_an_alert_command_that_never_returns_makes_the_channel_dead_after_alert_dead_s():
    gate = threading.Event()
    class Hang:
        def send(self, msg, title): gate.wait(10); return "rc 0"
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OURS]}, budget=900, alert_thread=True)
    s.notifier = Hang(); s.alert("another robot (yellow 2) entered our half", kind="intruder", robot="yellow2")
    try:
        for _ in range(301):
            clk.sleep(1.0); s.alert_poll()
            if s.dead is not None: break
        assert s.dead is not None and s.no_info(s.snap()).endswith(": the alert command fails")
        assert 299 <= s.dead["since"] - s.dead["fail_since"] <= 301
    finally: gate.set()

def test_the_alert_sender_thread_ending_is_restarted_once_then_the_channel_is_dead():
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OURS]}, budget=900, alert_thread=True)
    rec = []; log = s.log
    s.log = lambda kind, **k: (rec.append(kind), log(kind, **k))[1]
    for n in (1, 2):                                        # the thread ends (as on an error it does not catch)
        s.sender.halt.set(); s.sender.thread.join(2); assert not s.sender.thread.is_alive(); s.sender.halt.clear()
        s.alert_poll()
        if n == 1: assert s.sender.alive() and s.dead is None
    assert rec.count("alert_sender_restarted") == 1 and "alert_channel_dead" in rec and s.dead["thread"]
    assert s.no_info(s.snap()).endswith(": the alert sender stopped")

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


# ---- 7. a robot seen in a single frame
def test_another_robot_seen_in_a_single_frame_does_not_stop_a_step_and_one_seen_in_two_frames_does():
    for frames_with_it, stops in ((1, False), (2, True)):
        T = {"n": 0}
        def frames(t):
            rs = [OURS]
            if T.get("at") is not None and t >= T["at"] and T["n"] < frames_with_it:
                T["n"] += 1; rs.append(("yellow", 2, -0.9, 0.15))              # a misread id next to ours
            return {0: rs}
        s, run, note, world, clk = make_cams([SPIN(label="spin")], frames, budget=60, switch_after=0)
        T["at"] = clk.t + 1.0; s.run()
        assert bool(run.stops) == stops and s.done == [1] and len(run.cmds("characterize.py")) == (2 if stops else 1), frames_with_it
        assert bool([m for m in alerts(note) if "yellow 2" in m]) == stops, frames_with_it

def test_a_single_frame_sighting_counts_when_its_camera_sends_no_newer_frame_within_0_1_s():
    clk = A.SimClock(); q = []
    class Src:
        def read(self, timeout): return q.pop(0) if q else None
    v = A.RawVision(Src(), clk, thread=False)
    single = lambda: [d[:2] for d in v.snapshot()["vision_unconfirmed"] if d[0] == "yellow"]
    q.append((0, [("yellow", 2, -0.5, 0.0)])); assert single() == [("yellow", 2)]
    clk.t += 0.05; assert single() == [("yellow", 2)]
    clk.t += 0.051; assert single() == []                                 # no newer frame from camera 0 within 0.1 s
    q.append((1, [("yellow", 4, 0.5, 0.0)])); assert single() == [("yellow", 4)]
    clk.t += 0.02; q.append((1, [])); clk.t += 0.2
    assert single() == [("yellow", 4)]                                     # a newer frame of its camera without it
    q.append((1, [("yellow", 5, 0.5, 0.5)])); single(); clk.t += 0.017; q.append((1, [("yellow", 5, 0.5, 0.5)]))
    assert ("yellow", 5) not in single()                                   # two consecutive frames



# ---- 8. a robot detected in a flickering way
def flicker_view(pattern, n=120):
    """a RawVision fed n frames at 60 Hz, pattern(k) -> the robots of frame k; -> [(frame, colour + ids counted)]"""
    clk = A.SimClock(); q = []
    class Src:
        def read(self, timeout): return q.pop(0) if q else None
    v = A.RawVision(Src(), clk, thread=False); v.own = ("blue", 3); out = []
    for k in range(n):
        q.append((0, pattern(k))); sn = v.snapshot()
        out.append((k, sorted({d[:2] for d in sn["vision_dets"] if d not in sn["vision_unconfirmed"] and d[0] != "blue"})))
        clk.t += 1 / 60
    return out

def test_a_robot_seen_in_every_other_frame_is_counted_within_0_1_s_and_stays_counted():
    out = flicker_view(lambda k: [OURS] + ([("yellow", 2, -0.5, 0.0)] if k >= 10 and k % 2 == 0 else []))
    counted = [k for k, ids in out if ("yellow", 2) in ids]
    assert counted[0] - 10 <= 6 and counted == list(range(counted[0], 120))

def test_a_robot_whose_id_is_read_differently_frame_by_frame_is_counted():
    out = flicker_view(lambda k: [OURS] + ([("yellow", 2 if k % 2 else 5, -0.5, 0.0)] if k >= 10 else []))
    assert all(ids for k, ids in out if k >= 12)

def test_one_isolated_false_detection_is_not_counted():
    out = flicker_view(lambda k: [OURS] + ([("yellow", 2, 0.5, -0.5)] if k == 30 else []))
    assert all(ids == [] for k, ids in out)
    out = flicker_view(lambda k: [OURS] + ([("yellow", 2, -0.9, 0.1)] if k == 30 else []))    # next to ours: ours does not confirm it
    assert all(ids == [] for k, ids in out)

def test_a_confirmed_robot_missing_one_frame_stays_confirmed():
    out = flicker_view(lambda k: [OURS] + ([("yellow", 2, -0.5, 0.0)] if k >= 10 and k != 40 else []))
    assert all(ids == [("yellow", 2)] for k, ids in out if k >= 11)
    out = flicker_view(lambda k: [OURS] + ([("yellow", 2, -0.5, 0.0)] if 10 <= k < 20 or k >= 45 else []))   # 0.42 s gap: still kept
    assert all(ids == [("yellow", 2)] for k, ids in out if k >= 11)

def test_a_robot_detected_only_now_and_then_is_counted_from_its_second_sighting():
    for every in (20, 50):                                    # 0.33 s and 0.82 s between its detections
        out = flicker_view(lambda k: [OURS] + ([("yellow", 2, -0.5 + 0.002 * k, 0.0)] if k >= 10 and (k - 10) % every == 0 else []), n=150)
        assert [k for k, ids in out if ids][0] == 10 + every, every

def test_two_unrelated_single_detections_2_m_apart_within_1_s_are_not_counted():
    for gap in (18, 30, 48):                                  # 0.3, 0.5 and 0.8 s apart
        out = flicker_view(lambda k: [OURS] + ([("yellow", 2, -2.0, -1.0)] if k == 30 else [("yellow", 2, 0.0, -1.0)] if k == 30 + gap else []))
        assert all(ids == [] for k, ids in out), gap

def test_a_robot_detected_only_now_and_then_inside_our_half_stops_a_step():
    for every in (20, 50):
        T = {}
        def frames(t, every=every):
            k = round((t - T.get("at", 1e18)) * 60)
            return {0: [OURS] + ([("yellow", 2, -0.5, 0.0)] if k >= 0 and k % every == 0 else [])}
        s, run, note, world, clk = make_cams([SPIN(label="spin")], frames, budget=60, switch_after=0)
        T["at"] = clk.t + 1.0; s.run()
        assert run.stops and run.stops[0][1].startswith("another robot: yellow2"), every
        assert every / 60 - 0.02 <= run.stops[0][0] - T["at"] <= every / 60 + 0.2, every

def test_a_robot_seen_in_every_other_frame_stops_a_step():
    T = {}
    def frames(t):
        k = round((t - T.get("at", 1e18)) * 60)
        return {0: [OURS] + ([("yellow", 2, -0.5, 0.0)] if k >= 0 and k % 2 == 0 else [])}
    s, run, note, world, clk = make_cams([SPIN(label="spin")], frames, budget=60, switch_after=0)
    T["at"] = clk.t + 1.0; s.run()
    assert run.stops and run.stops[0][1].startswith("another robot: yellow2") and run.stops[0][0] - T["at"] <= 0.2
