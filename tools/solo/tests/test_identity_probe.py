"""Tests of tools/solo/auto_session.py: our robot's identity after its tag drops out (continuity, reappearance, lost),
the identity probe (a commanded turn on the spot watched in the raw vision), the shorter clear time of a hold whose
only cause was our robot not seen, and the end alert when the alert sender has stopped. Fakes for the field (no
robot, ROS or network): the simulated robot turns when a spin child runs; a twin does not.
    uv run --with pytest pytest -q tools/solo/tests/test_identity_probe.py"""
import math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import auto_session as A  # noqa: E402
from test_home_area import SPIN, alerts  # noqa: E402
from test_home_safety import make_cams, OURS  # noqa: E402
from test_alerts_identity import clean_session, PROBES  # noqa: E402

START, GOAL = (-2.0, -1.2), (-0.5, 1.2)
NOT_OURS = "did not follow a commanded turn: it is not ours, or ours does not respond; the session holds"


def probe_children(run):
    """[(start time, rate)] of the probe children the runner was asked to run"""
    return [(c["t"], float(c["cmd"].split(" spin ")[1].split()[0])) for c in run.calls if c["cmd"] in PROBES(run)]

def timed_calls(run, clk, on_run=None):
    """every call of the runner gets its start time; on_run(argv) is called as a child starts"""
    go = run.run
    def r(argv, env, timeout, tee=None, watch=None):
        if on_run: on_run(argv)
        t = clk.t; out = go(argv, env, timeout, tee=tee, watch=watch); run.calls[-1]["t"] = t; return out
    run.run = r

def twin_late(twin_at, t_twin=20.0, probes=6, twin_turn=None, budget=200, on_run=None):
    """a goto at 1 m/s; our robot's tag is not detected from 1 s on, for good (the connector keeps its last position);
    from t_twin a robot of our colour and id stands at twin_at (the connector then follows it). twin_turn(t) -> the
    twin's heading (default: it stands still)"""
    D = {}
    def frames(t):
        dt = t - D["t0"]; rs = []
        if dt < 1.0: rs.append(("blue", 3, *world.robot[:3]))
        if dt >= t_twin: rs.append(("blue", 3, *twin_at, twin_turn(t) if twin_turn else 0.3))
        return {0: rs}
    s, run, note, world, clk = make_cams([dict(kind="goto", x=GOAL[0], y=GOAL[1], label="g"), SPIN(label="after")], frames,
                                         robot=(*START, 0.0), budget=budget, switch_after=0, probes=probes)
    dist = math.hypot(GOAL[0] - START[0], GOAL[1] - START[1])
    run.script = [dict(match="goto.py", line="arrived", dur=dist, world={"robot": (*GOAL, 0.0)})]
    snap0 = world.snapshot; F = {}
    def snap():
        d = snap0(); dt = clk.t - D["t0"]
        if dt >= t_twin: d["robot"] = (*twin_at, 0.0); d["robot_age"] = 0.0
        elif dt >= 1.0:
            F.setdefault("p", (d["robot"], clk.t)); d["robot"] = F["p"][0]; d["robot_age"] = round(clk.t - F["p"][1], 2)
        return d
    world.snapshot = snap; timed_calls(run, clk, lambda argv: on_run and on_run(argv, clk)); D["t0"] = clk.t
    s.run()
    return s, run, note, clk, D["t0"]


# ---- 1. identity after a dropout
def test_a_twin_appearing_late_where_ours_vanished_at_speed_is_never_taken_as_ours():
    for twin_at in ((-1.0, 0.5), (0.5, 0.5), (1.0, 1.0), (-2.0, 1.2)):          # 0.97 to 2.6 m from where ours vanished
        s, run, note, clk, t0 = twin_late(twin_at)
        assert s.ours_lost is not None and s.done == [] and not s.not_ours, twin_at
        assert len(run.cmds("goto.py")) == 1 and not [c for c in run.cmds("characterize.py") if " spin 3.0 " in c], twin_at
        lost = [m for m in alerts(note) if m.startswith("which robot is ours was lost")]
        assert lost and "ours unseen for 19." in lost[0], twin_at

def test_a_second_copy_seen_before_ours_drops_out_makes_a_reappearance_within_0_5_m_lost():
    P = {}
    def others(dt):                                        # a twin 0.5 m from ours, seen while ours still is
        if dt >= 1.0:
            P.setdefault("p", (-2.0 + 0.53 + 0.5, -1.2 + 0.85)); return [("blue", 3, *P["p"], 0.0)]
        return []
    s, run, note, clk, starts, checks = clean_session(1.0, drop=(1.2, 8.0), others=others)
    assert s.done == [] and (s.ours_lost is not None or s.not_ours)
    assert s.us_last is None or math.hypot(s.us_last["x"] - P["p"][0], s.us_last["y"] - P["p"][1]) > 0.1   # never anchored on the twin

def test_our_robot_reappearing_within_0_5_m_after_a_long_dropout_is_ours():
    D = {}
    def frames(t):
        dt = t - D["t0"]
        if dt >= 4.0: world.set(robot=(-0.7, 0.0, 0.0))       # moved 0.3 m while unseen (the connector follows it)
        if 2.0 <= dt < 6.0: return {0: []}                 # 4 s unseen while a spin runs
        return {0: [("blue", 3, *world.robot[:3])]}
    s, run, note, world, clk = make_cams([SPIN(label="a"), SPIN(label="b")], frames, budget=120, switch_after=0)
    D["t0"] = clk.t
    world.set(robot=(-1.0, 0.0, 0.0)); s.run()
    assert s.ours_lost is None and s.done == [1, 2] and not [m for m in alerts(note) if "which robot is ours" in m]


# ---- 3. the identity probe
def test_lost_after_a_dropout_at_speed_beyond_0_5_m_is_resolved_by_the_probe_and_the_session_goes_on():
    s, run, note, clk, starts, checks = clean_session(1.0, (1.0, 1.0))
    pc = [c for c in run.calls if c["cmd"] in PROBES(run)]
    assert s.ours_lost is None and s.done == [1, 2] and s.probes_run == 1 and len(pc) == 2
    assert sorted(c["cmd"].split(" spin ")[1].split()[0] for c in pc) == ["+1.50", "-1.50"]
    assert all(c["watched"] and c["env"]["SOLO_BOX"] == "-1.23,-0.73,0.18,0.68" for c in pc)
    assert note.quiet == ["our robot recognised by its turn at (-0.98, +0.43)"]

def test_a_twin_appearing_late_fails_the_probe_and_the_session_holds_with_the_alert():
    s, run, note, clk, t0 = twin_late((-1.0, 0.5))
    fail = [m for m in alerts(note) if NOT_OURS in m]
    assert s.probes_run >= 1 and fail and fail[0] == f"the robot seen as blue 3 at (-1.00, +0.50) {NOT_OURS}"
    assert s.ours_lost is not None and s.done == [] and not note.quiet
    ch = probe_children(run)
    assert ch[0][0] - t0 >= 20.0 + s.clear_s - 0.2                   # after it stood still for the clear time
    assert all(b - a >= 60.0 for a, b in zip([t for t, r in ch[::2]], [t for t, r in ch[2::2]]))   # 60 s per place

def test_a_twin_that_happens_to_turn_one_way_fails_the_probe():
    T = {}
    def turn(t):                                           # turns at +1.5 rad/s whenever a probe child runs
        th = 0.3
        for t1, r in T.get("ch", []):
            th += 1.5 * min(max(t - t1 - 1.0, 0.0), 0.7)
        return th
    def on_run(argv, clk):
        if "spin" in argv and argv[argv.index("spin") + 2] == "0.70": T.setdefault("ch", []).append((clk.t, 1.5))
    s, run, note, clk, t0 = twin_late((-1.0, 0.5), twin_turn=turn, on_run=on_run)
    assert T["ch"] and [m for m in alerts(note) if NOT_OURS in m] and s.ours_lost is not None and s.done == [] and not note.quiet

def test_another_robot_entering_during_the_probe_stops_it():
    T = {}
    def others(dt):                                        # enters our half 0.3 s into the first probe child, for 3 s
        if T.get("probe") and "in" not in T: T["in"] = dt + 0.3
        return [("yellow", 1, -1.2, 1.0)] if "in" in T and T["in"] <= dt < T["in"] + 3.0 else []
    go = A.SimRunner.run
    def run_(self, argv, env, timeout, tee=None, watch=None):
        if "spin" in argv and argv[argv.index("spin") + 2] == "0.70": T["probe"] = True
        return go(self, argv, env, timeout, tee=tee, watch=watch)
    A.SimRunner.run = run_
    try: s, run, note, clk, starts, checks = clean_session(1.0, (1.0, 1.0), others=others)
    finally: A.SimRunner.run = go
    stopped = [w for t, w in run.stops if w.startswith("another robot: yellow1")]
    assert stopped and s.probes_run == 2 and s.ours_lost is None and s.done == [1, 2]
    assert note.quiet == ["our robot recognised by its turn at (-0.98, +0.43)"]

def test_the_probes_of_a_session_are_limited_and_the_hold_then_says_so():
    s, run, note, clk, t0 = twin_late((-1.0, 0.5), probes=2, budget=400)
    assert s.probes_run == 2 and len(probe_children(run)) == 4 and s.ours_lost is not None and s.done == []
    last = [m for m in alerts(note) if NOT_OURS in m][-1]
    assert last.endswith("(no identity probe left: 2 used)")

def test_probes_0_switches_the_probe_off_and_lost_holds():
    s, run, note, clk, t0 = twin_late((-1.0, 0.5), probes=0)
    assert s.probes_run == 0 and not PROBES(run) and s.ours_lost is not None and s.done == []
    D = {}
    def frames(t):
        dt = t - D["t0"]; return {0: [] if 2.0 <= dt < 3.0 else [("blue", 3, -1.0 + (0.8 if dt >= 3.0 else 0.0), 0.0, 0.0)]}
    s, run, note, world, clk = make_cams([SPIN(label="a")], frames, budget=60, switch_after=0, probes=0)
    D["t0"] = clk.t; s.run()
    assert s.ours_lost is not None and s.done == [] and not PROBES(run)

def test_the_probe_does_not_start_while_another_robot_is_near_the_candidate_or_a_second_copy_is_seen():
    for extra in ([("yellow", 2, 0.4, 0.5, 0.0)], [("blue", 3, 0.6, -1.0, 0.0)]):     # beside it (outside our half); a second copy
        D = {}
        def frames(t, extra=extra):
            dt = t - D["t0"]
            if dt < 1.0: return {0: [("blue", 3, *world.robot[:3])]}
            return {0: [("blue", 3, -0.2, 0.5, 0.0)] + (extra if dt >= 20 else [])}
        s, run, note, world, clk = make_cams([dict(kind="goto", x=GOAL[0], y=GOAL[1], label="g")], frames, robot=(*START, 0.0),
                                             budget=120, switch_after=0)
        run.script = [dict(match="goto.py", line="arrived", dur=2.8, world={"robot": (*GOAL, 0.0)})]
        snap0 = world.snapshot
        def snap():
            d = snap0()
            if clk.t - D["t0"] >= 1.0: d["robot"] = (-0.2, 0.5, 0.0); d["robot_age"] = 0.0
            return d
        world.snapshot = snap; D["t0"] = clk.t; timed_calls(run, clk)
        s.run()
        late = [t for t, r in probe_children(run) if t - D["t0"] >= 20]
        assert not late and s.ours_lost is not None, extra


# ---- 4. a hold caused only by our robot unseen
def restart_after(frames_extra=lambda dt: [], gap=(2.0, 3.0), camera_off=False):
    """a spin with our robot standing; its tag (or the whole camera) is not seen during gap -> the time from the end
    of the gap to the start of the repeated step"""
    D = {}
    def frames(t):
        dt = t - D["t0"]
        if gap[0] <= dt < gap[1]: return {} if camera_off else {0: frames_extra(dt)}
        return {0: [("blue", 3, -1.0, 0.0, 0.0)] + frames_extra(dt)}
    s, run, note, world, clk = make_cams([SPIN(label="a")], frames, budget=60, switch_after=0)
    timed_calls(run, clk); D["t0"] = clk.t; s.run()
    st = [c["t"] - D["t0"] for c in run.calls if "characterize.py" in c["cmd"]]
    assert len(st) == 2 and run.stops and s.done == [1]
    return st[1] - gap[1], run.stops[0][1]

def test_a_hold_only_for_our_robot_unseen_resumes_after_clear_own():
    dt, why = restart_after()
    assert why == "no information: our robot (blue 3) not seen in the vision" and 1.4 <= dt <= 2.0

def test_a_hold_with_another_cause_keeps_the_full_clear_time():
    dt, why = restart_after(camera_off=True)                            # the camera silent: no vision
    assert why.startswith("no information: no vision") and dt >= 4.9
    dt, why = restart_after(lambda dt: [("yellow", 1, 0.4, 0.0, 0.0)] if dt >= 1.0 else [])   # a robot near our half
    assert why == "no information: our robot (blue 3) not seen in the vision" and dt >= 4.9


# ---- 5. the end alert without the sender thread
def test_the_session_ended_alert_is_sent_directly_when_the_sender_thread_has_ended():
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OURS]}, budget=30, alert_thread=True)
    assert s.preflight() is None
    s.sender.halt.set(); s.sender.thread.join(2); s.sender_restarts = 1           # ended, and already restarted once
    rec = []; log = s.log
    s.log = lambda kind, **k: (rec.append((kind, k)), log(kind, **k))[1]
    s.preflight = lambda: None; s.run()
    assert any(m.startswith("session ended") for m in note.sent)
    d = [k for kind, k in rec if kind == "alert_sent_directly"]
    assert len(d) == 1 and d[0]["sent"] == "ok" and d[0]["message"].startswith("session ended")
