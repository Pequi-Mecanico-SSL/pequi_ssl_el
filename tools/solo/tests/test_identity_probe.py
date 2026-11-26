"""Tests of tools/solo/auto_session.py: our robot's identity after its tag drops out (continuity, reappearance, lost),
the identity probe (a commanded turn on the spot watched in the raw vision; a child that did not turn gives no
verdict), the shorter clear time of a hold whose only cause was our robot not seen, the start check of the vision's
mirror setting, the end alert when the alert sender has stopped, and the lost alert and its recovery. Fakes for the field (no
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
FOUND = "our robot recognised by its turn at (-0.98, +0.43): the session resumes"
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

def test_a_second_copy_seen_before_ours_drops_out_makes_a_close_reappearance_lost():
    P = {}
    def others(dt):                                        # a twin 0.5 m from ours, seen while ours still is
        if dt >= 1.0:
            P.setdefault("p", (-2.0 + 0.53 + 0.5, -1.2 + 0.85)); return [("blue", 3, *P["p"], 0.0)]
        return []
    s, run, note, clk, starts, checks = clean_session(1.0, drop=(1.2, 8.0), others=others)
    assert s.done == [] and (s.ours_lost is not None or s.not_ours)
    assert s.us_last is None or math.hypot(s.us_last["x"] - P["p"][0], s.us_last["y"] - P["p"][1]) > 0.1   # never anchored on the twin

def reappear_after_dropout(moved):
    """a spin with our robot at (-1, 0); its tag is not detected from 2 s to 6 s, and at 4 s it is moved `moved` m
    along +x (the connector follows it)"""
    D = {}
    def frames(t):
        dt = t - D["t0"]
        if dt >= 4.0 and "moved" not in D: D["moved"] = True; world.set(robot=(-1.0 + moved, 0.0, world.robot[2]))
        if 2.0 <= dt < 6.0: return {0: []}                 # 4 s unseen while a spin runs
        return {0: [("blue", 3, *world.robot[:3])]}
    s, run, note, world, clk = make_cams([SPIN(label="a"), SPIN(label="b")], frames, budget=120, switch_after=0)
    D["t0"] = clk.t
    world.set(robot=(-1.0, 0.0, 0.0)); s.run()
    return s, run, note

def test_our_robot_reappearing_within_0_15_m_after_a_long_dropout_is_ours_without_a_probe():
    s, run, note = reappear_after_dropout(0.12)
    assert s.ours_lost is None and s.done == [1, 2] and not [m for m in alerts(note) if "which robot is ours" in m]
    assert s.probes_run == 0 and not PROBES(run)

def test_our_robot_reappearing_0_3_m_away_after_a_long_dropout_is_lost_then_proved_by_its_turn():
    s, run, note = reappear_after_dropout(0.3)
    lost = [m for m in alerts(note) if m.startswith("which robot is ours was lost")]
    assert lost and "0.30 m from where ours last stood" in lost[0] and "(at most 0.15 m, ours unseen for 4." in lost[0]
    assert s.ours_lost is None and s.done == [1, 2] and s.probes_run == 1 and len(PROBES(run)) == 2
    assert "our robot recognised by its turn at (-0.70, +0.00): the session resumes" in alerts(note)

def test_a_twin_first_seen_0_3_or_0_45_m_beyond_where_ours_vanished_is_lost_and_fails_the_probe():
    ux, uy = GOAL[0] - START[0], GOAL[1] - START[1]; L = math.hypot(ux, uy); ux, uy = ux / L, uy / L
    van = (START[0] + ux * 1.0, START[1] + uy * 1.0)        # where ours vanished: 1 s at 1 m/s
    for off in (0.3, 0.45):
        for t_twin in (3.0, 20.0):
            tw = (van[0] + ux * off, van[1] + uy * off)
            s, run, note, clk, t0 = twin_late(tw, t_twin=t_twin, budget=120)
            lost = [m for m in alerts(note) if m.startswith("which robot is ours was lost")]
            assert lost and s.ours_lost is not None and s.done == [] and not s.not_ours, (off, t_twin)
            assert len(run.cmds("goto.py")) == 1 and not [c for c in run.cmds("characterize.py") if " spin 3.0 " in c], (off, t_twin)
            assert s.probes_run >= 1 and [m for m in alerts(note) if NOT_OURS in m], (off, t_twin)
            assert not [m for m in alerts(note) if "recognised" in m] and not note.quiet, (off, t_twin)


# ---- 3. the identity probe
def test_lost_after_a_dropout_at_speed_beyond_0_15_m_is_resolved_by_the_probe_and_the_session_goes_on():
    s, run, note, clk, starts, checks = clean_session(1.0, (1.0, 1.0))
    pc = [c for c in run.calls if c["cmd"] in PROBES(run)]
    assert s.ours_lost is None and s.done == [1, 2] and s.probes_run == 1 and len(pc) == 2
    assert sorted(c["cmd"].split(" spin ")[1].split()[0] for c in pc) == ["+1.50", "-1.50"]
    assert all(c["watched"] and c["env"]["SOLO_BOX"] == "-1.23,-0.73,0.18,0.68" for c in pc)
    assert not note.quiet and alerts(note)[-1] == FOUND

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
    assert not note.quiet and alerts(note)[-1] == FOUND

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

def aborting_probes(n_abort, line="robot not in vision", rc=1):
    """SimRunner.run with the first n_abort probe children ending at once with `line` and exit code rc (a child that
    did not turn); -> (the patched run, the original)"""
    go = A.SimRunner.run; k = [0]
    def run_(self, argv, env, timeout, tee=None, watch=None):
        if "spin" in argv and argv[argv.index("spin") + 2] == "0.70" and k[0] < n_abort:
            k[0] += 1; self.calls.append(dict(cmd=" ".join(argv), env=env, timeout=timeout, watched=watch is not None))
            data, ln = A.result_of([line if isinstance(line, str) else A.json.dumps(line)])
            return dict(rc=rc, timed_out=False, line=ln, data=data, t0=self.clock(), t1=self.clock())
        return go(self, argv, env, timeout, tee=tee, watch=watch)
    return run_, go

def logged(kinds):
    """Session.log recording (time, kind, fields) of the given kinds; -> (the record, the original log)"""
    rec = []; log = A.Session.log
    def log_(self, kind, **kw):
        if kind in kinds: rec.append((self.clock(), kind, kw))
        return log(self, kind, **kw)
    return rec, log_, log

def test_a_probe_child_that_did_not_turn_gives_no_verdict_and_the_next_probe_runs_after_10_s():
    cases = {"not in vision": ("robot not in vision", 1, "(rc 1): robot not in vision"),
             "interrupted": (dict(test="spin", args=["+1.50", "0.70"], interrupted=True, signal=2), 130, "(rc 130): {")}
    for name, (line, rc, why) in cases.items():
        run_, go = aborting_probes(1, line, rc); rec, log_, log = logged(("probe_start", "probe_verdict", "probe_stopped"))
        A.SimRunner.run = run_; A.Session.log = log_
        try: s, run, note, clk, starts, checks = clean_session(1.0, (1.0, 1.0))
        finally: A.SimRunner.run = go; A.Session.log = log
        v = [(t, k) for t, kind, k in rec if kind == "probe_verdict"]
        assert v[0][1]["ok"] is None and v[0][1]["no_verdict"].startswith("the probe child did not complete its turn " + why), name
        assert v[1][1]["ok"] is True and not [r for r in rec if r[1] == "probe_stopped"], name
        st = [t for t, kind, k in rec if kind == "probe_start"]
        assert len(st) == 2 and st[1] - v[0][0] >= 10.0 - 1e-6, name
        assert s.ours_lost is None and s.done == [1, 2] and s.probes_run == 2 and s.probe_bad == [], name
        assert not [m for m in alerts(note) if NOT_OURS in m] and alerts(note)[-1] == FOUND, name

def test_probes_without_a_verdict_count_and_the_last_alert_says_how_many_and_why():
    run_, go = aborting_probes(99)
    A.SimRunner.run = run_
    try: s, run, note, clk, t0 = twin_late((-1.0, 0.5), probes=2, budget=200)
    finally: A.SimRunner.run = go
    ch = probe_children(run)
    assert s.probes_run == 2 and len(ch) == 2 and ch[1][0] - ch[0][0] < 60.0      # no 60 s ban at that place
    assert s.ours_lost is not None and s.done == [] and not [m for m in alerts(note) if NOT_OURS in m]
    out = [m for m in alerts(note) if m.startswith("no identity probe left")]
    assert out == ["no identity probe left (2 used, 2 without a verdict, the last: the probe child did not complete its turn "
                   "(rc 1): robot not in vision): the session holds: put the robot back at (-1.47, -0.35) or restart the session"]


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


# ---- 5. the start check of the vision's mirror setting
def test_the_start_check_refuses_a_heading_that_differs_from_the_page_and_names_vision_mirror():
    cases = {"sign flipped": (0.8, -0.8, "the same with its sign flipped"), "far off": (0.8, 2.4, ""),
             "agree": (0.8, 0.6, None), "no heading on the page": (0.8, None, None), "no heading in the raw view": (None, 0.8, None)}
    for name, (raw, page, flip) in cases.items():
        det = ("blue", 3, -1.0, 0.3) + (() if raw is None else (raw,))
        s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [det]}, robot=(-1.0, 0.3) + (() if page is None else (page,)), budget=30)
        rec = []; log = s.log
        s.log = lambda kind, **k: (rec.append((kind, k)), log(kind, **k))[1]
        why = s.preflight()
        if flip is None:
            assert why is None, name
            assert bool([k for kind, k in rec if kind == "heading_check" and "skipped" in k]) == (raw is None or page is None), name
        else:
            assert why and "--vision-mirror" in why and why.startswith("our robot's heading in the raw vision (+0.80 rad)"), (name, why)
            assert (flip in why) if flip else ("sign flipped" not in why), name
            assert not note.sent, name                                    # no start alert: the session does not start


# ---- 6. the end alert without the sender thread
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


# ---- 7. the lost alert and its recovery
LOST = A.Session.LOST

def lost_record(s, x=-1.0, y=0.0, why="w"):
    s.ours_lost = dict(since=s.clock(), x=x, y=y, t_ok=None, why=why); return s.ours_lost

def test_a_lost_message_sent_in_any_alert_makes_the_recovery_an_alert_and_none_saying_it_holds_follows_it():
    D = {"gotos": 0}                    # a goto at 1 m/s: ours unseen from 1 to 2 s; again for 11 s from 0.8 s into the second goto
    def dropped(dt):
        return 1.0 <= dt < 2.0 or ("d2" in D and D["d2"] <= dt < D["d2"] + 11.0)
    def frames(t):
        return {0: [] if dropped(t - D["t0"]) else [("blue", 3, *world.robot[:3])]}
    s, run, note, world, clk = make_cams([dict(kind="goto", x=GOAL[0], y=GOAL[1], label="g"), SPIN(label="after")], frames,
                                         robot=(*START, 0.0), budget=300, switch_after=0)
    run.SPEED, run.REACT, run.DECEL = 1.0, 0.15, 3.0
    snap0 = world.snapshot; F = {}
    def snap():                         # the connector keeps the last position while ours is unseen
        d = snap0()
        if dropped(clk.t - D["t0"]):
            F.setdefault("p", (d["robot"], clk.t)); d["robot"] = F["p"][0]; d["robot_age"] = round(clk.t - F["p"][1], 2)
        else: F.clear()
        return d
    world.snapshot = snap
    def on_run(argv):
        if "goto.py" in " ".join(argv):
            D["gotos"] += 1
            if D["gotos"] == 2: D["d2"] = clk.t - D["t0"] + 0.8
    timed_calls(run, clk, on_run); D["t0"] = clk.t
    s.run()
    sent = alerts(note); found = [j for j, m in enumerate(sent) if m.startswith("our robot recognised by its turn")]
    assert s.ours_lost is None and s.done == [1, 2] and len(found) == 2 and not note.quiet
    assert all(any(j >= i for j in found) for i, m in enumerate(sent) if LOST in m)     # every lost message has its recovery after it
    assert all(m.endswith(": the session resumes") for m in (sent[j] for j in found))
    # the lost message leaving in an alert of another kind, held by the rate floor: the recovery carries it
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OURS]}, budget=60)
    assert s.preflight() is None
    s.alert("no vision for 3.0 s: the session holds", kind="no_info", robot="a"); s.alert_poll()
    L = lost_record(s)
    assert s.alert(f"{LOST}: w: the session holds", kind="no_info", robot="b") and L["alerted"] and s.floor_buf
    s.ours_lost = None; s.found_text(L, "our robot seen again at its place (-1.00, +0.00)", s.snap()); s.alert_poll()
    assert alerts(note)[-1] == f"our robot seen again at its place (-1.00, +0.00): the session resumes | earlier: {LOST}: w: the session holds"
    assert not s.floor_buf and not note.quiet

def test_the_recovery_alert_says_the_session_still_holds_while_another_robot_is_in_our_half():
    from test_alerts_identity import swap_session
    s, run, note, starts = swap_session(back=60, others=[("yellow", 1, -1.5, 0.5)])
    sent = alerts(note)
    assert "our robot seen again at its place (-1.00, +0.00); the session still holds: another robot (yellow 1) entered our half at (-1.50, +0.50)" in sent
    assert s.ours_lost is None and s.done == [] and not any(m.endswith("the session resumes") for m in sent)

def test_the_lost_alert_promises_a_probe_only_when_one_can_run():
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OURS]}, budget=60)
    lost_record(s); q = s.snap()
    back = "put the robot back at (-1.00, +0.00) or restart the session"; probe = "; the session will try to recognise it by a short turn on the spot"
    assert s.lost_tail(q, probe=True) == back + probe
    assert s.lost_tail(q) == back                                         # a hold that does not run the probe
    lost_record(s, x=2.0); none = dict(q, mine=[])
    assert s.lost_tail(none, probe=True) == "put the robot back at (+2.00, +0.00) or restart the session"   # the anchor beyond the probe's area
    assert s.lost_tail(dict(q, mine=[(-1.0, 0.0, 0.0)]), probe=True).endswith(probe)                     # the candidate inside it
    lost_record(s); s.probes_run = s.probes
    assert s.lost_tail(q, probe=True) == back                             # no probe left

def test_a_new_loss_within_the_backoff_of_the_last_is_alerted_at_once_and_a_repeat_of_it_is_not():
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OURS]}, budget=60)
    assert s.preflight() is None
    L = lost_record(s, why="first")
    assert s.alert(f"{LOST}: first; the session holds", kind="ours_lost", since=L["since"])
    s.ours_lost = None; clk.t += 10.0
    L = lost_record(s, why="second")
    assert s.alert(f"{LOST}: second; the session holds", kind="ours_lost", since=L["since"])
    clk.t += 5.0
    assert not s.alert(f"{LOST}: second; the session holds", kind="ours_lost", since=L["since"])   # the same loss, a later hold
    s.alert_poll()
    assert [m for m in alerts(note) if LOST in m] == [f"{LOST}: first; the session holds", f"{LOST}: second; the session holds"]

def test_a_failed_probe_keeps_identity_lost_when_the_tracking_cleared_it_during_the_turns(monkeypatch):
    run_step, D = A.Session.run_step, {}
    def rs(self, st, *a, **k):          # the tracking takes the candidate as ours while the turns run
        if st.get("label") == "identity probe" and self.ours_lost is not None and not D:
            D["lost"] = dict(self.ours_lost); self.ours_lost = None; self.us_last = dict(x=-1.0, y=0.5, t=self.clock())
        return run_step(self, st, *a, **k)
    monkeypatch.setattr(A.Session, "run_step", rs)
    s, run, note, clk, t0 = twin_late((-1.0, 0.5), probes=1)
    assert D and s.ours_lost is not None and (s.ours_lost["x"], s.ours_lost["y"]) == (D["lost"]["x"], D["lost"]["y"])
    assert s.done == [] and any(NOT_OURS in m for m in alerts(note))

def test_the_not_ours_message_names_the_mirror_when_the_positions_agree_with_y_flipped():
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [("blue", 3, -1.0, 0.3)]}, budget=60)
    world.set(robot=(-1.0, 0.3, 0.0)); s.snap(); world.set(robot=(-1.0, -0.3, 0.0)); q = s.snap()
    assert s.mismatch(q) == ("the tools' robot at (-1.00, -0.30) is 0.60 m from blue 3 at (-1.00, +0.30) "
                             "(the positions agree with y flipped: check --vision-mirror)")
    world.set(robot=(-0.4, 0.3, 0.0)); q = s.snap()
    assert s.mismatch(q) == "the tools' robot at (-0.40, +0.30) is 0.60 m from blue 3 at (-1.00, +0.30)"
