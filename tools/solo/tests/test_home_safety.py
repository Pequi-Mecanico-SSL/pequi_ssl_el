"""Tests of tools/solo/auto_session.py in a shared field: robots with our colour and id, cameras that never send, the
area between the home area and the walls, fetches and drives back with a robot inside the home area, a robot
entering the new half during the drive over, the header's env after a switch, stopping the strategy node, the
alert cap, mirrored plan names, and procs.run_child's time limit. Fakes for the field (no robot, ROS or network);
the stop tests start short local Python children.
    uv run --with pytest pytest -q tools/solo/tests/test_home_safety.py"""
import os, signal, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import auto_session as A  # noqa: E402
import procs as P         # noqa: E402
from test_home_area import HOME, SPIN, make, alerts, far_intruder  # noqa: E402


class Cams:
    """a fake vision source of several cameras: frames(t) -> {camera: [(colour, id, x, y)]} (a camera missing from
    the dict sends nothing); every camera sends one frame per 1/60 s"""
    def __init__(self, clk, frames):
        self.clk, self.frames, self.last, self.q = clk, frames, -1e9, []
    def read(self, timeout):
        if not self.q and self.clk.t - self.last >= 1 / 60 - 1e-9:
            self.last = self.clk.t; self.q = sorted(self.frames(self.clk.t).items())
        return self.q.pop(0) if self.q else None

def make_cams(steps, frames, cameras=(0,), robot=(-1.0, 0.0, 0.0), budget=300.0, **kw):
    """a session whose raw view comes from RawVision on Cams (fed separately from the page's robot)"""
    clk = A.SimClock()
    world = A.ScriptedWorld(clk, robot=robot, ball=(-1.2, 0.8), vision=A.RawVision(Cams(clk, frames), clk, thread=False))
    procs = A.SimProcs(); run = A.SimRunner(clk, world); note = A.SimNotifier()
    for i, st in enumerate(steps, 1): st.setdefault("n", i)
    kw.setdefault("home", HOME); kw.setdefault("our_id", 3); kw.setdefault("cameras", list(cameras))
    s = A.Session("blue", steps, world=world, runner=run, notifier=note, procs=procs, strategy=A.SimStrategy(procs), clock=clk,
                  sleep=clk.sleep, budget_s=budget, echo=False, **kw)
    return s, run, note, world, clk


# ---- 1. a robot with our colour and our id
VF = A.VISION_FRESH
OURS, TWIN = ("blue", 3, -1.0, 0.0), ("blue", 3, -0.6, 0.4)      # the twin: another team's robot, our colour and id

def test_a_second_robot_with_our_colour_and_id_stops_the_step_and_is_named_as_two_robots_seen_as_ours():
    t_in = [1e18]
    cases = {"same camera, ours first": lambda t: {0: [OURS] + ([TWIN] if t >= t_in[0] else []), 1: []},
             "same camera, ours last": lambda t: {0: ([TWIN] if t >= t_in[0] else []) + [OURS], 1: []},
             "ours on camera 0, the twin on camera 1": lambda t: {0: [OURS], 1: [TWIN] if t >= t_in[0] else []},
             "ours on camera 1, the twin on camera 0": lambda t: {0: [TWIN] if t >= t_in[0] else [], 1: [OURS]}}
    for name, frames in cases.items():
        s, run, note, world, clk = make_cams([SPIN(label="spin")], frames, cameras=(0, 1), budget=60, switch_after=0)
        t_in[0] = clk.t + 2.0; s.run()
        two = "two robots are seen as blue 3: at (-1.00, +0.00) and (-0.60, +0.40), the tools cannot tell which is ours"
        assert run.stops and run.stops[0][1] == "no information: " + two, name
        assert run.stops[0][0] - t_in[0] <= 0.1 + 1e-6 and not s.not_ours, name
        assert alerts(note)[0] == two + "; the session holds" and len(run.cmds("characterize.py")) == 1, name
        assert not any("another robot (blue 3)" in m for m in alerts(note)), name

def test_our_robot_seen_by_two_overlapping_cameras_is_one_robot():
    frames = lambda t: {0: [OURS], 1: [("blue", 3, -1.05, 0.03)]}
    s, run, note, world, clk = make_cams([SPIN(label="spin")], frames, cameras=(0, 1), budget=60)
    s.run()
    assert run.stops == [] and s.done == [1] and alerts(note) == []

def test_two_robots_with_our_colour_and_id_near_the_tools_robot_is_no_information():
    frames = lambda t: {0: [("blue", 3, -1.0, 0.2), ("blue", 3, -1.0, -0.2)]}
    s, run, note, world, clk = make_cams([SPIN(label="spin")], frames, budget=60)
    q = s.snap()
    assert q["raw_us"] is None and s.no_info(q) == "two robots are seen as blue 3: at (-1.00, -0.20) and (-1.00, +0.20), the tools cannot tell which is ours"
    s.run()                                                # no motion; the hold alerts at once
    assert run.calls == [] and alerts(note)[0].startswith("two robots are seen as blue 3")

def timed(s, clk):
    """[(t from now, log kind, message)] of the session's alert records, filled while it runs"""
    t0 = clk.t; rec = []; log = s.log
    def rlog(kind, **kw):
        if kind in ("alert", "alert_held"): rec.append((round(clk.t - t0, 1), kind, kw.get("message", "")))
        return log(kind, **kw)
    s.log = rlog
    return rec

def test_a_twin_touching_ours_in_one_frame_stops_the_step_and_holds_while_it_stays():
    t = {}
    def frames(now):
        dt = now - t["0"]
        if dt < 2: return {0: [OURS], 1: []}
        if dt < 4: return {0: [OURS], 1: [("blue", 3, -1.0, 0.8)]}            # approaching on camera 1
        if dt < 40: return {0: [OURS, ("blue", 3, -1.0, 0.22)], 1: []}        # touching ours, the same frame of camera 0
        return {0: [OURS], 1: []}
    s, run, note, world, clk = make_cams([SPIN(label="a"), SPIN(label="b")], frames, cameras=(0, 1), budget=120, switch_after=0)
    t["0"] = clk.t; starts = []; go = run.run
    run.run = lambda *a, **k: (starts.append(clk.t - t["0"]), go(*a, **k))[1]
    s.run()
    assert run.stops[0][0] - t["0"] <= 2.1 + 1e-6 and run.stops[0][1].startswith("no information: two robots are seen as blue 3")
    assert len(starts) == 3 and starts[1] >= 40 + 5 - 0.3            # nothing ran while the twin stayed
    assert s.done == [1, 2] and alerts(note)[0].startswith("two robots are seen as blue 3")
    rv = A.RawVision(Cams(clk, lambda now: {0: [OURS, ("blue", 3, -1.0, 0.22)]}), clk, thread=False, cameras=[0])
    clk.sleep(1 / 60); assert len(rv.snapshot()["vision_dets"]) == 2      # one frame: never merged

def test_one_robot_seen_by_two_cameras_with_different_frame_ages_at_speed_stays_one_robot():
    clk = A.SimClock(); t0 = clk.t; v = 1.5
    pos = lambda now: (-2.0 + v * (now - t0), 0.3)
    tick = [0]
    def frames(now):                                   # camera 0 every frame; camera 1 every 6th (frames up to 0.08 s old), 3 cm off
        tick[0] += 1; x, y = pos(now); f = {0: [("blue", 3, x, y)]}
        if tick[0] % 6 == 0: f[1] = [("blue", 3, x + 0.03, y - 0.02)] if now - t0 < 1.5 else []   # then it leaves camera 1's view
        return f
    rv = A.RawVision(Cams(clk, frames), clk, thread=False, cameras=[0, 1])
    counts = []
    while clk.t - t0 < 2.0:
        clk.sleep(1 / 60); counts.append(len([d for d in rv.snapshot()["vision_dets"] if d[:2] == ("blue", 3)]))
    assert set(counts) == {1}
    twins = A.RawVision(Cams(clk, lambda now: {0: [OURS], 1: [("blue", 3, -1.0, 0.2)]}), clk, thread=False, cameras=[0, 1])
    clk.sleep(1 / 60)
    assert len(twins.snapshot()["vision_dets"]) == 2                  # two cameras, 0.20 m apart, fresh: two robots

def test_two_robots_seen_as_ours_far_apart_hold_the_session_and_say_so():
    jump = [1e18]
    def frames(now):
        if now >= jump[0]: world.set(robot=(1.5, 0.5, 0.0))           # the connector follows the other one
        return {0: [OURS], 1: [("blue", 3, 1.5, 0.5)] if now >= jump[0] - 5 else []}
    s, run, note, world, clk = make_cams([SPIN(label="a")], frames, cameras=(0, 1), budget=60, switch_after=0)
    jump[0] = clk.t + 7.0; s.run()
    two = "two robots are seen as blue 3: at (-1.00, +0.00) and (+1.50, +0.50), the tools cannot tell which is ours"
    assert run.stops[0][1] == "no information: " + two and len(run.cmds("characterize.py")) == 1
    assert alerts(note)[0] == two + "; the session holds" and not any("another robot (blue 3)" in m for m in alerts(note))
    assert not s.not_ours and s.done == []


# ---- 3. a camera that never sends
def test_a_listed_camera_that_never_sent_is_no_information():
    clk = A.SimClock()
    v = A.RawVision(Cams(clk, lambda t: {0: [OURS]}), clk, thread=False, cameras=[0, 1])
    clk.sleep(0.02); q = v.snapshot()
    assert q["cam_age"] == {0: 0.0} and q["vision_age"] >= 1e9
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OURS]}, cameras=(0, 1), budget=60)
    s.run()
    assert run.calls == [] and "no vision from camera 1 since the start" in s.stop_why
    late = [1e18]                                       # camera 1 sends at first, then stops: the step stops
    s, run, note, world, clk = make_cams([SPIN(label="spin")], lambda t: {0: [OURS], **({1: []} if t < late[0] else {})},
                                         cameras=(0, 1), budget=60, switch_after=0)
    late[0] = clk.t + 2.0; s.run()
    assert run.stops and run.stops[0][1].startswith("no information: no vision for") and "(camera 1" in run.stops[0][1]

def test_a_home_session_without_the_camera_list_or_the_alert_command_does_not_start():
    s, run, note, *_ = make([SPIN()], cameras=None)
    s.run()
    assert run.calls == [] and "needs the field's cameras" in s.stop_why
    s, run, note, *_ = make([SPIN()]); s.notifier = A.Notifier(cmd="", notice_cmd="")
    s.run()
    assert run.calls == [] and "SOLO_NOTIFY_CMD" in s.stop_why

def test_a_home_session_starts_only_when_the_session_starting_alert_is_delivered(tmp_path):
    out = tmp_path / "sent"
    for rc in (3, 0):
        cmd = tmp_path / f"notify{rc}.sh"; cmd.write_text(f'#!/bin/sh\nprintf "%s\\n" "$1" >> {out}\nexit {rc}\n'); cmd.chmod(0o755)
        out.write_text("")
        s, run, note, *_ = make([SPIN()], head=dict(name="left_noball"), budget=60); s.notifier = A.Notifier(cmd=str(cmd), notice_cmd="")
        s.run()
        lines = out.read_text().splitlines()
        assert lines[0] == "session starting: left_noball, the -x half, budget 1 min"
        if rc: assert run.calls == [] and "did not deliver the start alert (failed: rc 3; rc 3)" in s.stop_why and lines[-1].startswith("session not started")
        else: assert len(run.calls) == 1 and s.done == [1]

def test_a_dry_run_names_the_start_alert_and_sends_nothing():
    import io
    s, run, note, *_ = make([SPIN()], head=dict(name="left_noball"), budget=600)
    buf = io.StringIO(); A.dry_run(s, out=buf, tables=False)
    assert "would send the alert 'session starting: left_noball, the -x half, budget 10 min'" in buf.getvalue() and note.sent == []

def test_a_camera_not_on_the_list_is_watched_from_its_first_frame():
    late = [1e18, 1e18]                                 # camera 2, not listed, sends from late[0] to late[1]
    s, run, note, world, clk = make_cams([SPIN(label="a"), SPIN(label="b")],
                                         lambda t: {0: [OURS], **({2: []} if late[0] <= t < late[1] else {})}, cameras=(0,), budget=60, switch_after=0)
    rec = []; log = s.log; s.log = lambda kind, **kw: (rec.append((kind, kw)), log(kind, **kw))[1]
    late[0] = clk.t + 2.0; late[1] = clk.t + 7.0; s.run()
    assert [kw["camera"] for k, kw in rec if k == "camera_unlisted"] == [2] and s.cameras == [0, 2]
    assert run.stops and run.stops[0][1].startswith("no information: no vision for") and "(camera 2" in run.stops[0][1]
    assert run.stops[0][0] - late[1] <= VF + 0.15

def test_frames_from_a_camera_not_on_the_list_at_the_start_refuse_the_start():
    s, run, note, world, clk = make_cams([SPIN()], lambda t: {0: [OURS], 3: []}, cameras=(0,), budget=60)
    s.run()
    assert run.calls == [] and "frames from camera(s) 3 that are not on the camera list (0)" in s.stop_why
    assert note.sent[0].startswith("session not started: frames from camera(s) 3")


# ---- 2. robots between the home area and the walls
def test_robots_between_the_home_area_and_the_walls_are_intruders():
    for spot in ((-2.3, 0.0), (-2.4, 0.1), (-1.0, 1.6), (-2.5, -1.75)):     # robot area, our goal, the +y strip, a corner
        s, run, note, *_ = make([SPIN(label="spin")], robot=(-0.6, -0.5, 0.0), timeline=[(2.0, dict(others=[("yellow1", *spot)]))],
                                budget=40, switch_after=0)
        s.run()
        assert run.stops and run.stops[0][1].startswith("another robot: yellow1"), spot
        assert alerts(note)[0].startswith("another robot (yellow 1) entered our half"), spot

def test_a_keeper_in_the_other_goal_leaves_the_other_half_occupied():
    s, run, note, *_ = make([SPIN()], timeline=[far_intruder(1.0), (1.0, dict(others=[("yellow2", -1.8, 1.0), ("yellow0", 2.25, 0.0)]))], budget=80)
    s.run()
    assert s.switches == [] and note.quiet == []
    s, run, note, *_ = make([SPIN()], home_auto=True, others=[("yellow2", -1.8, 1.0), ("yellow0", 2.3, 0.1)], budget=60)
    s.run()
    assert run.calls == [] and alerts(note)[0].startswith("both halves hold other robots (yellow 2, yellow 0)")

def test_home_auto_needs_the_half_empty_for_clear_s():
    tl = [(1.0, dict(others=[("yellow2", -1.8, 1.0), ("yellow1", 1.2, 0.5)])), (3.0, dict(others=[("yellow1", 1.2, 0.5)]))]   # ours held until 3 s
    s, run, note, world, strat, clk = make([SPIN()], home_auto=True, others=[("yellow1", 1.2, 0.5)], timeline=tl, budget=60)
    rec = []; log = s.log; s.log = lambda kind, **kw: (rec.append((kind, clk.t)), log(kind, **kw))[1]
    t0 = clk.t; s.run()
    chosen = [t for k, t in rec if k == "half_chosen"]
    assert A.half_name(s.home) == "-x" and len(run.cmds("characterize.py")) == 1 and s.switches == []
    assert chosen and chosen[0] - t0 >= 3.0 + 5.0 - 0.3                 # empty for clear_s after the robot left
    s, run, note, world, strat, clk = make([SPIN()], home_auto=True, others=[], timeline=[(0.5, dict(others=[("yellow1", 1.2, 0.5)]))], budget=60)
    s.run()                                                # the other half: never chosen on a single snapshot
    assert A.half_name(s.home) == "-x"


# ---- 4. a fetch with a robot inside our half
FETCH_STEPS = lambda: [dict(kind="place", x=-0.9, y=-0.3, env={"SOLO_BOX": "-2.43,-0.10,-1.68,1.68"}, label="p")]

def test_no_fetch_while_a_robot_is_inside_the_home_area():
    s, run, note, *_ = make(FETCH_STEPS(), ball=(0.5, 0.1), others=[("yellow2", -1.8, 1.0)], budget=120, park=(-0.85, 0.0), switch_after=0)
    s.run()
    assert run.cmds("place_ball.py") == []

def test_a_robot_entering_the_home_area_stops_the_fetch():
    def moved(): world.set(robot=(0.3, 0.05, 0.0))
    script = [dict(match="place_ball.py blue -0.85", line=dict(ok=True, status="done"), dur=6.0, call=moved)]
    s, run, note, world, strat, clk = make(FETCH_STEPS(), ball=(0.5, 0.1), script=script, park=(-0.85, 0.0), budget=300, switch_after=0,
                                           timeline=[(8.0, dict(others=[("yellow2", -1.8, 1.0)])), (40.0, dict(others=[]))])
    s.run()
    assert run.stops and run.stops[0][1] == "another robot: yellow2 at (-1.80, +1.00)"
    assert "place_ball.py blue -0.85" in run.calls[0]["cmd"]

def test_the_first_fetch_waits_clear_s():
    t_fetch = []
    script = [dict(match="place_ball.py blue -0.85", line=dict(ok=True, status="done"), world=dict(ball=(-0.85, 0.0)), call=lambda: t_fetch.append(clk.t))]
    s, run, note, world, strat, clk = make(FETCH_STEPS(), ball=(0.5, 0.1), park=(-0.85, 0.0), budget=60, script=script)
    t0 = clk.t; s.run()
    assert t_fetch and t_fetch[0] - t0 >= 5.0 - 1e-6 and s.fetch_inc is None          # clear for clear_s first


# ---- 5. driving back into the home area
def test_no_drive_back_while_a_robot_is_inside_the_home_area():
    s, run, note, *_ = make([SPIN()], robot=(0.6, 0.2, 0.0), others=[("yellow2", -1.8, 1.0)], budget=60, switch_after=0)
    s.run()
    assert run.cmds("goto.py") == [] and run.cmds("characterize.py") == []

def test_a_robot_entering_the_home_area_stops_the_drive_back():
    def enter(): world.set(others=[("yellow2", -1.8, 1.0)])
    script = [dict(match="goto.py blue -0.4 0.2", line="arrived: robot at (-0.40, +0.20) after 3.0 s", dur=5.0, call=enter)]
    s, run, note, world, strat, clk = make([SPIN()], robot=(0.6, 0.2, 0.0), script=script, budget=60, switch_after=0)
    s.run()
    assert run.stops and run.stops[0][1] == "another robot: yellow2 at (-1.80, +1.00)" and "goto.py" in run.calls[0]["cmd"]
    assert run.cmds("characterize.py") == []

def test_a_robot_entering_the_new_half_stops_the_drive_over_and_the_switch_back_waits_for_the_cooldown(tmp_path):
    def enter(): world.set(others=[("yellow2", -1.8, 1.0), ("yellow1", 1.8, -1.0)])
    script = [dict(match="goto.py blue 0.4 0.0", line="arrived: robot at (+0.40, +0.00) after 3.0 s", dur=5.0, call=enter)]
    s, run, note, world, strat, clk = make([SPIN(label="a"), SPIN(label="b")], script=script, vs=dict(OUT=str(tmp_path)), budget=300,
                                           timeline=[far_intruder(2.0), (60.0, dict(others=[("yellow1", 1.8, -1.0)]))])
    s.run()
    over = [c for c in run.calls if "goto.py blue 0.4 0.0" in c["cmd"]]
    assert over and run.stops[1][1] == "another robot: yellow1 at (+1.80, -1.00)"           # stop 0: the spin, by yellow2
    assert not any(c["env"].get("SOLO_BOX") == "0.6,1.4,-0.4,0.4" for c in run.calls if "characterize.py" in c["cmd"])  # nothing ran there
    assert len(s.switches) == 2 and s.switches[1] - s.switches[0] >= 60.0 and A.half_name(s.home) == "-x"
    assert s.done == [1, 2]


# ---- the header's env after a run-time switch
def test_a_run_time_switch_mirrors_the_header_env(tmp_path):
    steps = [dict(kind="test", test="spin", args=[3.0], label="a"), dict(kind="test", test="spin", args=[2.0], label="b")]
    head = {"kind": "session", "env": {"SOLO_BOX": "-1.4,-0.6,-0.4,0.4", "SOLO_BRAKE": "auto"}}
    s, run, note, world, strat, clk = make(steps, head=head, timeline=[far_intruder()], vs=dict(OUT=str(tmp_path)), budget=200)
    assert not [st for st in steps if st.get("_refused")]
    s.run()
    ch = [c for c in run.calls if "characterize.py" in c["cmd"]]
    assert len(s.switches) == 1 and len(ch) == 3 and ch[0]["env"]["SOLO_BOX"] == "-1.4,-0.6,-0.4,0.4"
    assert [c["env"]["SOLO_BOX"] for c in ch[1:]] == ["0.6,1.4,-0.4,0.4"] * 2 and s.done == [1, 2]
    assert s.env_for(dict(kind="place", x=0.85, y=0.0))["SOLO_BOX"] == "0.6,1.4,-0.4,0.4"


# ---- 6. stopping the strategy node
STUBBORN = "import signal, sys, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" \
           "print('ready', flush=True)\ntime.sleep(30)\n"

class OneChildProcs:
    """the process table holds only the given child (no other process is ever signalled)"""
    def __init__(self): self.child = None
    def table(self):
        c = self.child
        return [(c.pid, [sys.executable, "-B", self.path])] if c is not None and c.poll() is None else []

def test_strategy_off_kills_a_node_that_ignores_sigint_and_sigterm(tmp_path):
    node = tmp_path / "solo_node"; node.write_text(STUBBORN)
    procs = OneChildProcs(); procs.path = str(node)
    child = subprocess.Popen([sys.executable, "-B", str(node)], stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        procs.child = child
        st = A.Strategy(procs, run_dir=str(tmp_path), waits=(0.3, 0.3, 2.0)); st.child = child
        assert st.running()
        t0 = time.time(); off = st.off()
        assert off and child.poll() == -signal.SIGKILL and time.time() - t0 < 3.0
    finally:
        if child.poll() is None: child.kill()
        child.wait(2)

def test_strategy_on_refuses_while_the_old_node_lives(tmp_path):
    node = tmp_path / "solo_node"; node.write_text(STUBBORN)
    procs = OneChildProcs(); procs.path = str(node)
    child = subprocess.Popen([sys.executable, "-B", str(node)], stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        procs.child = child
        st = A.Strategy(procs, run_dir=str(tmp_path)); st.off = lambda: False       # the old node does not end
        assert st.on(dict(side="negative")) is False and st.why == "the old node is still running"
        assert child.poll() is None and not (tmp_path / "solo.pid").exists()
    finally:
        child.kill(); child.wait(2)


# ---- the step's process group after a stop
GROUP = "import os, signal, subprocess, sys, time\n" \
        "c = subprocess.Popen([sys.executable, '-B', '-c', 'import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); " \
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n" \
        "print('child', c.pid, flush=True)\ntime.sleep(30)\n"

def test_a_stopped_step_leaves_no_member_of_its_group(tmp_path):
    prog = tmp_path / "step.py"; prog.write_text(GROUP)
    r = A.Runner(echo=False); t0 = time.time()
    res = r.run([sys.executable, "-B", str(prog)], dict(os.environ), 20, watch=lambda: "stop" if time.time() - t0 > 1.0 else None)
    pid = int([l for l in res["tail"] if l.startswith("child")][0].split()[1])
    gone = False
    for _ in range(40):
        try:
            with open(f"/proc/{pid}/stat") as f: gone = f.read().rsplit(")", 1)[1].split()[0] == "Z"
        except OSError: gone = True
        if gone: break
        time.sleep(0.05)
    if not gone: os.kill(pid, signal.SIGKILL)
    assert res["stopped"] == "stop" and gone


# ---- the alert cap and the alert command
def test_alerts_are_capped_by_kind_and_robot_and_counted():
    s, run, note, world, strat, clk = make([])
    msg = lambda n: f"another robot ({s.who(n)}) entered our half at (-0.60, +0.30); our robot waits at (-1.00, +0.00)"
    sent = []
    for k in range(3): sent.append(s.alert(msg("yellow2"), kind="intruder", robot="yellow2")); clk.sleep(10)   # one robot crossing three times
    assert sent == [True, False, False] and len(note.sent) == 1
    assert s.alert(msg("blue5"), kind="intruder", robot="blue5") and len(note.sent) == 2          # another robot: at once
    assert s.alert("no vision for 1.0 s (camera 0): the session holds", kind="no_info") and len(note.sent) == 3
    clk.sleep(40); assert s.alert(msg("yellow2"), kind="intruder", robot="yellow2")              # the gap has passed
    assert note.sent[-1].endswith("(2 similar alerts held back since the last one)")
    t = clk.t; assert not s.alert(msg("yellow2"), kind="intruder", robot="yellow2", since=t)     # held while its cause is new ...
    clk.sleep(s.ALERT_DUE); assert s.alert(msg("yellow2"), kind="intruder", robot="yellow2", since=t)   # ... sent once it lasted
    assert not note.sent[-1].endswith("held back since the last one)")                          # the one held was this one
    assert not s.alert(msg("yellow2"), kind="intruder", robot="yellow2")                        # held, and never sent:
    s.finish([], [])                                                                            # the session's end counts it
    assert note.sent[-1].startswith("session ended") and note.sent[-1].endswith("alerts held back and never sent: 1 intruder (yellow 2)")

def test_an_intruder_crossing_in_and_out_sends_one_alert_a_minute():
    tl = []
    for k in range(8): tl += [(2.0 + 15 * k, dict(others=[("yellow2", -0.6, 0.3)])), (3.0 + 15 * k, dict(others=[]))]
    s, run, note, *_ = make([SPIN(label=f"s{k}") for k in range(12)], timeline=tl, budget=130, switch_after=0)
    s.run()
    a = [m for m in alerts(note) if "entered our half" in m]
    assert 2 <= len(a) <= 3 and len(s.intrusions) >= 6 and "held back" in a[-1]

def test_a_camera_silent_again_for_good_is_alerted_again_after_info_alert():
    t = {}
    def frames(now):                                    # camera 1 silent 12 s, back, then silent for good from 25 s
        dt = now - t["0"]
        return {0: [OURS]} if 2 <= dt < 14 or dt >= 25 else {0: [OURS], 1: []}
    s, run, note, world, clk = make_cams([SPIN(label=f"s{k}") for k in range(6)], frames, cameras=(0, 1), budget=600, switch_after=0)
    t["0"] = clk.t; rec = timed(s, clk); s.run()
    nv = [(dt, m) for dt, k, m in rec if k == "alert" and m.startswith("no vision for")]
    assert len(nv) >= 2 and nv[0][0] <= 2.6 + 10 + 0.5 and 25.5 + 10 - 0.5 <= nv[1][0] <= 25.5 + 10 + 0.5
    assert "(camera 1)" in nv[1][1] and "held back" not in nv[1][1]

def test_another_robot_entering_after_one_that_crossed_is_alerted_at_once():
    tl = [(2.0, dict(others=[("yellow2", -0.6, 0.3)])), (3.0, dict(others=[])), (30.0, dict(others=[("blue5", -1.8, -1.0)]))]
    s, run, note, world, strat, clk = make([SPIN(label=f"s{k}") for k in range(6)], timeline=tl, budget=300, switch_after=0)
    rec = timed(s, clk); s.run()
    b5 = [(dt, m) for dt, k, m in rec if k == "alert" and "(blue 5)" in m]
    assert b5[0][0] <= 30.0 + 0.3 and b5[0][1].startswith("another robot (blue 5) entered our half at (-1.80, -1.00)")

def test_the_same_robot_back_and_staying_is_alerted_once_it_has_stayed_a_few_seconds():
    tl = [(2.0, dict(others=[("yellow2", -0.6, 0.3)])), (3.0, dict(others=[])), (20.0, dict(others=[("yellow2", -1.8, 1.0)]))]
    s, run, note, world, strat, clk = make([SPIN(label=f"s{k}") for k in range(3)], timeline=tl, budget=100, switch_after=0)
    rec = timed(s, clk); s.run()
    y2 = [dt for dt, k, m in rec if k == "alert" and "(yellow 2) entered" in m]
    assert len(y2) == 2 and 20.0 + s.ALERT_DUE - 0.3 <= y2[1] <= 20.0 + s.ALERT_DUE + 0.5

def test_a_hold_under_information_that_comes_and_goes_alerts_and_reminds():
    t = {}
    def frames(now):                                    # from 2 s our robot unseen 0.8 s out of every 2 s
        dt = now - t["0"]
        return {0: []} if dt > 2 and (dt % 2.0) < 0.8 else {0: [OURS]}
    s, run, note, world, clk = make_cams([SPIN(label=f"s{k}") for k in range(3)], frames, budget=600, switch_after=0)
    t["0"] = clk.t; rec = timed(s, clk); s.run()
    a = [(dt, m) for dt, k, m in rec if k == "alert" and not m.startswith(("session starting", "session ended"))]
    assert run.stops and a and a[0][0] <= run.stops[0][0] - t["0"] + 10 + 0.5
    assert a[0][1].startswith("the session has held for 10 s without 5 s known and clear") and "our robot (blue 3) not seen" in a[0][1]
    assert len(a) == 3 and "reminder 2/3" in a[1][1] and 120 - 0.5 <= a[1][0] - a[0][0] <= 120 + 0.5

def test_a_robot_entering_after_a_no_information_stop_is_alerted_at_once():
    t = {}
    def frames(now):                                    # a 1 s gap stops the step; then a robot inside our half, staying
        dt = now - t["0"]
        return {} if 2 <= dt < 3 else {0: [OURS, ("yellow", 1, -1.8, 1.0)]} if dt >= 3 else {0: [OURS]}
    s, run, note, world, clk = make_cams([SPIN(label="a")], frames, budget=300, switch_after=0)
    t["0"] = clk.t; rec = timed(s, clk); s.run()
    assert run.stops[0][1].startswith("no information: no vision for")
    first = [(dt, m) for dt, k, m in rec if k == "alert" and "(yellow 1)" in m][0]
    assert first[0] <= 3.0 + 0.5 and first[1].startswith("another robot (yellow 1) entered our half at (-1.80, +1.00); our robot stopped")

def test_home_auto_without_a_chosen_half_alerts_with_the_reason():
    cases = {"two robots are seen as blue 3": lambda t: {0: [("blue", 3, -1.0, 0.2), ("blue", 3, -1.0, -0.2)]},
             "our robot (blue 3) not seen in the vision": lambda t: {0: []}}
    for why, frames in cases.items():
        s, run, note, world, clk = make_cams([SPIN(label="a")], frames, budget=300, home_auto=True)
        rec = timed(s, clk); s.run()
        a = [(dt, m) for dt, k, m in rec if k == "alert" and "no half chosen" in m]
        assert run.calls == [] and a and a[0][1].startswith(why), why
        assert a[0][0] <= s.robot_wait + s.clear_s + s.info_alert + 0.5 and len(a) == 3 and "reminder 2/3" in a[1][1], why

def test_a_failing_alert_command_is_tried_twice_and_logged(tmp_path):
    cmd = tmp_path / "notify.sh"; cnt = tmp_path / "n"
    cmd.write_text(f"#!/bin/sh\necho x >> {cnt}\nexit 3\n"); cmd.chmod(0o755)
    r = A.Notifier(cmd=str(cmd), notice_cmd="").send("m", "t")
    assert r == "failed: rc 3; rc 3" and cnt.read_text().count("x") == 2


# ---- mirroring: plan names, step tests without their start
def test_mirrored_plans_are_named_by_their_source_and_mirror_back_to_the_shipped_plan(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"; a.mkdir(); b.mkdir()
    for d in (a, b): (d / "p.jsonl").write_text('{"test": "spin", "args": [3.0], "at": [-1.0, 0.0, 0], "box": "-1.4,-0.6,-0.4,0.4"}\n')
    out = tmp_path / "m"
    ma, mb = (A.mirror_step(dict(kind="series", plan=str(d / "p.jsonl")), str(out)) for d in (a, b))
    assert ma["plan"] != mb["plan"] and os.path.basename(ma["plan"]).startswith("p_") and ma["plan"].endswith("_mx.jsonl")
    assert A.mirror_step(ma, str(out))["plan"] == str(a / "p.jsonl")

def test_a_step_test_without_its_start_is_refused_in_a_home_session():
    steps = [dict(kind="test", test="step", args=["fwd", 1.0, -1.0, 0.0, 0], env={"SOLO_BOX": HOME}),
             dict(kind="test", test="step", args=["fwd", 1.0, -1.6, 0.0, 0, 0.8], env={"SOLO_BOX": HOME})]
    s, *_ = make(steps)
    assert "explicit start" in steps[0].get("_refused", "") and not steps[1].get("_refused")


# ---- procs.run_child: the time limit stops with SIGINT first
def test_run_child_time_limit_sends_sigint_then_sigkill():
    class Clock:
        t = 0.0
        def __call__(self): return self.t
        def sleep(self, dt): self.t += dt
    class Child:
        def __init__(self, after): self.sigs, self.rc, self.t_sig, self.after = [], None, None, after
        def poll(self):
            if self.rc is None and self.t_sig is not None and self.after is not None and clk.t - self.t_sig >= self.after: self.rc = -2
            return self.rc
        def send_signal(self, s):
            self.sigs.append(s); self.t_sig = clk.t if self.t_sig is None else self.t_sig
            if s == signal.SIGKILL: self.rc = -9
        def communicate(self, timeout=None): return ("", None)
    for after, sigs in ((0.5, [signal.SIGINT]), (None, [signal.SIGINT, signal.SIGKILL])):
        clk = Clock(); c = Child(after)
        rc, out, how = P.run_child(["x"], lambda: None, popen=lambda *a, **k: c, clock=clk, sleep=clk.sleep, timeout=10.0)
        assert how == "timeout" and c.sigs == sigs
