"""Tests of tools/solo/auto_session.py without a robot, ROS or network: the step runners are fakes that return scripted
results, the world is a scripted sequence of robot and ball positions on a simulated clock.
    uv run --with pytest pytest -q tools/solo/tests/test_auto_session.py"""
import json, os, subprocess, sys, time
import pytest
HERE = os.path.dirname(os.path.abspath(__file__))
SOLO = os.path.dirname(HERE)
sys.path.insert(0, SOLO)
import auto_session as A  # noqa: E402


Clock, FakeProcs, FakeStrategy, FakeNotifier, FakeRunner = A.SimClock, A.SimProcs, A.SimStrategy, A.SimNotifier, A.SimRunner


def make(steps, script=(), robot=(0.0, 0.6, 0.0), ball=(0.3, -0.9), timeline=(), table=(), budget=1800.0, tmp=None, **kw):
    clk = Clock()
    tl = [(clk.t + dt, ch) for dt, ch in timeline]
    world = A.ScriptedWorld(clk, robot=robot, ball=ball, timeline=tl)
    procs = FakeProcs(table); strat = FakeStrategy(procs)
    run = FakeRunner(clk, world, script); note = FakeNotifier()
    ev = os.path.join(tmp, "events.jsonl") if tmp else None; lp = os.path.join(tmp, "auto", "s.jsonl") if tmp else None
    for i, st in enumerate(steps, 1): st.setdefault("n", i)
    s = A.Session("blue", steps, world=world, runner=run, notifier=note, procs=procs, strategy=strat, clock=clk, sleep=clk.sleep,
                  budget_s=budget, log_path=lp, events_path=ev, echo=False, **kw)
    return s, run, note, world, strat, clk

def ram_pair(x, y, sp, d, g):
    return [dict(kind="place", x=x, y=y, group=g, label=f"place {g}"),
            dict(kind="test", test="ram", args=[sp, d, 1.0, 0.25], needs_ball=True, group=g, label=f"ram {g}")]

def alerts_but_end(note):
    return [m for m in note.sent if not m.startswith("session ended")]


# ------------------------------------------------------------------ geometry
def test_geometry():
    assert A.obstacles(0, 1.65)[0][1] == "+y wall" and abs(A.obstacles(0, 1.65)[0][0] - 0.06) < 1e-9
    assert A.obstacles(-2.3, 0.52)[0][1] == "-x goal frame"
    assert A.obstacles(2.38, 0.0)[0][1] == "+x goal frame"          # inside the goal, against its back wall
    assert A.free_point(2.4, 1.7) == (1.75, 1.0) and A.free_point(0.3, -1.7) == (0.3, -1.0)
    assert A.ball_class((2.35, 0.1))[0] == "goal" and "+x goal" in A.ball_class((2.35, 0.1))[1]
    assert A.ball_class((-2.24, 0.45))[0] == "frame"
    assert A.ball_class((2.4, -1.6))[0] == "corner" and "(+x, -y) corner" in A.ball_class((2.4, -1.6))[1]
    assert A.ball_class((0.5, 1.77))[0] == "wall" and A.ball_class((0.5, 1.6))[0] == "outside"
    assert A.ball_class((1.9, 0.2))[0] == "defense" and A.ball_class((0.0, 0.0))[0] == "field" and A.ball_class(None)[0] == "lost"
    assert A.widen("-2.2,2.2,-1.4,1.4") == "-2.35,2.35,-1.55,1.55" and A.widen("-2.4,2.4,-1.6,1.6") == "-2.43,2.43,-1.68,1.68"


def test_false_match_process_check():
    """only a python process whose script is a commanding program counts; command lines that mention one do not"""
    false = [(10, ["bash", "-c", "tail -f logs/series.py.log"]), (11, ["vim", "tools/solo/characterize.py"]),
             (12, ["python3", "tools/solo/hq.py", "blue"]), (13, ["grep", "solo_node"]),
             (14, ["tmux", "new", "-d", "python3 tools/solo/place_ball.py blue 0 0"]),
             (15, ["python3", "-c", "import series.py"]), (16, ["python3", "/opt/ros/humble/bin/ros2", "run", "solo_strategy", "solo_node"]),
             (17, ["less", "/tmp/place_ball.py.out"]), (1, ["python3", "tools/solo/auto_session.py", "blue", "series.py"])]
    assert A.busy(FakeProcs(false)) == []
    real = false + [(20, ["python3", "-u", "/x/tools/solo/series.py", "blue", "p.jsonl"]), (21, ["/usr/bin/python3", "/x/lib/solo_strategy/solo_node", "--ros-args"])]
    assert A.busy(FakeProcs(real)) == ["series.py", "solo_node"]
    assert A.is_launcher(false[6][1]) and not A.is_instance(false[6][1], "solo_node")


def test_false_match_does_not_block_a_session():
    steps = [dict(kind="goto", x=1.0, y=0.5), dict(kind="test", test="spin", args=[3.0])]
    table = [(10, ["bash", "-c", "watch -n1 'ps aux | grep characterize.py'"]), (11, ["python3", "tools/solo/hq.py", "blue"])]
    s, run, note, *_ = make(steps, table=table)
    s.run()
    assert len(run.cmds("goto.py")) == 1 and len(run.cmds("characterize.py")) == 1 and s.done == [1, 2] and alerts_but_end(note) == []


def test_real_commander_blocks_motion_until_it_ends():
    steps = [dict(kind="test", test="spin", args=[3.0])]
    table = [(30, ["python3", "/x/tools/solo/characterize.py", "blue", "step", "fwd", "1.0"])]
    s, run, note, world, strat, clk = make(steps, table=table)
    t0 = clk.t
    orig = s.busy_now
    def busy_now(allow=()):
        if clk.t - t0 > 150: s.procs.rows = []                   # the other test ends after 150 s
        return orig(allow)
    s.busy_now = busy_now
    s.run()
    assert len(alerts_but_end(note)) == 1 and "another program commands the robot (characterize.py)" in note.sent[0]
    assert s.done == [1] and run.calls and run.calls[0]["cmd"].count("characterize.py") == 1


# ------------------------------------------------------------------ the plan's sessions
@pytest.mark.parametrize("name,n_place,n_strat", [("short_10", 7, 2), ("short_20", 18, 8)])
def test_plan_sessions_all_well(tmp_path, name, n_place, n_strat):
    head, steps, vs = A.load_session(os.path.join(SOLO, "sessions", name + ".jsonl"), "blue", {"P": "/plan", "OUT": "/out"})
    clk = Clock(); world = A.ScriptedWorld(clk, robot=(0.0, 0.6, 0.0), ball=(0.3, -0.9))
    procs = FakeProcs(); strat = FakeStrategy(procs); run = FakeRunner(clk, world); note = FakeNotifier()
    s = A.Session("blue", steps, head, vs, world=world, runner=run, notifier=note, procs=procs, strategy=strat, clock=clk, sleep=clk.sleep,
                  budget_s=3600, events_path=str(tmp_path / "ev.jsonl"), log_path=str(tmp_path / "auto" / "s.jsonl"), echo=False)
    s.run()
    assert alerts_but_end(note) == [] and note.sent[-1].startswith("session ended: all steps done")
    assert len(run.cmds("place_ball.py")) == n_place and strat.ons == n_strat
    a = run.cmds("series.py")[0]
    assert "/plan/steps_ramped.jsonl" in a and "--max 8" in a and "--out /out/logs" in a and run.calls[[c["cmd"] for c in run.calls].index(a)]["env"]["SOLO_BOX"] == ""
    ram = [c for c in run.calls if " ram " in c["cmd"]][0]
    assert ram["env"]["SOLO_RAMP"] == "1.7" and ram["env"]["SOLO_BOX"] == "-2.05,2.05,0.25,1.05" and ram["cmd"].endswith("ram 1.2 0 1.0 0.25 --out /out/logs")
    ev = [json.loads(l) for l in open(tmp_path / "ev.jsonl")]
    assert ev[0]["state"].startswith("short session start") and ev[-1]["state"].startswith("short session end")
    assert ev[0]["data"] == "use" and ev[-1]["data"] == "skip" and all(set(e) == {"from", "to", "state", "data", "source"} for e in ev)
    log = [json.loads(l) for l in open(tmp_path / "auto" / "s.jsonl")]
    st = [r for r in log if r["kind"] == "step" and r.get("cmd")]
    assert all({"cmd", "start", "end", "rc", "result"} <= set(r) for r in st)


# ------------------------------------------------------------------ the ball
def test_ball_in_corner_alert_once_ballfree_go_on_then_ball_steps_resume():
    steps = ram_pair(-0.85, 0.65, 1.5, 0, "a") + ram_pair(0.85, 0.65, 1.5, 180, "b") + \
            [dict(kind="series", plan="C.jsonl", label="spins"), dict(kind="test", test="spin", args=[6.0], label="spin")]
    corner = (2.52, 1.73)                                       # past the placer's margin: it cannot push it
    script = [dict(match="place_ball.py", line=dict(ok=True, status="done", ball=[-0.85, 0.65]), world=dict(ball=(-0.85, 0.65))),
              dict(match="characterize.py blue ram", line=dict(test="ram", ball_end=list(corner)), world=dict(ball=corner))]
    s, run, note, world, strat, clk = make(steps, script, timeline=[(150, dict(ball=(0.2, -0.3)))])
    s.run()
    al = alerts_but_end(note)
    assert len(al) == 1 and al[0].startswith("ball in the (+x, +y) corner at (+2.52, +1.73): put it anywhere")
    order = [" ".join([os.path.basename(c["cmd"].split()[1])] + c["cmd"].split()[2:]) for c in run.calls]
    i_series = next(i for i, c in enumerate(order) if c.startswith("series.py")); i_spin = next(i for i, c in enumerate(order) if " spin " in c)
    i_place_b = max(i for i, c in enumerate(order) if c.startswith("place_ball.py blue 0.85"))
    assert i_series < i_place_b and i_spin < i_place_b             # ball-free steps ran while the ball was in the corner
    assert s.done == [1, 2, 5, 6, 3, 4] and s.ball_incident is None


def test_ball_in_goal_no_ballfree_left_waits_and_realerts_three_times():
    steps = ram_pair(-0.85, 0.65, 1.2, 0, "a") + ram_pair(0.85, 0.65, 1.2, 180, "b")
    script = [dict(match="place_ball.py", line=dict(ok=True, status="done", ball=[-0.85, 0.65])),
              dict(match="characterize.py", line=dict(test="ram"), world=dict(ball=(2.33, 0.05)))]
    s, run, note, *_ = make(steps, script, budget=900, realert=180)
    s.run()
    al = alerts_but_end(note)
    assert len(al) == 3 and al[0].startswith("ball in the +x goal at (+2.33, +0.05): put it anywhere inside the field")
    assert "reminder 3/3" in al[2] and note.sent[-1].startswith("session ended: budget used")
    assert len(run.cmds("place_ball.py")) == 1                     # no placement tried from inside the goal


def test_ball_rescue_by_the_placer_no_alert():
    steps = ram_pair(-0.85, 0.65, 1.5, 0, "a") + ram_pair(0.85, 0.65, 1.5, 180, "b")
    script = [dict(match="place_ball.py", line=dict(ok=True, status="done", ball=[-0.85, 0.65])),
              dict(match="characterize.py", line=dict(test="ram"), world=dict(ball=(2.3, 1.55)))]   # corner, within the margin
    s, run, note, *_ = make(steps, script)
    s.run()
    pl = run.cmds("place_ball.py")
    assert alerts_but_end(note) == [] and len(pl) == 3 and pl[1].endswith("blue 1.95 1.2")
    assert run.calls[[c["cmd"] for c in run.calls].index(pl[1])]["env"]["SOLO_BOX"] == ""
    assert s.done == [1, 2, 3, 4]


def test_placer_unreachable_ball_past_line_alerts_and_resumes():
    steps = ram_pair(0.85, 0.65, 1.2, 180, "a") + [dict(kind="test", test="spin", args=[6.0])]
    script = [dict(match="place_ball.py", line=dict(ok=False, status="unreachable: no line-up point behind the ball inside the allowed area (margin 0.25)", ball=[0.3, 1.62]))]
    s, run, note, *_ = make(steps, script, ball=(0.3, 1.62), timeline=[(120, dict(ball=(0.0, 0.9)))])
    s.run()
    al = alerts_but_end(note)
    assert len(al) == 1 and "the robot cannot push it from there" in al[0] and "(+0.30, +1.62)" in al[0]
    assert s.done == [3, 1, 2]


def test_ball_lost_alert():
    steps = ram_pair(0.85, 0.65, 1.2, 180, "a")
    s, run, note, *_ = make(steps, ball=None, timeline=[(30, dict(ball=(0.1, 0.1)))])
    s.run()
    al = alerts_but_end(note)
    assert len(al) == 1 and al[0].startswith("ball not seen since") and s.done == [1, 2]


# ------------------------------------------------------------------ placement failures
def test_placement_timeout_retried_wider_slower_then_on_fail_skips_group():
    steps = ram_pair(-0.85, 0.65, 1.2, 0, "a") + [dict(kind="test", test="spin", args=[6.0])]
    fail = dict(ok=False, status="failed: timeout (60 s)", ball=[0.0, 0.3])
    script = [dict(match="place_ball.py", line=fail), dict(match="place_ball.py", line=fail)]
    s, run, note, *_ = make(steps, script)
    s.head = dict(env={"SOLO_BOX": "-2.2,2.2,-1.4,1.4"})
    s.run()
    pl = [c for c in run.calls if "place_ball.py" in c["cmd"]]
    assert len(pl) == 2 and pl[1]["cmd"].endswith("--carry 0.25") and pl[1]["env"]["SOLO_BOX"] == "-2.35,2.35,-1.55,1.55"
    assert pl[0]["env"]["SOLO_BOX"] == "-2.2,2.2,-1.4,1.4"
    assert run.cmds(" ram ") == [] and s.done == [3] and s.skipped == [1, 2] and alerts_but_end(note) == []


def test_placement_left_area_retry_succeeds():
    steps = ram_pair(-0.85, 0.65, 1.2, 0, "a")
    script = [dict(match="place_ball.py", line=dict(ok=False, status="failed: robot left the allowed area at (-2.21, 0.70)"), world=dict(robot=(-2.0, 0.7, 0.0)))]
    s, run, note, *_ = make(steps, script)
    s.run()
    assert len(run.cmds("place_ball.py")) == 2 and s.done == [1, 2]


# ------------------------------------------------------------------ the robot
def test_robot_stuck_one_freeing_attempt_then_alert_and_resume():
    steps = ram_pair(0.85, 1.2, 1.2, 180, "a") + [dict(kind="test", test="spin", args=[6.0])]
    stuck = (1.0, 1.65, 0.0)
    script = [dict(match="place_ball.py", line=dict(ok=False, status="failed: robot stuck at (1.00, 1.65)"), world=dict(robot=stuck)),
              dict(match="goto.py", line="vision lost the robot: robot at (+1.00, +1.65) after 0.6 s", world=dict(robot=stuck))]
    s, run, note, *_ = make(steps, script, timeline=[(150, dict(robot=(0.0, 0.0, 0.0)))])
    s.run()
    al = alerts_but_end(note)
    gotos = run.cmds("goto.py")
    assert len(gotos) == 1 and gotos[0].endswith("goto.py blue 1.0 1.0 --v 0.25")       # straight away from the +y wall
    assert len(al) == 1 and al[0] == "robot stuck against the +y wall at (+1.00, +1.65): free it and put it inside the field"
    assert 3 in s.done and not s.motion_halted


def test_robot_near_goal_frame_after_abort_is_freed():
    steps = [dict(kind="test", test="step", args=["fwd", 1.5], label="step"), dict(kind="test", test="spin", args=[6.0])]
    script = [dict(match="characterize.py blue step", line="ABORT: robot at the edge of the allowed area (last at (-2.3, 0.52, 3.1))", rc=1,
                   world=dict(robot=(-2.3, 0.52, 3.1)))]
    s, run, note, *_ = make(steps, script)
    s.run()
    assert run.cmds("goto.py")[0].endswith("blue -1.75 0.52 --v 0.25") and alerts_but_end(note) == [] and s.done == [2] and s.skipped == [1]


def test_abort_mid_field_follows_on_fail_without_freeing():
    steps = [dict(kind="test", test="step", args=["fwd", 1.5], on_fail="alert", label="fast step"), dict(kind="wait", seconds=1)]
    script = [dict(match="characterize.py", line="ABORT: vision lost the robot (last at (0.1, 0.5, 0))", rc=1)]
    s, run, note, *_ = make(steps, script)
    s.run()
    al = alerts_but_end(note)
    assert run.cmds("goto.py") == [] and len(al) == 1 and al[0].startswith("step 1 (fast step) failed: ABORT: vision lost")


def test_on_fail_retry_and_stop():
    steps = [dict(kind="test", test="spin", args=[6.0], on_fail="retry"), dict(kind="test", test="spin", args=[10.0], on_fail="stop"),
             dict(kind="test", test="spin", args=[14.0])]
    ab = dict(match="characterize.py", line="ABORT: x", rc=1)
    s, run, note, *_ = make(steps, [ab, dict(match="characterize.py", line=dict(test="spin")), ab])
    s.run()
    assert len(run.cmds("characterize.py")) == 3 and s.done == [1] and "on_fail stop" in s.stop_why and " 14.0 " not in " ".join(run.cmds())


def test_robot_not_seen_alert_then_resume():
    steps = [dict(kind="test", test="spin", args=[6.0])]
    s, run, note, *_ = make(steps, robot=(0.4, 0.4, 0.0), timeline=[(0.0, dict(robot=None)), (120, dict(robot=(0.4, 0.4, 0.0)))])
    s.world.robot_t -= 30
    s.run()
    al = alerts_but_end(note)
    assert len(al) == 1 and al[0].startswith("robot not seen since") and "last at (+0.40, +0.40)" in al[0] and s.done == [1]


def test_step_time_limit_from_session():
    steps = [dict(kind="series", plan="A.jsonl", timeout=300)]
    s, run, note, *_ = make(steps, [dict(match="series.py", line="    REMOVE THE BALL from the test area ...", rc=-2, timed_out=True, dur=300)])
    s.run()
    assert run.calls[0]["timeout"] == 300 and s.skipped == [1]


# ------------------------------------------------------------------ budget, Ctrl-C, strategy
def test_budget_finishes_current_step_strategy_off_end_marker(tmp_path):
    steps = [dict(kind="place", x=1.9, y=0.0, group="p"), dict(kind="strategy", on=True, seconds=10, needs_ball=True, group="p")] + \
            [dict(kind="series", plan="A.jsonl", label=f"s{i}") for i in range(5)] + \
            [dict(kind="cmd", argv=["robot_stack.sh", "save"], final=True), dict(kind="marker", text="end", data="skip", final=True)]
    script = [dict(match="series.py", line="series done", dur=40)] * 5
    s, run, note, world, strat, clk = make(steps, script, budget=90, tmp=str(tmp_path))
    s.run()
    assert strat.ons == 1 and strat.offs >= 1 and not strat.running()
    assert len(run.cmds("series.py")) == 2 and run.cmds()[-1] == "robot_stack.sh save"
    assert note.sent[-1].startswith("session ended: budget used") and "3 not run" in note.sent[-1]
    assert json.loads(open(tmp_path / "events.jsonl").read().splitlines()[-1])["state"] == "end"


def test_budget_ends_while_strategy_was_left_on():
    steps = [dict(kind="wait", seconds=100)]
    s, run, note, world, strat, clk = make(steps, budget=50)
    strat.on()
    s.run()
    assert not strat.running() and note.sent[-1].startswith("session ended: budget used")


def test_ctrl_c_finishes_current_step():
    steps = [dict(kind="test", test="spin", args=[6.0]), dict(kind="test", test="spin", args=[10.0])]
    holder = {}
    script = [dict(match="characterize.py", line=dict(test="spin"), call=lambda: holder["s"].request_stop())]
    s, run, note, *_ = make(steps, script)
    holder["s"] = s
    s.run()
    assert len(run.cmds("characterize.py")) == 1 and s.done == [1] and s.stop_why == "Ctrl-C"


def test_strategy_step_waits_and_stops_and_needs_a_start_line():
    steps = [dict(kind="strategy", on=True, seconds=10, label="push")]
    s, run, note, world, strat, clk = make(steps)
    t0 = clk.t; s.run()
    assert strat.ons == 1 and not strat.running() and clk.t - t0 >= 10 and s.done == [1]
    s, run, note, world, strat, clk = make([dict(kind="strategy", on=True, seconds=10, on_fail="alert")])
    strat.can = False; s.run()
    assert strat.ons == 0 and "failed" in alerts_but_end(note)[0]


# ------------------------------------------------------------------ the real runner and the dry run
def test_runner_time_limit_stops_the_process_group():
    r = A.Runner(echo=False)
    t0 = time.time()
    out = r.run([sys.executable, "-c", "import time, sys; print('{\"started\": 1}', flush=True); time.sleep(30)"], dict(os.environ), timeout=1.0)
    assert out["timed_out"] and time.time() - t0 < 5 and out["data"] == {"started": 1}
    out = r.run([sys.executable, "-c", "print('x'); print('ABORT: robot at the edge'); raise SystemExit(1)"], dict(os.environ), timeout=5)
    assert out["rc"] == 1 and out["line"] == "ABORT: robot at the edge" and not out["timed_out"]


def test_dry_run_prints_commands_and_decisions(capsys):
    head, steps, vs = A.load_session(os.path.join(SOLO, "sessions", "short_10.jsonl"), "blue", {"P": "/plan", "OUT": "/out"})
    s = A.Session("blue", steps, head, vs, procs=FakeProcs(), strategy=FakeStrategy(FakeProcs()), budget_s=600, echo=False)
    import io
    buf = io.StringIO(); A.dry_run(s, out=buf); out = buf.getvalue()
    assert "place_ball.py blue -0.85 0.65" in out and "characterize.py blue ram 1.5 180 1.0 0.25" in out
    assert "strategy on 10 s, then off" in out and "situation -> what the supervisor does" in out


# ------------------------------------------------------------------ the ball inside a ball-free test's box
def test_ball_in_test_box_is_parked_first():
    steps = [dict(kind="series", plan="A.jsonl", keep_ball_out="-2.0,2.0,0.25,1.0", park=[1.2, -0.8])]
    s, run, note, world, *_ = make(steps, ball=(0.0, 0.6))
    s.run()
    assert [c["cmd"].split()[1].endswith(n) for c, n in zip(run.calls, ("place_ball.py", "series.py"))] == [True, True]
    assert run.cmds("place_ball.py")[0].endswith("blue 1.2 -0.8") and s.done == [1]


def test_ball_in_test_box_without_park_follows_on_fail():
    steps = [dict(kind="series", plan="A.jsonl", keep_ball_out="-2.0,2.0,0.25,1.0", on_fail="alert", label="steps"), dict(kind="wait", seconds=1)]
    s, run, note, *_ = make(steps, ball=(0.0, 0.6))
    s.run()
    al = alerts_but_end(note)
    assert run.cmds("series.py") == [] and len(al) == 1 and "the ball lies inside the test box" in al[0] and s.done == [2]


# ------------------------------------------------------------------ the strategy switch (no process is signalled)
def test_strategy_switch_like_hq(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir(); py = tmp_path / "bin" / "python3"; py.write_text("#!/bin/sh\n"); py.chmod(0o755)
    (tmp_path / "lib").mkdir(); (tmp_path / "lib" / "solo_node").write_text("x")
    node = [str(py), str(tmp_path / "lib" / "solo_node"), "--ros-args", "-p", "team:=blue"]
    launcher = ["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "run", "solo_strategy", "solo_node", "--ros-args"]
    procs = FakeProcs([(501, launcher), (502, node), (503, ["vim", "solo_node.py"])])
    killed, started = [], []
    def kill(pid, sig):
        killed.append((pid, sig)); procs.rows = [r for r in procs.rows if r[0] != pid]
    class P:
        pid = 777
        def poll(self): return None
    monkeypatch.setattr(A.os, "kill", kill)
    monkeypatch.setattr(A.subprocess, "Popen", lambda argv, **kw: started.append(argv) or P())
    st = A.Strategy(procs, run_dir=str(tmp_path))
    assert st.running() and st.off() and not st.running()
    assert sorted(p for p, _ in killed) == [501, 502] and json.load(open(tmp_path / "solo.cmd")) == node
    assert st.on() and started == [node] and open(tmp_path / "solo.pid").read() == "777"


def test_supervisor_fault_still_ends_cleanly():
    steps = [dict(kind="strategy", on=True, seconds=5), dict(kind="test", test="spin", args=[6.0]), dict(kind="marker", text="end", final=True)]
    s, run, note, world, strat, clk = make(steps)
    def boom(*a, **k): raise RuntimeError("bad step")
    s.runner.run = boom
    s.run()
    assert not strat.running() and "supervisor error" in s.stop_why and note.sent[-1].startswith("session ended: supervisor error")


def test_robot_never_seen_message():
    s, run, note, *_ = make([dict(kind="goto", x=0, y=0)], robot=None, budget=60)
    s.run()
    assert alerts_but_end(note)[0].startswith("robot not seen since the session started")


def test_page_world_reads_state(monkeypatch):
    """the field page's /state: robot pose and its age, ball (None when older than 1 s there), stillness (no network)"""
    seq = [dict(us=[1.0, 0.5, 0.1], us_age=0.05, ball=[0.3, -0.2], strategy="off", target=None, place=dict(active=True))] * 8 + [dict(us=[1.0, 0.5, 0.1], us_age=3.0, ball=None, strategy="off")]
    class R:
        def __init__(self, d): self.d = d
        def read(self): return json.dumps(self.d).encode()
    calls = []
    def urlopen(url, timeout=1.0):
        calls.append(url); return R(seq[min(len(calls) - 1, len(seq) - 1)])
    monkeypatch.setattr(A.urllib.request, "urlopen", urlopen)
    w = A.PageWorld("http://127.0.0.1:8700/", period=0.25)
    try:
        time.sleep(1.9)
        s = w.snapshot()
        assert calls[0] == "http://127.0.0.1:8700/state" and s["robot"] == (1.0, 0.5, 0.1) and s["ball"] == (0.3, -0.2)
        assert s["ball_still"] and s["robot_age"] < 0.5 and s["page_age"] < 0.5 and s["page_driving"]
        time.sleep(0.8)
        s = w.snapshot()
        assert s["robot_age"] > 2.5 and s["ball"] == (0.3, -0.2) and s["ball_age"] > 0.2
    finally:
        w.stop()                                     # before the patched urlopen is undone: no request to a real port
    assert not w.thread.is_alive()
    n = len(calls); time.sleep(0.4); assert len(calls) == n


def test_strategy_on_refuses_a_saved_line_that_is_not_the_nodes(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(A.subprocess, "Popen", lambda argv, **kw: started.append(argv))
    st = A.Strategy(FakeProcs(), run_dir=str(tmp_path))
    for argv in (["python3", "-c<code>#/solo_node"], ["/usr/bin/python3", "-c", "x"], ["sh", "-c", "x"], "python3 solo_node",
                 [sys.executable, os.path.abspath(__file__), "--ros-args"]):
        (tmp_path / "solo.cmd").write_text(json.dumps(argv))
        assert st.on() is False and st.why, argv
    assert started == []
    steps = [dict(kind="strategy", on=True, seconds=5, on_fail="alert")]
    s, run, note, world, strat, clk = make(steps)
    s.strat = st; s.run()
    assert started == [] and "failed" in alerts_but_end(note)[0]


# ------------------------------------------------------------------ the commanding programs (procs.py)
def test_goto_nudge_frame_check_block_motion_but_not_our_own_step():
    for prog in ("goto.py", "nudge.py", "frame_check.py"):
        table = [(40, ["python3", f"/x/tools/solo/{prog}", "blue", "0.1", "0.1"])]
        assert A.busy(FakeProcs(table)) == [prog]
    s, run, note, world, strat, clk = make([dict(kind="wait", seconds=1)], table=[(41, ["python3", "/x/goto.py", "blue", "1", "1"])])
    assert s.busy_now() == ["goto.py"]                                   # someone else's goto.py
    class Live:
        pid = 41
        def poll(self): return None
    s.runner = A.Runner(echo=False); s.runner.p = Live()
    assert s.busy_now() == []                                            # the step this session runs itself
    s.procs.rows.append((42, ["python3", "/y/goto.py", "blue", "0", "0"]))
    assert s.busy_now() == ["goto.py"]


def test_parking_the_ball_waits_for_the_robot_and_for_other_programs():
    steps = [dict(kind="series", plan="A.jsonl", keep_ball_out="-2.0,2.0,0.25,1.0", park=[1.2, -0.8])]
    s, run, note, world, strat, clk = make(steps, ball=(0.0, 0.6), table=[(30, ["python3", "/x/characterize.py", "blue", "spin", "3"])])
    t0 = clk.t; seen = []
    orig = s.busy_now
    def busy_now(allow=()):
        if clk.t - t0 > 100: s.procs.rows = []                          # the other program ends after 100 s
        return orig(allow)
    s.busy_now = busy_now
    orig_run = run.run
    def spy(argv, env, timeout, tee=None):
        seen.append((clk.t - t0, " ".join(argv))); return orig_run(argv, env, timeout, tee)
    run.run = spy
    s.run()
    place = [t for t, c in seen if "place_ball.py" in c]
    assert place and place[0] > 100 and "another program commands the robot (characterize.py)" in note.sent[0] and s.done == [1]
    s, run, note, world, strat, clk = make(steps, ball=(0.0, 0.6), robot=None, budget=60)
    s.run()
    assert run.cmds("place_ball.py") == [] and alerts_but_end(note)[0].startswith("robot not seen")


def test_runner_gives_each_signal_its_time():
    import inspect
    src = inspect.getsource(A.Runner.stop)
    assert "(signal.SIGINT, 1.5), (signal.SIGTERM, 1.5), (signal.SIGKILL" in src


# ------------------------------------------------------------------ the shipped sessions and their plans
def arg_forms_ok(test, args):
    """the argument forms of characterize.py as its docstring gives them"""
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if test == "step": return len(args) in (2, 6) and args[0] in ("fwd", "side") and all(num(v) for v in args[1:])
    if test == "spin": return len(args) in (1, 2) and all(num(v) for v in args)
    if test == "ram": return len(args) in (1, 4) and all(num(v) for v in args)
    if test == "goto": return len(args) == 3 and all(num(v) for v in args)
    if test == "sweep": return len(args) in (5, 6) and all(num(v) for v in args) and int(args[4]) >= 1
    return False


def test_characterize_docstring_has_these_forms():
    doc = open(os.path.join(SOLO, "characterize.py")).read().split('"""')[1]
    for form in ("step fwd|side <speed m/s> [x0 y0 dir_deg length]", "spin <rate rad/s> [seconds]", "ram <speed m/s> [dir_deg runup past]", "goto x y th_deg",
                 "sweep x0 x1 y0 y1 rows [speed]"):
        assert form in doc, form


@pytest.mark.parametrize("name", sorted(f for f in os.listdir(os.path.join(SOLO, "sessions")) if f.endswith(".jsonl")))
def test_every_plan_a_shipped_session_names_exists_and_fits(name):
    head, steps, vs = A.load_session(os.path.join(SOLO, "sessions", name), "blue")       # the defaults, no --var
    plans = [A.expand(st["plan"], vs) for st in steps if st["kind"] == "series"]
    assert plans or any(st["kind"] == "test" for st in steps)        # a ball session may run tests without a plan
    for st in steps:
        if st["kind"] == "test": assert arg_forms_ok(st["test"], st.get("args", [])), (name, st)
    for p in plans:
        assert os.path.isfile(p) and os.path.dirname(p) == os.path.join(SOLO, "sessions", "plans"), p
        lines = [json.loads(l) for l in open(p) if l.strip() and not l.startswith("#")]
        assert lines, p
        for d in lines:
            assert arg_forms_ok(d["test"], d["args"]), (p, d)
            assert set(d) <= {"label", "test", "args", "at", "spot", "surface", "box", "brake_box", "settle", "speeds", "back"}, d
    for st in steps:
        if st["kind"] == "series" and st.get("from"):
            assert st["from"] <= len([l for l in open(A.expand(st["plan"], vs)) if l.strip()])
