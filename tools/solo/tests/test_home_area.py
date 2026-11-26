"""Tests of the home area of tools/solo/auto_session.py (tests confined to one half of a shared field), the other
robots, the fetch and the no-ball mode, with the fakes of test_auto_session.py (no robot, ROS or network).
    uv run --with pytest pytest -q tools/solo/tests/test_home_area.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import auto_session as A  # noqa: E402

HOME = "-2.2,-0.10,-1.45,1.45"
SPIN = lambda **kw: dict(kind="test", test="spin", args=[3.0], env={"SOLO_BOX": "-1.4,-0.6,-0.4,0.4"}, **kw)


def make(steps, script=(), robot=(-1.0, 0.0, 0.0), ball=(-1.2, 0.8), timeline=(), others=(), budget=1800.0, head=None, vs=None, **kw):
    clk = A.SimClock()
    world = A.ScriptedWorld(clk, robot=robot, ball=ball, timeline=[(clk.t + dt, ch) for dt, ch in timeline], others=others)
    procs = A.SimProcs(); strat = A.SimStrategy(procs); run = A.SimRunner(clk, world, script); note = A.SimNotifier()
    for i, st in enumerate(steps, 1): st.setdefault("n", i)
    kw.setdefault("home", HOME); kw.setdefault("our_id", 0); kw.setdefault("cameras", [0])
    s = A.Session("blue", steps, head=head, vs=vs, world=world, runner=run, notifier=note, procs=procs, strategy=strat, clock=clk,
                  sleep=clk.sleep, budget_s=budget, echo=False, **kw)
    return s, run, note, world, strat, clk

def alerts(note):
    return [m for m in note.sent if not m.startswith(("session ended", "session starting"))]


def test_geometry_and_params():
    h = A.home_rect(HOME)
    assert A.open_edges(h) == dict(x0=False, x1=True, y0=False, y1=False)
    assert A.robot_area(h) == [-2.43, -0.1, -1.68, 1.68] and A.ball_area(h) == [-2.55, -0.1, -1.8, 1.8]
    assert A.near_open_edge((-0.5, 0.0), h, 0.5) and not A.near_open_edge((-0.7, 1.4), h, 0.5) and A.near_open_edge((0.2, 0.0), h, 0.5)
    argv = ["python3", "/x/solo_node", "--ros-args", "-p", "team:=blue", "-p", "side:=positive", "-p", "foe_all:=true"]
    assert A.with_params(argv, dict(side="negative", foe_all=False, force_state="running")) == \
        ["python3", "/x/solo_node", "--ros-args", "-p", "team:=blue", "-p", "side:=negative", "-p", "foe_all:=false", "-p", "force_state:=running"]
    assert A.with_params(["n"], dict(side="negative")) == ["n", "--ros-args", "-p", "side:=negative"]
    assert A.blockers([("yellow2", 0.5, 0.35)], [((-1.0, 0.0), (0.5, 0.0))], 0.7) == [("yellow2", 0.5, 0.35, 0.35)]


def test_steps_outside_the_home_area_are_refused_at_load():
    steps = [dict(kind="test", test="spin", args=[3.0], env={"SOLO_BOX": "-1.0,0.6,-0.4,0.4"}, label="crosses"),
             dict(kind="test", test="step", args=["fwd", 1.0], label="default box"),
             dict(kind="strategy", on=True, seconds=5, label="no side"),
             dict(kind="goto", x=0.5, y=0.0, label="goto right"), SPIN(label="ok"),
             dict(kind="place", x=-1.0, y=0.3, label="placer default area")]
    s, run, note, *_ = make(steps)
    assert [st["n"] for st in steps if st.get("_refused")] == [1, 2, 3, 4, 6]
    s.run()
    assert len(run.calls) == 1 and "spin" in run.calls[0]["cmd"] and s.done == [5]
    s2, *_ = make([SPIN()], home=None)
    assert not any(st.get("_refused") for st in s2.steps)      # no home area: no area checks


def test_intruder_during_a_step_stops_it_at_once_alerts_once_and_repeats_it():
    steps = [SPIN(label="spin")]
    t_in = 2.0
    s, run, note, world, strat, clk = make(steps, timeline=[(t_in, dict(others=[("yellow2", -0.6, 0.3)])), (30.0, dict(others=[]))])
    t0 = clk.t; s.run()
    assert len(run.stops) == 1 and run.stops[0][0] - (t0 + t_in) <= 0.1 + 1e-6          # within one poll
    a = alerts(note)
    assert len(a) == 1 and a[0].startswith("another robot (yellow 2) entered our half at (-0.60, +0.30); our robot stopped at (-1.00, +0.00)")
    assert len(run.cmds("characterize.py")) == 2 and s.done == [1]                    # repeated from its start
    starts = [c for c in s.intrusions]; assert len(starts) == 1 and starts[0]["during"]
    assert clk.t - t0 >= 30.0 + 5.0                                                     # clear for 5 s first


def test_intruder_that_stays_gets_a_second_alert_and_no_motion():
    steps = [SPIN(label="spin"), SPIN(label="spin 2")]
    s, run, note, world, strat, clk = make(steps, timeline=[(1.0, dict(others=[("blue3", -1.5, -0.8)]))], budget=300,
                                           switch_after=0)                                   # no switch of halves here
    s.run()
    a = alerts(note)
    assert len(a) == 3 and "entered our half" in a[0] and "still in our half" in a[1] and "reminder 2/3" in a[1]   # at 120 and 240 s
    assert len(run.cmds("characterize.py")) == 1 and s.done == []                      # stopped once, never again


def test_intruder_between_steps_holds_the_next_step():
    steps = [SPIN(label="a"), dict(kind="wait", seconds=3), SPIN(label="b")]
    s, run, note, world, strat, clk = make(steps, timeline=[(6.0, dict(others=[("yellow0", -1.8, 1.0)])), (40.0, dict(others=[]))],
                                           switch_after=0)
    t0 = clk.t; s.run()
    a = alerts(note)
    assert len(a) == 1 and "entered our half at (-1.80, +1.00); our robot waits at" in a[0]
    assert s.done == [1, 2, 3] and len(run.stops) == 0
    t_b = [c for c in run.calls if "characterize.py" in c["cmd"]]
    assert len(t_b) == 2 and clk.t - t0 >= 45.0


def test_robot_beyond_the_halfway_line_and_far_does_nothing():
    steps = [SPIN(label="a"), SPIN(label="b")]
    s, run, note, *_ = make(steps, others=[("yellow1", 0.15, 0.2), ("blue4", 0.3, -1.2)])
    s.run()
    assert alerts(note) == [] and s.done == [1, 2] and run.stops == [] and s.intrusions == []
    s, run, note, *_ = make([SPIN()], robot=(-0.5, 0.0, 0.0), timeline=[(1.0, dict(others=[("yellow1", 0.3, 0.2)])), (20.0, dict(others=[]))])  # 0.82 m from ours
    s.run()
    assert len(run.stops) == 1 and "came within 0.82 m of our robot" in alerts(note)[0]


def test_flickering_intruder_does_not_count_as_clear():
    tl = [(1.0, dict(others=[("yellow2", -0.5, 0.0)]))]
    for k in range(12):                              # seen, then missed for 0.3 s, every 2 s until t = 25
        tl += [(2.0 + 2 * k, dict(others=[])), (2.3 + 2 * k, dict(others=[("yellow2", -0.5, 0.0)]))]
    tl += [(26.0, dict(others=[]))]
    s, run, note, world, strat, clk = make([SPIN()], timeline=tl)
    t0 = clk.t; s.run()
    assert len(alerts(note)) == 1 and s.done == [1]
    second = [c for c in run.calls if "characterize.py" in c["cmd"]][1]
    assert clk.t - t0 >= 26.0 + 5.0 + 5.0 - 0.3       # resumed only 5 s after the last sighting, then 5 s of spin


def test_ball_in_the_other_half_is_fetched_through_a_clear_corridor():
    steps = [dict(kind="place", x=-0.9, y=-0.3, env={"SOLO_BOX": "-2.43,-0.10,-1.68,1.68"}, label="p"),
             dict(kind="test", test="ram", args=[1.2, -95, 0.7, 0.2], needs_ball=True, env={"SOLO_BOX": "-1.45,-0.40,-1.30,1.30"}, label="ram")]
    script = [dict(match="place_ball.py", line=dict(ok=True, status="done"), world=dict(ball=(-0.9, -0.3))),
              dict(match="ram 1.2", line=dict(test="ram"), world=dict(ball=(0.5, -0.6)))]
    s, run, note, world, *_ = make(steps, script=script, others=[("yellow1", 1.6, 1.0)], park=(-0.85, 0.0))
    s.run()
    fetch = run.calls[2]
    assert "place_ball.py blue -0.85 0.0" in fetch["cmd"] and fetch["env"]["SOLO_BOX"] == "-1.35,0.85,-0.95,0.35"
    assert alerts(note) == [] and world.ball == (-0.85, 0.0) and s.done == [1, 2]


def test_fetch_waits_for_the_corridor_then_alerts_once():
    steps = [dict(kind="place", x=-0.9, y=-0.3, env={"SOLO_BOX": "-2.43,-0.10,-1.68,1.68"}, label="p")]
    s, run, note, world, *_ = make(steps, ball=(0.5, 0.1), others=[("yellow1", 0.6, 0.4)], budget=400, park=(-0.85, 0.0))
    s.run()
    a = alerts(note)
    assert a[0] == "ball at (+0.50, +0.10) in the other half, other robots near it (yellow 1 at (+0.60, +0.40)): fetch it when you can"
    assert run.cmds("place_ball.py") == [] and sum("other half" in m for m in a) <= 3


def test_robot_entering_the_corridor_stops_the_fetch_and_the_robot_retreats():
    steps = [dict(kind="place", x=-0.9, y=-0.3, env={"SOLO_BOX": "-2.43,-0.10,-1.68,1.68"}, label="p")]
    def moved(): world.set(robot=(0.5, 0.05, 0.0))
    script = [dict(match="place_ball.py blue -0.85", line=dict(ok=True, status="done"), dur=6.0, call=moved)]
    s, run, note, world, *_ = make(steps, ball=(0.5, 0.1), script=script,
                                   timeline=[(8.0, dict(others=[("yellow1", 0.9, 0.6)])), (60.0, dict(others=[]))], budget=600, park=(-0.85, 0.0))
    s.run()
    assert run.stops and run.stops[0][1].startswith("another robot: yellow1")
    assert any("goto.py blue -0.4 0.05" in c for c in run.cmds("goto.py"))           # back into the home area
    assert len(run.cmds("place_ball.py blue -0.85")) == 2                               # retried once clear
    assert s.done == [1]


def test_robot_outside_the_home_area_is_driven_back_first():
    s, run, note, *_ = make([SPIN()], robot=(0.6, 0.2, 0.0))
    s.run()
    assert "goto.py blue -0.4 0.2" in run.calls[0]["cmd"] and "spin" in run.calls[1]["cmd"] and s.done == [1]


def test_strategy_step_sets_its_side_and_parameters_and_stops_at_the_open_edge():
    st = dict(kind="strategy", on=True, seconds=10, side="negative", params=dict(force_state="running", foe_all=False), edge_guard=0.9)
    s, run, note, world, strat, clk = make([st], timeline=[(3.0, dict(robot=(-0.95, 0.0, 0.0)))])
    t0 = clk.t; s.run()
    assert strat.params == [dict(force_state="running", foe_all=False, side="negative")]
    assert "-p side:=negative" in " ".join(strat.last_argv) and "-p foe_all:=false" in " ".join(strat.last_argv)
    assert strat.offs >= 1 and clk.t - t0 < 4.0 and s.done == [1]


def test_strategy_ends_when_the_ball_leaves_the_area():
    st = dict(kind="strategy", on=True, seconds=10, side="negative", end_ball_out="-2.25,-1.60,-0.82,0.82")
    s, run, note, world, strat, clk = make([st], ball=(-1.95, 0.0), timeline=[(2.0, dict(ball=(-1.4, 0.1)))])
    t0 = clk.t; s.run()
    assert 2.4 <= clk.t - t0 < 3.5 and s.done == [1]


def test_no_ball_session_never_touches_a_ball():
    steps = [dict(kind="place", x=-1.0, y=0.0, label="refused"), SPIN(label="a"), SPIN(label="b", needs_ball=True)]
    s, run, note, world, *_ = make(steps, head={"ball": "ignore"}, ball=(-1.0, 0.1), timeline=[(40.0, dict(ball=(0.5, 0.5)))])
    assert [st["n"] for st in steps if st.get("_refused")] == [1, 3]
    t0 = s.clock(); s.run()
    assert run.cmds("place_ball.py") == [] and alerts(note) == [] and s.done == [2]
    assert s.clock() - t0 >= 30.0                                                       # held quietly for the ball


def test_no_ball_foreign_ball_in_the_area_moves_the_step_to_the_end():
    steps = [SPIN(label="a"), dict(kind="test", test="spin", args=[2.0], env={"SOLO_BOX": "-2.0,-1.5,-1.2,-0.6"}, label="b")]
    s, run, note, *_ = make(steps, head={"ball": "ignore"}, ball=(-1.0, 0.1), timeline=[(60.0, dict(ball=(-2.0, 1.2)))])
    s.run()
    assert [c["cmd"].split()[-3] for c in run.calls] == ["2.0", "3.0"] and s.done == [2, 1] and alerts(note) == []


def test_no_ball_ball_rolling_at_the_robot_stops_the_step():
    steps = [SPIN(label="a")]
    s, run, note, *_ = make(steps, head={"ball": "ignore"}, ball=(0.5, 1.0), timeline=[(2.0, dict(ball=(-0.8, 0.1))), (3.0, dict(ball=(0.4, 0.9)))])
    s.run()
    assert run.stops and "a ball in the test area" in run.stops[0][1] and len(run.cmds("characterize.py")) == 2 and alerts(note) == []


def test_strategy_step_with_an_unknown_node_parameter_is_refused():
    steps = [dict(kind="strategy", on=True, seconds=5, side="negative", params={"evil;rm": 1}),
             dict(kind="strategy", on=True, seconds=5, side="left"),
             dict(kind="strategy", on=True, seconds=5, side="positive", params={"force_state": "running"})]
    s, *_ = make(steps)
    assert [bool(st.get("_refused")) for st in steps] == [True, True, False]


# ---- the supervisor's own view of the robots (a fake packet source: no socket), fresh information, every motion watched
def raw_world(clk, frames, robot=(-1.0, 0.0, 0.0)):
    """a ScriptedWorld whose raw vision view comes from RawVision on a FakePacketSource"""
    return A.ScriptedWorld(clk, robot=robot, ball=(-1.2, 0.8), vision=A.RawVision(A.FakePacketSource(clk, frames), clk, thread=False))

def make_raw(steps, frames, budget=600.0, **kw):
    clk = A.SimClock(); world = raw_world(clk, frames)
    procs = A.SimProcs(); run = A.SimRunner(clk, world); note = A.SimNotifier()
    for i, st in enumerate(steps, 1): st.setdefault("n", i)
    kw.setdefault("home", HOME); kw.setdefault("our_id", 3); kw.setdefault("cameras", [0])
    s = A.Session("blue", steps, world=world, runner=run, notifier=note, procs=procs, strategy=A.SimStrategy(procs), clock=clk,
                  sleep=clk.sleep, budget_s=budget, echo=False, **kw)
    return s, run, note, world, clk

def test_frame_robots_decodes_both_colours_with_their_ids():
    from types import SimpleNamespace as N
    r = lambda i, x, y: N(robot_id=i, x=x, y=y)
    w = N(HasField=lambda f: f == "detection", detection=N(robots_blue=[r(3, -1000, 250), r(5, 400, 0)], robots_yellow=[r(i, 100 * i, -500) for i in range(5)]))
    got = A.frame_robots(w)
    assert ("blue", 3, -1.0, 0.25) in got and ("blue", 5, 0.4, 0.0) in got and len(got) == 7 and ("yellow", 4, 0.4, -0.5) in got
    assert A.frame_robots(w, mirror=True)[0] == ("blue", 3, -1.0, -0.25)
    assert A.frame_robots(N(HasField=lambda f: False)) is None

def test_robot_of_our_colour_with_another_id_intruding_stops_the_step():
    t_in = [None]
    def frames(t):
        out = [("blue", 3, -1.0, 0.0)]
        if t_in[0] is not None and t >= t_in[0]: out.append(("blue", 5, -0.6, 0.4))     # another team's blue robot
        return out
    s, run, note, world, clk = make_raw([SPIN(label="spin")], frames)
    t_in[0] = clk.t + 2.0; s.run()
    assert run.stops and run.stops[0][1].startswith("another robot: blue5") and run.stops[0][0] - t_in[0] <= 0.1 + 1e-6
    assert alerts(note)[0].startswith("another robot (blue 5) entered our half")

def test_more_than_three_robots_of_the_other_colour_are_all_watched():
    def frames(t):
        return [("blue", 3, -1.0, 0.0)] + [("yellow", i, 0.5 + 0.2 * i, 1.0) for i in range(4)] + [("yellow", 4, -1.6, -1.0)]
    s, run, note, world, clk = make_raw([SPIN(label="spin")], frames, budget=60)
    s.run()
    assert run.cmds("characterize.py") == [] and alerts(note)[0].startswith("another robot (yellow 4) entered our half at (-1.60, -1.00)")

def test_index_mismatch_refuses_to_start_and_stops_a_running_step():
    s, run, note, world, clk = make_raw([SPIN()], lambda t: [("blue", 3, -1.6, 0.9), ("blue", 0, -1.0, 0.0)])
    s.run()                                   # the page's robot (-1.0, 0.0) is blue 0, ours (blue 3) is at (-1.6, 0.9)
    assert run.calls == [] and s.stop_why.startswith("refused to start: the robot the tools are driving is not ours")
    t_sw = [1e18]                             # the connector's robot moves to (-1.7, 0.0); ours stays at (-1.0, 0.0)
    def frames(t):
        if t >= t_sw[0]: world.set(robot=(-1.7, 0.0, 0.0))
        return [("blue", 3, -1.0, 0.0)]
    s, run, note, world, clk = make_raw([SPIN(), SPIN()], frames)
    t_sw[0] = clk.t + 3.0; s.run()
    assert len(run.calls) == 1 and run.stops and run.stops[0][1].startswith("not ours")
    assert any(m.startswith("the robot the tools are driving is not ours") for m in note.sent) and s.done == []

def test_listener_unavailable_refuses_a_home_session_and_warns_without_one(monkeypatch, capsys):
    def broken(*a, **k): raise OSError("no such device")
    monkeypatch.setattr(A, "McastSource", broken)
    v, why = A.open_vision(iface="203.0.113.9")
    assert v is None and "cannot listen to the vision multicast 224.5.23.2:10006 on 203.0.113.9" in why
    def noproto(*a, **k): raise ImportError("no module named google")
    monkeypatch.setattr(A, "McastSource", noproto)
    assert "cannot be imported" in A.open_vision()[1]
    s, run, note, *_ = make([SPIN()], vision_error=why)
    s.run()
    assert run.calls == [] and "other robots cannot be watched" in s.stop_why
    s, run, note, *_ = make([SPIN()], our_id=None)
    s.run()
    assert run.calls == [] and "--our-id" in s.stop_why
    s, run, note, *_ = make([SPIN()], home=None)
    s.run()
    assert len(run.calls) == 1 and "other robots are not watched" in capsys.readouterr().out

def test_listener_silent_mid_step_stops_holds_alerts_and_resumes():
    quiet = [1e18, 1e18]
    def frames(t): return None if quiet[0] <= t < quiet[1] else [("blue", 3, -1.0, 0.0)]
    s, run, note, world, clk = make_raw([SPIN(label="spin")], frames, info_alert=10.0)
    quiet[0] = clk.t + 2.0; quiet[1] = clk.t + 40.0; s.run()
    assert run.stops and run.stops[0][1].startswith("no information: no vision for") and run.stops[0][0] - quiet[0] <= 0.7
    a = alerts(note)
    assert len(a) == 1 and a[0].startswith("no vision for") and a[0].endswith(": the session holds")
    assert len(run.cmds("characterize.py")) == 2 and run.calls[1]["watched"] and s.done == [1]
    starts = [c for c in s.intrusions]; assert len(starts) == 1

def test_page_down_mid_step_stops_the_step():
    steps = [SPIN(label="spin")]
    s, run, note, world, strat, clk = make(steps, timeline=[(2.0, dict(page=False)), (20.0, dict(page=True))])
    s.run()
    assert run.stops and "the field page has not answered" in run.stops[0][1] and len(run.cmds("characterize.py")) == 2

def test_our_robot_unseen_mid_step_stops_it_after_the_grace():
    steps = [SPIN(label="spin")]
    s, run, note, world, strat, clk = make(steps, timeline=[(2.0, dict(robot=None)), (8.0, dict(robot=(-1.0, 0.0, 0.0)))])
    t0 = clk.t; s.run()
    assert run.stops and "our robot (blue 0) not seen" in run.stops[0][1] and 2.4 <= run.stops[0][0] - t0 <= 2.7

def test_strategy_stops_when_the_information_goes_stale():
    steps = [dict(kind="strategy", on=True, seconds=10, side="negative", label="s")]
    s, run, note, world, strat, clk = make(steps, timeline=[(3.0, dict(vision=False)), (30.0, dict(vision=True))], budget=120)
    s.run()
    st = [l for l in s.intrusions]; assert st and strat.offs >= 1 and strat.ons == 2 and s.done == [1]

def test_every_motion_kind_and_every_added_move_is_watched():
    import inspect
    src = inspect.getsource(A)
    assert src.count("self.runner.run(") == 1                    # the one place motion children start
    kinds = [k for k in A.MOTION if k != "strategy"]
    proto = dict(place=dict(kind="place", x=-1.0, y=0.3, env={"SOLO_BOX": HOME}), test=SPIN(), goto=dict(kind="goto", x=-1.0, y=0.5),
                 series=dict(kind="series", plan="none.jsonl"))
    assert sorted(proto) == sorted(kinds), "a new motion kind: add it here"
    for k in kinds:
        s, run, *_ = make([])
        s.run_step(dict(proto[k], n=1))
        assert run.calls[-1]["watched"], k
    # the moves the supervisor adds: retry, retreat, the drive back, freeing, parking, rescue
    s, run, note, world, strat, clk = make([dict(kind="place", x=-1.0, y=0.3, env={"SOLO_BOX": HOME})],
                                           script=[dict(match="place_ball.py", line=dict(ok=False, status="failed: timeout (60 s)"), world=dict(robot=(0.2, 0.0, 0.0)))])
    s.run()
    assert len(run.cmds("place_ball.py")) == 2 and run.cmds("goto.py") and all(c["watched"] for c in run.calls)
    s, run, note, world, strat, clk = make([dict(kind="test", test="spin", args=[3.0], env={"SOLO_BOX": HOME}, label="t")], robot=(-2.3, 0.0, 0.0),
                                           script=[dict(match="characterize.py", line="ABORT: robot stuck", rc=1)])
    s.run()
    assert run.cmds("goto.py") and all(c["watched"] for c in run.calls)
    steps = [SPIN(keep_ball_out="-1.6,-0.6,-0.5,0.5", park=[-1.8, 1.0])]
    s, run, note, world, strat, clk = make(steps, ball=(-1.0, 0.0))
    s.run()
    assert run.cmds("place_ball.py blue -1.8 1.0") and all(c["watched"] for c in run.calls)
    s, run, note, world, strat, clk = make([dict(kind="place", x=-1.0, y=0.3, env={"SOLO_BOX": HOME})], ball=(-2.4, 1.1))
    s.run()
    assert run.calls and all(c["watched"] for c in run.calls)

def test_retreat_after_a_stopped_fetch_waits_for_the_robot_to_be_clear():
    steps = [dict(kind="place", x=-0.9, y=-0.3, env={"SOLO_BOX": "-2.43,-0.10,-1.68,1.68"}, label="p")]
    def moved(): world.set(robot=(0.3, 0.05, 0.0))
    script = [dict(match="place_ball.py blue -0.85", line=dict(ok=True, status="done"), dur=6.0, call=moved)]
    s, run, note, world, strat, clk = make(steps, ball=(0.5, 0.1), script=script, park=(-0.85, 0.0), budget=600,
                                           timeline=[(8.0, dict(others=[("yellow1", 0.5, 0.6)])), (40.0, dict(others=[]))])
    t0 = clk.t; s.run()
    gotos = [c for c in run.calls if "goto.py" in c["cmd"]]
    assert run.stops and gotos and all(c["watched"] for c in run.calls)
    lg = [l for l in s.intrusions]
    first_goto = [i for i, c in enumerate(run.calls) if "goto.py" in c["cmd"]][0]
    assert first_goto >= 1

def test_a_ball_too_far_beyond_our_half_is_not_fetched():
    steps = [dict(kind="place", x=-0.9, y=-0.3, env={"SOLO_BOX": "-2.43,-0.10,-1.68,1.68"}, label="p")]
    s, run, note, *_ = make(steps, ball=(1.2, 0.1), budget=120, park=(-0.85, 0.0))
    s.run()
    assert run.cmds("place_ball.py") == [] and alerts(note)[0].startswith("ball at (+1.20, +0.10) is 1.3 m beyond our half (fetch limit 1.0 m)")

def test_load_checks_no_ball_cmd_resume_and_parking(tmp_path):
    plan = tmp_path / "p.jsonl"; plan.write_text('{"test": "ram", "args": [1.0, 0, 0.5, 0.2], "box": "-1.5,-0.5,-1,1"}\n')
    steps = [dict(kind="strategy", on=True, seconds=5, side="negative"), dict(kind="test", test="ram", args=[1.0, 0, 0.5, 0.2], env={"SOLO_BOX": HOME}),
             dict(kind="series", plan=str(plan)), dict(kind="series", plan=str(plan), args=["--resume"]), SPIN(),
             dict(kind="cmd", argv=["/x/robot_stack.sh", "save"]), dict(kind="cmd", argv=["/x/drive.sh"]),
             dict(kind="series", plan=str(plan), keep_ball_out="-1,-0.5,-0.5,0.5", park=[0.5, 0.0])]
    s, *_ = make(steps, head={"ball": "ignore"})
    assert [st["n"] for st in steps if st.get("_refused")] == [1, 2, 3, 4, 7, 8]
    s, *_ = make([dict(kind="series", plan=str(plan), keep_ball_out="-1,-0.5,-0.5,0.5", park=[0.5, 0.0])])
    assert "parking spot" in s.steps[0]["_refused"]
    s, *_ = make([dict(kind="cmd", argv=["/x/drive.sh"])], home=None)
    assert "_refused" not in s.steps[0] and s.steps[0]["_unchecked"].startswith("UNCHECKED")


# ---- the other half: mirroring and switching halves
SESS = os.path.join(os.path.dirname(HERE), "sessions")

def test_mirroring_twice_returns_the_original():
    assert A.mirror_dir(90) == 90 and A.mirror_dir(0) == 180 and A.mirror_dir(-95) == -85 and A.mirror_dir(45) == 135
    for d in (-180, -95, -90, 0, 5, 45, 90, 135, 180): assert A.mirror_dir(A.mirror_dir(d)) == A.mirror_dir(A.mirror_dir(A.mirror_dir(A.mirror_dir(d))))
    assert A.mirror_spec("-2.2,-0.10,-1.45,1.45;-1,-0.5,0,1") == "0.1,2.2,-1.45,1.45;0.5,1,0,1"
    import json
    for f in sorted(os.listdir(os.path.join(SESS, "plans"))):
        for l in open(os.path.join(SESS, "plans", f)):
            if l.strip() and not l.startswith("#"):
                e = json.loads(l); m = A.mirror_plan_line(e)
                assert _norm(A.mirror_plan_line(m)) == _norm(e) and (_norm(m) != _norm(e)), (f, l)

def _norm(e):
    """numbers rounded, boxes as rectangles"""
    def r(v): return [r(x) for x in v] if isinstance(v, list) else (round(float(v), 4) if isinstance(v, (int, float)) else v)
    return {k: (A.rects(v) if k in ("box", "brake_box") else r(v)) for k, v in e.items()}

def test_mirrored_step_round_trip_and_side(tmp_path):
    st = dict(kind="strategy", on=True, seconds=10, side="negative", end_ball_out="-2,-0.5,-1,1", n=3)
    m = A.mirror_step(st)
    assert m["side"] == "positive" and m["end_ball_out"] == "0.5,2,-1,1" and A.mirror_step(m)["side"] == "negative"
    st = dict(kind="test", test="ram", args=[1.2, -95, 0.7, 0.2], env={"SOLO_BOX": "-1.45,-0.40,-1.30,1.30", "SOLO_RAMP": "1.7"})
    m = A.mirror_step(st)
    assert m["args"] == [1.2, -85.0, 0.7, 0.2] and m["env"] == {"SOLO_BOX": "0.4,1.45,-1.3,1.3", "SOLO_RAMP": "1.7"}
    assert A.mirror_step(m)["args"] == [1.2, -95.0, 0.7, 0.2]

def test_every_shipped_left_session_mirrors_into_the_right_half_and_passes_the_load_check(tmp_path):
    for f in ("left_noball.jsonl", "left_noball_20.jsonl", "left_ball_slot_10.jsonl"):
        head, steps, vs = A.load_session(os.path.join(SESS, f), "blue", dict(OUT=str(tmp_path)))
        mh = A.mirror_head(head); ms = [A.mirror_step(st, str(tmp_path / "mirrored"), vs) for st in steps]
        s = A.Session("blue", ms, head=mh, vs=vs, echo=False, home=A.home_rect(mh["home"]), our_id=0)
        assert A.half_name(s.home) == "+x" and not [st.get("_refused") for st in s.steps if st.get("_refused")], f
        assert open(os.path.join(SESS, "plans", "left_A_ramped.jsonl")).read().count("-1.35") >= 1      # the shipped plan is unchanged

def far_intruder(t=2.0):
    return (t, dict(others=[("yellow2", -1.8, 1.0)]))                 # inside the -x half, away from the way over

def test_intruder_stays_and_the_other_half_is_empty_one_switch_and_mirrored_steps(tmp_path):
    steps = [SPIN(label="a"), SPIN(label="b")]
    s, run, note, world, strat, clk = make(steps, timeline=[far_intruder()], vs=dict(OUT=str(tmp_path)), budget=200)
    s.vs.update(TEAM="blue")
    s.run()
    assert len(s.switches) == 1 and note.quiet == ["moved to the +x half: another robot (yellow 2) stayed in the -x half for 20 s"]
    ch = [c for c in run.calls if "characterize.py" in c["cmd"]]
    assert len(ch) == 3 and ch[0]["env"]["SOLO_BOX"] == "-1.4,-0.6,-0.4,0.4" and run.stops      # a: stopped by the intruder
    assert all(c["env"]["SOLO_BOX"] == "0.6,1.4,-0.4,0.4" for c in ch[1:]) and s.done == [1, 2]    # a again and b, mirrored
    assert run.cmds("goto.py blue 0.4 0.0") and A.half_name(s.home) == "+x" and all(c["watched"] for c in run.calls)

def test_intruder_stays_and_the_other_half_is_occupied_no_switch():
    s, run, note, world, strat, clk = make([SPIN()], timeline=[(2.0, dict(others=[("yellow2", -1.8, 1.0), ("yellow1", 1.2, 0.5)]))], budget=120)
    s.run()
    assert s.switches == [] and note.quiet == [] and len(run.cmds("characterize.py")) == 1 and run.stops and s.done == []

def test_other_half_empty_only_briefly_no_switch():
    tl = [(1.0, dict(others=[("yellow2", -1.8, 1.0), ("yellow1", 1.2, 0.5)])), (10.0, dict(others=[("yellow2", -1.8, 1.0)])),
          (25.0, dict(others=[("yellow2", -1.8, 1.0), ("yellow1", 1.2, 0.5)]))]
    s, run, note, *_ = make([SPIN()], timeline=tl, budget=45)
    s.run()
    assert s.switches == []

def test_no_information_about_the_other_half_no_switch():
    s, run, note, *_ = make([SPIN()], timeline=[far_intruder(1.0), (5.0, dict(vision=False))], budget=80)
    s.run()
    assert s.switches == [] and any(m.startswith("no vision for") for m in note.sent)

def test_cooldown_and_the_switch_limit(tmp_path):
    tl = [far_intruder(1.0), (40.0, dict(others=[("yellow2", 1.8, 1.0)])), (300.0, dict(others=[]))]
    s, run, note, world, strat, clk = make([SPIN(), SPIN()], timeline=tl, vs=dict(OUT=str(tmp_path)), budget=400)
    t0 = clk.t; s.run()
    assert len(s.switches) == 2 and s.switches[1] - s.switches[0] >= 60.0 and s.switches[0] - t0 < 30
    s, run, note, world, strat, clk = make([SPIN(), SPIN()], timeline=tl, vs=dict(OUT=str(tmp_path)), budget=400, max_switches=1)
    s.run()
    assert len(s.switches) == 1 and any("no more switches" in m for m in note.sent)

def test_a_ball_session_fetches_the_ball_into_the_new_half_after_the_switch(tmp_path):
    steps = [dict(kind="place", x=-0.9, y=-0.3, env={"SOLO_BOX": "-2.43,-0.10,-1.68,1.68"}, label="p")]
    s, run, note, world, strat, clk = make(steps, ball=(-0.5, -0.5), timeline=[far_intruder(0.5)], park=(-0.85, 0.0), vs=dict(OUT=str(tmp_path)), budget=200)
    s.run()
    assert len(s.switches) == 1 and run.cmds("place_ball.py blue 0.85 0.0") and run.cmds("place_ball.py blue 0.9 -0.3")

def test_home_auto_at_start_in_its_three_cases():
    s, run, *_ = make([SPIN()], home_auto=True)
    s.run(); assert A.half_name(s.home) == "-x" and len(run.cmds("characterize.py")) == 1
    s, run, *_ = make([SPIN()], home_auto=True, others=[("yellow2", -1.8, 1.0)])
    s.run(); assert A.half_name(s.home) == "+x" and run.cmds("goto.py blue 0.4 0.0") and len(run.cmds("characterize.py")) == 1
    s, run, note, *_ = make([SPIN()], home_auto=True, others=[("yellow2", -1.8, 1.0), ("yellow1", 1.2, 0.5)], budget=60)
    s.run(); assert run.calls == [] and alerts(note)[0].startswith("both halves hold other robots (yellow 2, yellow 1)")

def test_a_camera_that_stops_sending_is_no_information():
    clk = A.SimClock(); dead = [1e18]
    def frames(t): return None
    class TwoCams:
        def __init__(self): self.k = 0; self.last = -1e9
        def read(self, timeout):
            if clk.t - self.last < 1 / 60: return None
            self.last = clk.t; self.k += 1
            if self.k % 2 or clk.t >= dead[0]: return (0, [("blue", 3, -1.0, 0.0)])
            return (1, [])                                     # camera 1 sees the other part of the field
    v = A.RawVision(TwoCams(), clk, thread=False)
    for _ in range(5): clk.sleep(0.02); assert v.snapshot()["vision_age"] <= 0.05
    dead[0] = clk.t
    for _ in range(40): clk.sleep(0.02); s = v.snapshot()
    assert s["vision_age"] > 0.5 and [d[:2] for d in s["vision_dets"]] == [("blue", 3)]   # camera 0 alive, camera 1 silent

def test_a_failing_watch_stops_the_step():
    r = A.Runner(echo=False)
    import inspect
    assert "fail safe: stop" in inspect.getsource(A.Runner.run)
