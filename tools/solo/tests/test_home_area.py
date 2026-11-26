"""Tests of the home area of tools/solo/auto_session.py (tests confined to one half of a shared field), the other
robots, the fetch and the no-ball mode, with the fakes of test_auto_session.py (no robot, ROS or network).
    uv run --with pytest pytest -q tools/solo/tests/test_home_area.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import auto_session as A  # noqa: E402

HOME = "-2.2,-0.10,-1.45,1.45"
SPIN = lambda **kw: dict(kind="test", test="spin", args=[3.0], env={"SOLO_BOX": "-1.4,-0.6,-0.4,0.4"}, **kw)


def make(steps, script=(), robot=(-1.0, 0.0, 0.0), ball=(-1.2, 0.8), timeline=(), others=(), budget=1800.0, head=None, **kw):
    clk = A.SimClock()
    world = A.ScriptedWorld(clk, robot=robot, ball=ball, timeline=[(clk.t + dt, ch) for dt, ch in timeline], others=others)
    procs = A.SimProcs(); strat = A.SimStrategy(procs); run = A.SimRunner(clk, world, script); note = A.SimNotifier()
    for i, st in enumerate(steps, 1): st.setdefault("n", i)
    kw.setdefault("home", HOME)
    s = A.Session("blue", steps, head=head, world=world, runner=run, notifier=note, procs=procs, strategy=strat, clock=clk,
                  sleep=clk.sleep, budget_s=budget, echo=False, **kw)
    return s, run, note, world, strat, clk

def alerts(note):
    return [m for m in note.sent if not m.startswith("session ended")]


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
    assert not any(st.get("_refused") for st in s2.steps)      # no home area: today's behaviour


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
    s, run, note, world, strat, clk = make(steps, timeline=[(1.0, dict(others=[("blue3", -1.5, -0.8)]))], budget=300)
    s.run()
    a = alerts(note)
    assert len(a) == 3 and "entered our half" in a[0] and "still in our half" in a[1] and "reminder 2/3" in a[1]   # at 120 and 240 s
    assert len(run.cmds("characterize.py")) == 1 and s.done == []                      # stopped once, never again


def test_intruder_between_steps_holds_the_next_step():
    steps = [SPIN(label="a"), dict(kind="wait", seconds=3), SPIN(label="b")]
    s, run, note, world, strat, clk = make(steps, timeline=[(6.0, dict(others=[("yellow0", -1.8, 1.0)])), (40.0, dict(others=[]))])
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
              dict(match="ram 1.2", line=dict(test="ram"), world=dict(ball=(0.8, -0.6)))]
    s, run, note, world, *_ = make(steps, script=script, others=[("yellow1", 1.6, 1.0)], park=(-0.85, 0.0))
    s.run()
    fetch = run.calls[2]
    assert "place_ball.py blue -0.85 0.0" in fetch["cmd"] and fetch["env"]["SOLO_BOX"] == "-1.35,1.15,-0.95,0.35"
    assert alerts(note) == [] and world.ball == (-0.85, 0.0) and s.done == [1, 2]


def test_fetch_waits_for_the_corridor_then_alerts_once():
    steps = [dict(kind="place", x=-0.9, y=-0.3, env={"SOLO_BOX": "-2.43,-0.10,-1.68,1.68"}, label="p")]
    s, run, note, world, *_ = make(steps, ball=(0.9, 0.1), others=[("yellow1", 0.6, 0.4)], budget=400, park=(-0.85, 0.0))
    s.run()
    a = alerts(note)
    assert a[0] == "ball at (+0.90, +0.10) in the other half, other robots near it (yellow 1 at (+0.60, +0.40)): fetch it when you can"
    assert run.cmds("place_ball.py") == [] and sum("other half" in m for m in a) <= 3


def test_robot_entering_the_corridor_stops_the_fetch_and_the_robot_retreats():
    steps = [dict(kind="place", x=-0.9, y=-0.3, env={"SOLO_BOX": "-2.43,-0.10,-1.68,1.68"}, label="p")]
    def moved(): world.set(robot=(0.5, 0.05, 0.0))
    script = [dict(match="place_ball.py blue -0.85", line=dict(ok=True, status="done"), dur=6.0, call=moved)]
    s, run, note, world, *_ = make(steps, ball=(0.9, 0.1), script=script,
                                   timeline=[(2.0, dict(others=[("yellow1", 0.9, 0.6)])), (60.0, dict(others=[]))], budget=600, park=(-0.85, 0.0))
    s.run()
    assert run.stops and ("corridor" in run.stops[0][1] or "near ours" in run.stops[0][1])
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
