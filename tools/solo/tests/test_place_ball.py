"""Unit tests of tools/solo/place_ball.py (no ROS): a few seconds.
    uv run --with pytest pytest -q tools/solo/tests/test_place_ball.py"""
import math, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from place_ball import BallPlacer, roll, roll_speed, HOLD  # noqa: E402
import ballsim  # noqa: E402


def test_roll_model_matches_the_measured_distances():
    assert abs(roll(0.5) - 0.64) < 0.02 and abs(roll(0.8) - 1.48) < 0.03 and abs(roll(1.0) - 2.17) < 0.04
    for s in (0.05, 0.3, 1.0, 2.0):
        assert abs(roll(roll_speed(s)) - s) < 1e-4


def test_never_drives_blind_and_waits_for_a_resting_ball():
    p = BallPlacer((0.5, 0.0))
    for i in range(50):
        vx, vy, w, st = p.step(i * 0.02, None, (0.0, 0.0))
        assert (vx, vy, w) == (0.0, 0.0, 0.0)
    p = BallPlacer((0.5, 0.0))
    for i in range(100):                       # a rolling ball: no command, status waiting
        vx, vy, w, st = p.step(i * 0.02, (-1.0, 0.0, 0.0), (i * 0.02 * 0.3, 0.0))
        assert (vx, vy, w) == (0.0, 0.0, 0.0) and st.startswith("waiting")


def first_plan(target, ball, robot=(0.0, -1.0, 0.0), **kw):
    p = BallPlacer(target, **kw)
    for i in range(60):
        vx, vy, w, st = p.step(i * 0.02, robot, ball)
        if p.finished or p.state != "settle": break
    return p, st


def test_unreachable_cases_are_reported_without_moving():
    for tgt, ball, word in [((0, 0), (2.33, 0.1), "in a goal"), ((0, 0), (2.24, 0.43), "goal frame"),
                            ((2.4, 0.0), (0, 0), "spot outside"), ((0, 0), (0.0, 1.78), "beyond the margin"),
                            ((-2.0, 0.2), (0, 0.5), "defense area")]:
        p, st = first_plan(tgt, ball, keep_out_defense=True)
        assert p.finished and st.startswith("unreachable") and word in st, (tgt, ball, st)


def test_plan_choice_creep_carry_dose_and_the_line_up_point_behind_the_ball():
    for d, kind in ((0.12, "creep"), (0.6, "carry"), (2.0, "dose")):
        p, _ = first_plan((-1.0 + d, 0.2), (-1.0, 0.2), tol=0.05)
        assert p.plan["kind"] == kind, (d, p.plan["kind"])
        S, b, u = p.plan["stand"], p.plan["ball"], p.plan["u"]
        assert (b[0] - S[0]) * u[0] + (b[1] - S[1]) * u[1] > HOLD + 0.04       # behind the ball, not touching it
    p, _ = first_plan((1.0, 0.2), (-1.0, 0.2), tol=0.05)                       # a dose aims short
    assert p.plan["v"] * 0.9 < roll_speed(2.0 - 0.3) + 1e-6


def test_near_the_touch_line_the_line_up_stays_inside_the_margin():
    p, _ = first_plan((0.0, 1.0), (0.3, 1.5))
    S = p.plan["stand"]
    assert abs(S[1]) <= 1.5 + 0.25 + 1e-9 and p.allowed(*S)


def test_simulated_placements_succeed():
    rng = random.Random(11); errs = []
    for i in range(12):
        tgt = ballsim.rand_pt(rng); b = ballsim.rand_pt(rng); rb = ballsim.rand_pt(rng)
        while math.hypot(rb[0] - b[0], rb[1] - b[1]) < 0.3: rb = ballsim.rand_pt(rng)
        r = ballsim.run(tgt, (rb[0], rb[1], 0.0), b, rng, tol=0.05)
        if r["status"].startswith("unreachable"): continue
        assert r["status"] == "done", r["status"]
        assert r["robot_ball_end"] > 0.3 + HOLD - 0.02            # backed away from the ball
        errs.append(r["true_error"])
    assert errs and sum(e <= 0.05 for e in errs) >= len(errs) - 1


def test_short_move_and_moving_start():
    rng = random.Random(3)
    r = ballsim.run((0.6, 0.3), (0.0, 0.0, 0.0), (0.5, 0.3), rng, tol=0.05)
    assert r["status"] == "done" and r["true_error"] < 0.05 and r["pushes"] <= 2
    r = ballsim.run((0.0, 0.0), (0.5, -0.5, 0.0), (-1.0, 0.5), rng, ball_v=(0.6, -0.2), tol=0.05)
    assert r["status"] == "done" and r["true_error"] < 0.05


def real_rest():
    """a resting ball as vision reported it on the real field (x jitter sd 4 mm, 1.5-2 cm peak to peak)"""
    rows = [l.split(",") for l in open(os.path.join(HERE, "data", "ball_rest_jitter.csv")) if not l.startswith("#")]
    return [(float(a), float(b), float(c)) for a, b, c in rows]


def replay(p, samples, robot, t_end, t0=0.0):
    """step at 50 Hz with the robot standing still, the recorded ball samples as they arrived (the stuck check is off:
    the replayed robot does not follow the commands)"""
    p.stuck_check = lambda t, r, out: out
    i, t, st = 0, t0, None
    while t < t_end and not p.finished:
        ball = None
        while i < len(samples) and samples[i][0] + t0 <= t: ball = samples[i][1:]; i += 1
        vx, vy, w, st = p.step(t, robot, ball); t += 0.02
    return t, st


def test_a_jittery_resting_ball_on_the_spot_ends_done_quickly():
    # the real case: a ball resting 0.048 m from the spot, robot 0.21 m from it; the placer waited for a 0.5 s window
    # with less than 1 cm peak to peak, which this jitter rarely gives, until the timeout
    s = real_rest(); robot = (1.158, -0.230, -1.44)
    p = BallPlacer((1.2, -0.4), tol=0.15)
    t, st = replay(p, s, robot, 12.0)
    assert p.finished and st == "done" and p.result()["ok"], st
    assert t < 1.5 + 4.0 + 2.0 + 0.5                     # settle + backoff (robot does not move here) + confirm
    assert p.result()["remeasured"] is True and p.result()["error"] < 0.06


def test_the_final_measurement_is_bounded_and_a_ball_on_the_spot_is_ok_at_the_timeout():
    s = real_rest(); robot = (1.158, -0.230, -1.44)
    p = BallPlacer((1.2, -0.4), tol=0.15); p.stuck_check = lambda t, r, out: out
    i, t = 0, 0.0
    while p.state == "settle" and t < 5.0:
        ball = None
        while s[i][0] <= t: ball = s[i][1:]; i += 1
        p.step(t, robot, ball); t += 0.02
    assert p.state in ("backoff", "confirm") and t < 2.0
    t = p.t_prev; p.balls.clear()                         # nothing seen any more (the robot in the way)
    p.set("confirm", t, "measuring where the ball rests")
    for k in range(200):
        p.step(t + 0.02 * (k + 1), robot, None)
        if p.finished: break
    assert p.finished and p.status == "done" and 0.02 * k <= 2.1
    r = p.result(); assert r["ok"] and r["remeasured"] is False and r["error"] < 0.06
    rng = random.Random(1)                                # a timeout while waiting for a still ball that vision
    noisy = [(a, x + rng.gauss(0, 0.02), y) for a, x, y in s]     # reports far noisier still: on the spot -> ok
    p = BallPlacer((1.2, -0.4), tol=0.15, timeout=1.5)
    t, st = replay(p, noisy, robot, 6.0)
    assert p.finished and st == "done" and p.result()["ok"] and t < 1.6, (st, t)


def drive_kin(p, robot, ball_fn, T, freeze=False):
    """a robot that does what it is told (body frame), optionally pinned in place; ball_fn(t, robot) -> ball or None"""
    x, y, th = robot; t = 0.0; cmds = []
    while t < T and not p.finished:
        vx, vy, w, st = p.step(t, (x, y, th), ball_fn(t, (x, y, th))); cmds.append((t, vx, vy))
        if not freeze:
            c, s = math.cos(th), math.sin(th)
            x += (c * vx - s * vy) * 0.02; y += (s * vx + c * vy) * 0.02; th += w * 0.02
        t += 0.02
    return (x, y, th), cmds


def test_a_ball_hidden_on_the_front_is_looked_for_by_backing_away():
    # the real case: after a creep push the robot stood where the ball was last seen, the ball on its front, unseen
    hidden = (1.298 + HOLD * math.cos(0.71), 0.186 + HOLD * math.sin(0.71))
    seen = lambda t, r: (1.292, 0.189) if t < 0.3 else (hidden if math.hypot(r[0] - hidden[0], r[1] - hidden[1]) > 0.2 else None)
    p = BallPlacer((1.4, 0.3))
    r, cmds = drive_kin(p, (1.298, 0.186, 0.71), seen, 4.0)
    assert math.hypot(r[0] - hidden[0], r[1] - hidden[1]) > 0.3 and p.ball_last == hidden, (r, p.status)
    p = BallPlacer((0.8, 0.8))                             # no ball ever seen: back off, look, then give up
    r, cmds = drive_kin(p, (1.298, 0.186, 0.71), lambda t, r: None, 30.0)
    assert any(vx < -0.1 for _, vx, _ in cmds[:200]) and p.finished and p.status.startswith("failed: ball not seen"), p.status


def test_stuck_against_a_wall_backs_off_once_then_ends():
    p = BallPlacer((0.0, 0.0))
    r, cmds = drive_kin(p, (1.0, 0.0, math.pi), lambda t, r: (-1.0, 0.5), 20.0, freeze=True)
    assert p.finished and p.status.startswith("failed: robot stuck at"), p.status
    res = p.result(); assert res["stuck"] == 2 and res["stuck_failed"]


def test_the_busy_check_skips_the_ancestors_and_counts_only_real_instances():
    from place_ball import ancestors, busy_programs
    mine = ancestors()
    assert os.getpid() in mine and os.getppid() in mine
    table = [(os.getppid(), ["python3", "tools/solo/series.py", "blue", "plan.jsonl"]),            # the caller: skipped
             (77001, ["vim", "notes_about_place_ball.py", "series.py"]),                         # mentions only
             (77002, ["python3", "/x/tools/solo/characterize.py", "blue"])]                      # a real one
    assert busy_programs(table, mine) == ["characterize.py"]
    assert busy_programs([(77003, ["python3", "-u", "tools/solo/place_ball.py", "yellow", "0", "0"])], mine) == ["place_ball.py"]
    assert busy_programs([(77004, ["/usr/bin/python3", "/opt/ros/lib/solo/solo_node", "--ros-args"]),
                          (77005, ["/usr/bin/python3", "/opt/ros/bin/ros2", "run", "solo_strategy", "solo_node"])], mine) == ["solo_node"]
    assert busy_programs([(77006, ["python3", "/x/goto.py", "blue", "1", "0"]), (77007, ["python3", "/x/nudge.py", "0", "0", "0", "1"]),
                          (77008, ["python3", "/x/frame_check.py", "blue"])], mine) == ["frame_check.py", "goto.py", "nudge.py"]


# ---- exits: signals, zeros, the command line ------------------------------------------------------------------
import json, signal, types  # noqa: E402
import pytest  # noqa: E402
import place_ball as PB  # noqa: E402


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def wait(self, dt): self.t += dt


def test_a_signal_ends_the_placement_and_zeros_follow_for_0_3_s():
    for sig, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGINT, "SIGINT")):
        st = PB.Stop(); pl = BallPlacer((0.5, 0.0))
        assert st.end(pl, 1.0) is False and not pl.finished
        st(sig, None); st(signal.SIGINT, None)                          # the first signal is kept
        assert st.end(pl, 2.0) and pl.finished and pl.status == "failed: stopped by " + name
        clk = FakeClock(); sent = []
        k = PB.Stop.zeros(lambda *c: sent.append((clk.t, c)), lambda: True, clk.wait, clk)
        assert all(c == (0.0, 0.0, 0.0) for _, c in sent) and sent[-1][0] > 0.27 and 15 <= k <= 16
    clk = FakeClock(); sent = []; ok = iter([True, True, False])         # the ROS context goes down: no more sends
    assert PB.Stop.zeros(lambda *c: sent.append(c), lambda: next(ok), clk.wait, clk) == 2


def test_stop_installs_handlers_for_sigint_and_sigterm():
    got = {}
    fake = types.SimpleNamespace(SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM, signal=lambda s, h: got.__setitem__(s, h))
    st = PB.Stop(); st.install(fake)
    assert set(got) == {signal.SIGINT, signal.SIGTERM} and all(h is st for h in got.values())


def test_init_rclpy_without_its_signal_handlers_when_it_can(monkeypatch):
    calls = []
    rc = types.SimpleNamespace(init=lambda **kw: calls.append(kw))
    sigs = types.ModuleType("rclpy.signals"); sigs.SignalHandlerOptions = types.SimpleNamespace(NO="NO")
    monkeypatch.setitem(sys.modules, "rclpy.signals", sigs)
    assert PB.init_rclpy(rc) is True and calls == [dict(signal_handler_options="NO")]
    monkeypatch.setitem(sys.modules, "rclpy.signals", None); calls.clear()   # an older distribution
    assert PB.init_rclpy(rc) is False and calls == [{}]


def test_command_line_numbers_must_be_finite_and_positive(monkeypatch):
    for opt in ("--speed", "--carry", "--tol", "--margin", "--wall", "--timeout"):
        for v in ("nan", "inf", "-inf", "0", "-0.2", "abc"):
            monkeypatch.setattr(sys, "argv", ["place_ball.py", "blue", "0", "0", opt, v])
            with pytest.raises(SystemExit) as e: PB.main()
            assert opt in str(e.value) and "finite positive" in str(e.value), (opt, v)
    assert PB.positive("--tol", "0.05") == 0.05


def ros_stubs(on_spin):
    """rclpy and the message packages as stubs: spin_once calls on_spin(k) with the count of calls so far"""
    def mod(name, **kw):
        m = types.ModuleType(name); m.__dict__.update(kw); return m
    W = types.SimpleNamespace(sent=[], subs={}, k=0, ok=True, init=None)
    class V: x = y = z = 0.0
    class Twist:
        def __init__(self): self.linear, self.angular = V(), V()
    class Pub:
        def publish(self, m): W.sent.append((m.linear.x, m.linear.y, m.angular.z))
    class Node:
        def __init__(self, *a): pass
        def create_publisher(self, *a): return Pub()
        def create_subscription(self, typ, topic, cb, q): W.subs[topic] = cb
        def destroy_node(self): pass
    def spin_once(n, timeout_sec=0.0):
        W.k += 1; on_spin(W)
    def init(**kw): W.init = kw
    def shutdown(): W.ok = False
    m = {"rclpy": mod("rclpy", init=init, ok=lambda: W.ok, spin_once=spin_once, shutdown=shutdown),
         "rclpy.node": mod("rclpy.node", Node=Node), "rclpy.qos": mod("rclpy.qos", qos_profile_sensor_data=None),
         "rclpy.signals": mod("rclpy.signals", SignalHandlerOptions=types.SimpleNamespace(NO="NO")),
         "geometry_msgs": mod("geometry_msgs"), "geometry_msgs.msg": mod("geometry_msgs.msg", Pose2D=object, Twist=Twist)}
    return m, W


def test_sigterm_in_the_command_line_tool_zeros_result_and_log(monkeypatch, tmp_path, capsys):
    handlers = {}
    monkeypatch.setattr(PB.signal, "signal", lambda s, h: handlers.__setitem__(s, h))   # nothing is really installed
    monkeypatch.setattr(PB, "busy_programs", lambda table, mine: [])
    def on_spin(W):
        pose = types.SimpleNamespace(x=-0.6, y=-0.4, theta=0.0); ball = types.SimpleNamespace(x=0.0, y=0.0, theta=0.0)
        W.subs["/vision/poses/blue/robot0"](pose); W.subs["/vision/poses/ball"](ball)
        if W.k == 40: handlers[signal.SIGTERM](signal.SIGTERM, None)       # as the supervisor's SIGTERM would
    mods, W = ros_stubs(on_spin)
    for k, v in mods.items(): monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(sys, "argv", ["place_ball.py", "blue", "0.6", "0.3", "--out", str(tmp_path)])
    PB.main()
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert W.init == dict(signal_handler_options="NO") and set(handlers) == {signal.SIGINT, signal.SIGTERM}
    assert res["status"] == "failed: stopped by SIGTERM" and not res["ok"] and os.path.isfile(res["log"])
    tail = W.sent[-15:]; assert len(W.sent) >= 15 and all(c == (0.0, 0.0, 0.0) for c in tail)
