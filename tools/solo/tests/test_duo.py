"""Tests of src/solo_strategy/solo_strategy/duo.py (the ROS-free part of the two-robot node): what each robot counts as,
the set-up checks, and that one robot's command is the same arithmetic as solo_node's. No ROS, no network.
    uv run --with pytest pytest -q tools/solo/tests/test_duo.py"""
import ast, math, os, sys
import pytest
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "src", "solo_strategy"))
from solo_strategy import duo as D                                   # noqa: E402

LIM = D.Limits()


def seen(r, t0, t1, x=0.0, y=0.0, th=0.0, dt=1 / 60):
    t = t0
    while t <= t1 + 1e-9:
        r.observe(t, x, y, th); t += dt


def test_a_robot_seen_now_plays_with_its_vision_pose():
    r = D.Robot(3); seen(r, 0.0, 1.0, x=0.5, y=-0.2, th=0.3)
    v = r.view(1.0, LIM)
    assert v.status == D.PLAY and v.drives and v.present and v.scale == 1.0
    assert v.pose[:3] == (0.5, -0.2, 0.3)


def test_blind_slow_down_then_zero_then_absent():
    r = D.Robot(3); seen(r, 0.0, 1.0)
    assert r.view(1.2, LIM).status == D.PLAY and r.view(1.2, LIM).scale == 1.0     # 0.2 s: dead reckoning, full caps
    v = r.view(1.4, LIM)                                                           # 0.4 s: between blind_slow and blind_limit
    assert v.status == D.PLAY and LIM.blind_floor < v.scale < 1.0
    v = r.view(1.6, LIM)                                                           # 0.6 s: no command, still on the field
    assert v.status == D.UNSEEN and not v.drives and v.present and v.pose[3:] == (0.0, 0.0)
    v = r.view(3.1, LIM)                                                           # 2.1 s: off the field for the policy
    assert v.status == D.GONE and not v.present and not v.drives


def test_without_prediction_pose_timeout_stops_it():
    lim = D.Limits(predict=False)
    r = D.Robot(1); seen(r, 0.0, 1.0)
    assert r.view(1.15, lim).status == D.PLAY
    assert r.view(1.25, lim).status == D.UNSEEN


def test_never_seen_is_gone():
    assert D.Robot(1).view(5.0, LIM).status == D.GONE


def test_benched_wins_over_everything():
    r = D.Robot(1); seen(r, 0.0, 1.0)
    v = r.view(1.0, LIM, benched=True)
    assert v.status == D.BENCHED and not v.present and not v.drives


@pytest.mark.parametrize("x,y,off", [(2.6, 0.0, False), (2.75, 0.0, True), (0.0, -1.9, False), (0.0, 2.0, True), (-2.8, 1.0, True)])
def test_a_robot_carried_beyond_the_lines_is_off_the_field(x, y, off):
    r = D.Robot(1); seen(r, 0.0, 0.5, x=x, y=y)
    assert (r.view(0.5, LIM).status == D.OFF_FIELD) == off


def test_body_command_matches_solo_node_arithmetic():
    """the same numbers as solo_node's play path: cap, latency turn by last_w, ramp from the last field velocity"""
    r = D.Robot(1); r.last_w = 1.0; r.prev_vf = (0.3, 0.0)
    th, lat, dt, ramps = 0.4, 0.1, 1 / 30, (1.7, 1.5)
    bx, by, w = D.body_command(r, 2.0, 0.0, 5.0, th, 1.0, 1.5, 4.0, lat, ramps, dt)
    vfx, vfy = 1.5, 0.0                                        # capped at 1.5 m/s
    thp = th + 1.0 * lat
    c, s = math.cos(thp), math.sin(thp)
    tx, ty = c * vfx + s * vfy, -s * vfx + c * vfy
    px, py = c * 0.3, -s * 0.3
    ex = D.ramp(px, tx, ramps[0], dt); ey = D.ramp(py, ty, ramps[1], dt)
    assert (bx, by) == pytest.approx((ex, ey)) and w == 4.0 and r.last_w == 4.0
    assert r.last_vf == pytest.approx((c * ex - s * ey, s * ex + c * ey))


def test_body_command_scale_lowers_both_caps():
    r = D.Robot(1)
    bx, by, w = D.body_command(r, 1.0, 0.0, 4.0, 0.0, 0.5, 1.0, 4.0, 0.0, (0.0, 0.0), 1 / 30)
    assert math.hypot(bx, by) == pytest.approx(0.5) and w == pytest.approx(2.0)


def test_ramp_is_solo_nodes_ramp():
    src = open(os.path.join(REPO, "src", "solo_strategy", "solo_strategy", "solo_node.py")).read()
    tree = ast.parse(src)
    f = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_ramp")
    body = ast.get_source_segment(src, f)
    assert "abs(tgt) <= abs(prev) and prev * tgt >= 0.0" in body and "prev = 0.0" in body
    assert D.ramp(0.0, 1.0, 1.5, 0.1) == pytest.approx(0.15)
    assert D.ramp(0.5, 0.2, 1.5, 0.1) == 0.2                 # slowing: at once
    assert D.ramp(0.5, -1.0, 1.5, 0.1) == pytest.approx(-0.15)  # reversing: from zero
    assert D.ramp(0.0, 1.0, 0.0, 0.1) == 1.0                 # no ramp


def test_slots_put_the_keeper_first():
    assert D.slots([1, 3], 1) == [1, 3] and D.slots([1, 3], 3) == [3, 1] and D.slots([4], 4) == [4]


@pytest.mark.parametrize("ids,domains,topics,keeper,bench,ok", [
    ([1, 3], [77, 78], ["/cmd_vel", "/cmd_vel"], 1, [], True),
    ([1, 3], [-1, 78], ["/cmd_vel", "/cmd_vel"], 3, [3], True),
    ([1, 3], [77, 77], ["/cmd_vel", "/cmd_vel"], 1, [], False),       # both robots would hear the same commands
    ([1, 3], [-1, 77], ["/cmd_vel", "/cmd_vel"], 1, [], False),       # -1 is this process's domain (77 here)
    ([1, 3], [77, 77], ["/a/cmd_vel", "/b/cmd_vel"], 1, [], True),    # one domain, a topic each (namespaced robots)
    ([1, 1], [77, 78], ["/cmd_vel", "/cmd_vel"], 1, [], False),
    ([1, 3], [77], ["/cmd_vel", "/cmd_vel"], 1, [], False),
    ([1, 3], [77, 78], ["/cmd_vel"], 1, [], False),
    ([1, 3], [77, 300], ["/cmd_vel", "/cmd_vel"], 1, [], False),
    ([1, 3], [77, 78], ["/cmd_vel", "/cmd_vel"], 2, [], False),
    ([1, 3], [77, 78], ["/cmd_vel", "/cmd_vel"], 1, [2], False),
    ([1, 16], [77, 78], ["/cmd_vel", "/cmd_vel"], 1, [], False),
    ([2], [78], ["/cmd_vel"], 2, [], True),
    ([], [], [], 0, [], False),
])
def test_check_ids(ids, domains, topics, keeper, bench, ok):
    why = D.check_ids(ids, domains, topics, keeper, bench, own_domain=77)
    assert (why == "") == ok, why


def test_node_declares_what_it_documents():
    """every parameter named in duo_node's header is declared, and the node reads our robots by vision id"""
    src = open(os.path.join(REPO, "src", "solo_strategy", "solo_strategy", "duo_node.py")).read()
    declared = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "d" and n.args and isinstance(n.args[0], ast.Constant):
            declared.add(n.args[0].value)
    for name in ("ids", "keeper", "domains", "cmd_topics", "speed_caps", "bench", "gone_after", "off_margin"):
        assert name in declared
    assert "/robot{vid}'" in src and "_robot_count:=16" in src
