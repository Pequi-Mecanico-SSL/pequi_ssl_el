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
    for name in ("ids", "keeper", "domains", "cmd_topics", "speed_caps", "bench", "gone_after", "off_margin", "live_file", "stack_topic"):
        assert name in declared
    assert "/robot{vid}'" in src and "_robot_count:=16" in src


@pytest.mark.parametrize("bad", ["vfx", "vfy", "w", "th"])
def test_a_non_finite_policy_answer_gives_zero_not_a_growing_command(bad):
    r = D.Robot(1)
    for k in range(30):
        r.prev_vf, r.last_vf = r.last_vf, (0.0, 0.0)
        a = dict(vfx=0.5, vfy=0.2, w=1.0, th=0.3)
        if k >= 10:
            a[bad] = float("nan") if k % 2 else float("inf")
        c = D.body_command(r, a["vfx"], a["vfy"], a["w"], a["th"], 1.0, 1.5, 4.0, 0.1, (1.7, 1.5), 1 / 30)
        if k >= 10:
            assert c == (0.0, 0.0, 0.0) and r.last_vf == (0.0, 0.0) and r.last_w == 0.0
        else:
            assert all(math.isfinite(v) for v in c) and c != (0.0, 0.0, 0.0)


def test_live_roles_survive_a_restart_of_the_same_set_up(tmp_path):
    """the keeper and bench set live are what a node with the same team and ids starts with (the page's restart)"""
    f = str(tmp_path / "duo_live.json")
    args = ([1, 3], [77, 78], ["/cmd_vel", "/cmd_vel"], 77)
    assert D.load_live(f, "yellow", *args) is None                       # no file: the start-up roles
    D.save_live(f, "yellow", [1, 3], 3, {1})
    assert D.load_live(f, "yellow", *args) == (3, {1})
    assert D.load_live(f, "blue", *args) is None                         # another set-up ignores it
    assert D.load_live(f, "yellow", [1, 4], [77, 78], ["/cmd_vel", "/cmd_vel"], 77) is None
    assert D.load_live("", "yellow", *args) is None
    D.save_live("", "yellow", [1, 3], 3, set())                           # '' = not kept: nothing written, no error
    open(f, "w").write('{"team": "yellow", "ids": [1, 3], "keeper": 7, "bench": []}')
    assert D.load_live(f, "yellow", *args) is None                       # a keeper that is not one of ids
    open(f, "w").write('{"team": "yel')
    assert D.load_live(f, "yellow", *args) is None                       # half a file


def test_two_stacks_on_one_domain_block_its_robots_none_only_warns():
    blocked, txt = D.stack_problems({77: 2, 78: 0})
    assert blocked == {77} and "2 robot stacks on domain 77" in txt and "no robot stack heard on domain 78" in txt
    assert D.stack_problems({77: 1, 78: 1}) == (set(), "")
    assert D.stack_problems({77: 0, 78: 1})[0] == set()


def test_run_duo_start_clears_the_live_roles_and_passes_the_file():
    src = open(os.path.join(REPO, "tools", "solo", "run_duo.sh")).read()
    assert "rm -f $D/solo.cmd $D/duo_live.json" in src and "-p live_file:=$D/duo_live.json" in src
    assert 'duo_att) KEEP="vo_m,vb_m,om"' in src and "--bot $BOT --keep" in src


@pytest.mark.parametrize("rung,want", [("1", ["SOLO_BOT=duo_att", "run_duo.sh start yellow 1 3 false", '"a_nf": 2', "speed_cap:=1.5"]),
                                       ("2", ["SOLO_BOT=duo_nk", "run_duo.sh start yellow 1 3 false", '"lu_on": 1', '"nk_oa": 0.15']),
                                       ("3", ["SOLO_ROBOT_INDEX=1", "SOLO_OWN_COUNT=16", "run_real.sh start yellow false", '"lu_on": 1',
                                              "recover:=true"])])
def test_run_rung_picks_the_set_up_by_one_number(rung, want):
    import subprocess
    out = subprocess.run(["bash", os.path.join(REPO, "tools", "solo", "run_rung.sh"), "show", rung, "yellow", "1", "3"],
                         capture_output=True, text=True, timeout=10).stdout
    for w in want:
        assert w in out, (w, out)
    assert ("duo_att" in out) == (rung == "1")


def test_profiles_are_the_field_profile_and_the_duo_att_pick():
    import json
    prof = os.path.join(REPO, "tools", "solo", "profiles")
    field = json.load(open(os.path.join(prof, "field.json")))
    assert field == {"lu_on": 1, "lu_keeper": 1, "nk_rd": 0, "nk_aim": 0, "nk_m": 0.3, "nk_still": 0.5, "nk_od": 0.3, "nk_ko_shot": 2.5,
                     "nk_ko_rel": 1.2, "sp_wait": 2, "nk_sp_lo": -9, "nk_post": 0.28, "nk_oa": 0.15, "nk_da_run": 0.18, "nk_hold": 2.0}
    att = json.load(open(os.path.join(prof, "duo_att.json")))
    assert att["a_nf"] == 2 and att["a_nfv"] > 0 and "d_role" not in att
