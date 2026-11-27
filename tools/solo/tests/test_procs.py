"""Tests of tools/solo/procs.py (the commanding programs, the strategy node's command line check): no process is
started or signalled.
    uv run --with pytest pytest -q tools/solo/tests/test_procs.py"""
import ast, importlib.util, os, re
import pytest
HERE = os.path.dirname(os.path.abspath(__file__))
SOLO = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SOLO))
spec = importlib.util.spec_from_file_location("procs_under_test", os.path.join(SOLO, "procs.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def test_node_params_match_the_nodes_declare_parameter_calls():
    src = open(os.path.join(REPO, "src", "solo_strategy", "solo_strategy", "solo_node.py")).read()
    names = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            f = n.func
            if (isinstance(f, ast.Name) and f.id == "d") or (isinstance(f, ast.Attribute) and f.attr == "declare_parameter"):
                names.add(n.args[0].value)
    assert "d = self.declare_parameter" in src and len(names) > 20
    assert names == set(P.NODE_PARAMS)


def test_every_velocity_publisher_here_is_a_commander():
    """every script in tools/solo that creates a Twist publisher is in COMMANDERS (hq.py is the field page itself)"""
    pubs = set()
    for f in sorted(os.listdir(SOLO)):
        if f.endswith(".py") and re.search(r"create_publisher\(\s*Twist\b", open(os.path.join(SOLO, f)).read()):
            pubs.add(f)
    assert {"goto.py", "nudge.py", "frame_check.py", "characterize.py", "series.py", "ram_series.py", "place_ball.py"} <= pubs
    assert pubs - {"hq.py"} <= set(P.COMMANDERS) and "solo_node" in P.COMMANDERS


def test_is_instance_skips_interpreter_options_and_ignores_mentions():
    yes = [["python3", "-u", "/x/goto.py", "blue", "1", "0"], ["python3", "-B", "-X", "dev", "goto.py"], ["python3", "-Xdev", "goto.py"],
           ["/usr/bin/python3.10", "-W", "ignore", "-uB", "tools/solo/goto.py"], ["python3", "--", "goto.py"], ["goto.py", "blue"],
           ["/usr/bin/python3", "/ws/lib/solo_strategy/solo_node", "--ros-args"]]
    no = [["python3", "-c", "import goto.py"], ["python3", "-cimport goto"], ["python3", "-uc", "x", "goto.py"], ["python3", "-m", "goto.py"],
          ["vim", "goto.py"], ["bash", "-c", "python3 goto.py"], ["python3", "tools/solo/hq.py", "goto.py"], ["python3"], [],
          ["less", "/tmp/goto.py.log"], ["python3", "-", "goto.py"]]
    for a in yes:
        n = "solo_node" if "solo_node" in " ".join(a) else "goto.py"
        assert P.is_instance(a, n), a
    for a in no:
        assert not P.is_instance(a, "goto.py"), a
    assert P.is_launcher(["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "run", "solo_strategy", "solo_node", "--ros-args"])
    assert not P.is_instance(["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "run", "solo_strategy", "solo_node"], "solo_node")


def test_running_excludes_the_given_pids():
    table = [(10, ["python3", "/x/goto.py", "blue", "0", "0"]), (11, ["python3", "/x/series.py", "blue", "p"]), (12, ["vim", "nudge.py"])]
    assert P.running(table=table, exclude=set()) == ["goto.py", "series.py"]
    assert P.running(table=table, exclude={10}) == ["series.py"]
    mine = P.ancestors(); assert os.getpid() in mine and os.getppid() in mine


@pytest.fixture
def files(tmp_path):
    (tmp_path / "bin").mkdir(); (tmp_path / "lib").mkdir()
    py = tmp_path / "bin" / "python3"; py.write_text("#!/bin/sh\n"); py.chmod(0o755)
    py310 = tmp_path / "bin" / "python3.10"; py310.write_text("#!/bin/sh\n"); py310.chmod(0o755)
    noexec = tmp_path / "bin" / "python3.11"; noexec.write_text("x"); noexec.chmod(0o644)
    node = tmp_path / "lib" / "solo_node"; node.write_text("x")
    other = tmp_path / "lib" / "other"; other.write_text("x")
    (tmp_path / "lib" / "dir").mkdir(); (tmp_path / "lib" / "dir" / "solo_node").mkdir()
    return dict(py=str(py), py310=str(py310), noexec=str(noexec), node=str(node), other=str(other), nodedir=str(tmp_path / "lib" / "dir" / "solo_node"))


def test_check_node_cmd_accepts_only_the_nodes_own_form(files):
    py, node = files["py"], files["node"]
    ok = [[py, node, "--ros-args"], [py, node, "--ros-args", "-p", "team:=blue", "-p", "robot_index:=0", "-p", "cmd_topic:=/cmd_vel"],
          [files["py310"], node, "--ros-args", "-p", "recover:=true", "-p", "recover_states:=[2, 3]"]]
    for a in ok:
        assert P.check_node_cmd(a) == "", a
    hostile = {
        "a -c payload in the script place": [py, "-c__import__('os').system('x')#/solo_node"],
        "a -c payload after the node": [py, node, "--ros-args", "-c", "x"],
        "a -c element": [py, "-c", node, "--ros-args"],
        "a -c parameter value element": [py, node, "--ros-args", "-p", "-cteam:=x"],
        "a relative interpreter": ["python3", node, "--ros-args"],
        "a relative interpreter path": ["bin/python3", node, "--ros-args"],
        "not python": ["/bin/sh", node, "--ros-args"],
        "python2": [os.path.join(os.path.dirname(py), "python2"), node, "--ros-args"],
        "a missing interpreter": [os.path.join(os.path.dirname(py), "python3.99"), node, "--ros-args"],
        "an interpreter that is not executable": [files["noexec"], node, "--ros-args"],
        "a script not named solo_node": [py, files["other"], "--ros-args"],
        "a relative node": [py, "lib/solo_node", "--ros-args"],
        "a missing node": [py, node + "_x", "--ros-args"],
        "a node that is a folder": [py, files["nodedir"], "--ros-args"],
        "no --ros-args": [py, node, "-p", "team:=blue"],
        "an unknown parameter": [py, node, "--ros-args", "-p", "team:=blue", "-p", "evil:=1"],
        "a remap": [py, node, "--ros-args", "-r", "__node:=x"],
        "a params file": [py, node, "--ros-args", "--params-file", "/tmp/p.yaml"],
        "a dangling -p": [py, node, "--ros-args", "-p"],
        "no :=": [py, node, "--ros-args", "-p", "team"],
        "a newline": [py, node, "--ros-args", "-p", "team:=blue\nx"],
        "a NUL": [py, node + "\0", "--ros-args"],
        "an option before the node": [py, "-u", node, "--ros-args"],
        "too short": [py, node],
        "not a list": " ".join([py, node, "--ros-args"]),
        "a dict": {"argv": [py, node]},
        "not strings": [py, node, "--ros-args", "-p", 1],
        "None": None,
    }
    for name, a in hostile.items():
        why = P.check_node_cmd(a)
        assert why, name


def test_run_child_forwards_a_stop_and_gives_the_child_its_grace():
    import signal
    class Clock:
        t = 0.0
        def __call__(self): return self.t
        def sleep(self, dt): self.t += dt
    class FakeChild:
        def __init__(self, clk, ends_after_signal):
            self.clk, self.after, self.sigs, self.rc, self.t_sig = clk, ends_after_signal, [], None, None
        def poll(self):
            if self.rc is None and self.t_sig is not None and self.after is not None and self.clk.t - self.t_sig >= self.after: self.rc = -2
            return self.rc
        def send_signal(self, s):
            self.sigs.append(s); self.t_sig = self.clk.t if self.t_sig is None else self.t_sig
            if s == signal.SIGKILL: self.rc = -9
        def communicate(self, timeout=None): return ('{"test": "spin", "interrupted": true}\n', None)
    for after, how, sigs in ((0.4, "stopped", [signal.SIGINT]), (None, "killed", [signal.SIGINT, signal.SIGKILL])):
        clk = Clock(); child = FakeChild(clk, after)
        rc, out, h = P.run_child(["x"], lambda: signal.SIGINT if clk.t >= 1.0 else None, popen=lambda *a, **k: child, clock=clk, sleep=clk.sleep)
        assert h == how and child.sigs == sigs and '"interrupted": true' in out
        if after is None: assert 2.5 <= clk.t <= 2.7                       # 1.5 s of grace, not 0.25 s


def _declared(path):
    names = set()
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            f = n.func
            if (isinstance(f, ast.Name) and f.id == "d") or (isinstance(f, ast.Attribute) and f.attr == "declare_parameter"):
                names.add(n.args[0].value)
    return names


def test_duo_params_match_the_duo_nodes_declare_parameter_calls():
    names = _declared(os.path.join(REPO, "src", "solo_strategy", "solo_strategy", "duo_node.py"))
    assert len(names) > 20 and names == set(P.DUO_PARAMS)
    assert "duo_node" in P.COMMANDERS and set(P.STRATEGY_NODES) == set(P.NODES)


def test_check_node_cmd_for_the_duo_node(tmp_path):
    (tmp_path / "bin").mkdir(); (tmp_path / "lib").mkdir()
    py = tmp_path / "bin" / "python3"; py.write_text("#!/bin/sh\n"); py.chmod(0o755)
    duo = tmp_path / "lib" / "duo_node"; duo.write_text("x")
    solo = tmp_path / "lib" / "solo_node"; solo.write_text("x")
    line = [str(py), str(duo), "--ros-args", "-p", "team:=yellow", "-p", "ids:=[1, 3]", "-p", "domains:=[77, 78]"]
    assert P.check_node_cmd(line, P.STRATEGY_NODES) == ""
    assert "not named" in P.check_node_cmd(line)                                   # the default stays solo_node only
    assert "unknown parameter" in P.check_node_cmd(line[:3] + ["-p", "robot_index:=0"], P.STRATEGY_NODES)   # solo_node's, not duo_node's
    assert "unknown parameter" in P.check_node_cmd([str(py), str(solo), "--ros-args", "-p", "ids:=[1]"], P.STRATEGY_NODES)
    assert P.is_launcher(["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "run", "solo_strategy", "duo_node"], P.STRATEGY_NODES)
    assert not P.is_launcher(["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "run", "solo_strategy", "duo_node"])
