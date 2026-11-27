"""Tests of a trained policy in policy_server.py (--ckpt). They need the strategy simulator and a checkpoint, and are
skipped without them (no ROS, no robot, no network; about three minutes):
    POLICY_LIVE=<strategy simulator checkout> POLICY_CKPT=<checkpoint .pkl> [POLICY_PYTHON=<its Python, default
    $POLICY_LIVE/.venv/bin/python>] [POLICY_ROBOT=three_wheel_fast] [SSL_ROBOTS=...] \\
        uv run --with pytest --with numpy pytest -q tools/solo/tests/test_policy_rl.py
SSL_ROBOTS defaults to robots/three_wheel_fast.json:robots/three_wheel_fast_5a.json of the simulator checkout."""
import json, math, os, socket, subprocess
import pytest
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SIM, CKPT = os.environ.get("POLICY_LIVE"), os.environ.get("POLICY_CKPT")
ROBOT = os.environ.get("POLICY_ROBOT", "three_wheel_fast")
needs = pytest.mark.skipif(not (SIM and CKPT), reason="POLICY_LIVE=<strategy simulator checkout> and POLICY_CKPT=<checkpoint> not set")


def sim_env():
    return dict(os.environ, PYTHONPATH=f"{SIM}/src:{SIM}/ssl-rules/src", JAX_PLATFORMS="cpu",
                SSL_ROBOTS=os.environ.get("SSL_ROBOTS", "robots/three_wheel_fast.json:robots/three_wheel_fast_5a.json"))


def python():
    return os.environ.get("POLICY_PYTHON", f"{SIM}/.venv/bin/python")


@needs
def test_the_server_shows_the_policy_what_the_simulator_showed_it():
    r = subprocess.run([python(), os.path.join(HERE, "rl_replay.py"), os.path.abspath(CKPT), "--robot", ROBOT, "--seconds", "30", "--port", "47172"],
                       cwd=SIM, env=sim_env(), capture_output=True, text=True)
    print(r.stdout[-600:])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


@needs
def test_play_stop_prepare_halt_and_the_live_parameters():
    port = 47173
    p = subprocess.Popen([python(), os.path.join(REPO, "src/solo_strategy/solo_strategy/policy_server.py"), "--ckpt", os.path.abspath(CKPT),
                          "--robot", ROBOT, "--port", str(port)], cwd=SIM, env=sim_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in p.stdout:
            if "listening" in line: break
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(10)
        def ask(q):
            s.sendto(json.dumps(q).encode(), ("127.0.0.1", port)); return json.loads(s.recv(65535))
        Q = dict(t=0.0, state=3, ours=False, cmd_t=0.0, still_since=0.0, last_team=0, kick_pos=[0, 0],
                 us=[-1.5, 0.0, 0.0, 0, 0, 1], opp=[[1.0, 0.5, 3.1, 0.0, 0.0, 1]], ball=[-0.5, 0.2, 0.0, 0.0])
        c0 = ask({"cfg": "get"})
        assert c0["bot"].startswith("rl:") and set(c0["keys"]) == {"rl_touch", "rl_rel", "rl_clock"}
        # play: finite, inside the caps, and the speed grows no faster than the robot's command ramp allows
        sp = []
        for k in range(30):
            r = ask({**Q, "t": k / 30, "ball": [-0.5 + 1e-4 * k, 0.2, 0.0, 0.0]})
            assert r["mode"] == "play" and all(math.isfinite(x) for x in r["v"] + [r["w"], r["drib"]])
            assert math.hypot(*r["v"]) <= 1.5 + 1e-3 and abs(r["w"]) <= 6.0 + 1e-3
            sp.append(math.hypot(*r["v"]))
        assert max(b - a for a, b in zip([0.0] + sp, sp)) < 4.0 / 30 and max(sp) > 0.2
        # a stop: toward the restart pose, slower than the stop limit
        for k in range(45):
            r = ask({**Q, "t": 1 + k / 30, "state": 2})
        assert r["mode"] == "stop" and math.hypot(*r["v"]) <= 0.6 + 1e-3
        # PREPARE_KICKOFF for us: a stop pose, not the policy; after NORMAL_START the policy
        r = ask({**Q, "t": 3.0, "state": 5, "ours": True, "prep": True, "ball": [0.0, 0.0, 0.0, 0.0]})
        assert r["mode"] == "stop"
        assert ask({**Q, "t": 3.1, "state": 5, "ours": True, "ball": [0.0, 0.0, 0.0, 0.0]})["mode"] == "play"
        # a halt: zero at once
        r = ask({**Q, "t": 3.2, "state": 1})
        assert r["mode"] == "halt" and r["v"] == [0.0, 0.0] and r["w"] == 0.0
        # the carried distance: the ball at the robot's front, both moving 0.3 m
        for k in range(40):
            x = -1.0 + 0.3 * k / 40
            r = ask({**Q, "t": 4 + k / 30, "us": [x, 0.0, 0.0, 0.3, 0.0, 1], "ball": [x + 0.10, 1e-5 * k, 0.3, 0.0]})
        assert 0.2 < r["drib"] < 0.45
        r2 = ask({"cfg": "set", "set": {"rl_clock": 1.5}})
        assert r2.get("error") is None and r2["now"] == {"rl_clock": 1.5} and r2["compiles"] == c0["compiles"]
        x = -0.7
        r3 = ask({**Q, "t": 5.4, "us": [x, 0.0, 0.0, 0.3, 0.0, 1], "ball": [x + 0.10, 1e-3, 0.3, 0.0]})
        assert r3["drib"] > 1.3 * r["drib"]
        assert "error" in ask({"cfg": "set", "set": {"rl_rel": 0.0}}) and "error" in ask({"cfg": "set", "set": {"nk_hold": 1.0}})
        assert ask({**Q, "us": [-1.5, 0.0, 0.0, 0, 0, 0]})["mode"] == "robot not seen"
    finally:
        p.terminate()
