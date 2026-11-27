"""Tests of the field reports (fspin_report.py, steps_report.py, rams_report.py) on synthetic characterize.py logs,
and (FSPIN_ROS=1 only) fspin_runs.sh end to end against tests/fake_field.py on a loopback-only ROS domain.
    uv run --with pytest pytest -q tools/solo/tests/test_reports.py"""
import math, os, subprocess, sys
import pytest
HERE = os.path.dirname(os.path.abspath(__file__))
SOLO = os.path.dirname(HERE)
HEAD = "t,kind,x,y,th,cmd_vx,cmd_vy,cmd_w\n"


def run(script, *args):
    return subprocess.run([sys.executable, "-I", os.path.join(SOLO, script), *map(str, args)], capture_output=True, text=True, check=True).stdout


def fspin_log(path, lat, w, true_lat=0.05, v=0.8, deg_per_s=170.0):
    """a straight path along +y whose direction is off by (true_lat - lat) x deg_per_s (deg) x sign(w)"""
    a = math.radians(90.0 + math.copysign(1.0, w) * (true_lat - lat) * deg_per_s)
    with open(path, "w") as f:
        f.write(HEAD)
        for k in range(60): f.write(f"{k / 60:.4f},r,-1.1,-1.0,0.0,0,0,0\n")
        for k in range(150):
            t = 1.0 + k / 60; s = v * k / 60
            f.write(f"{t:.4f},r,{-1.1 + s * math.cos(a):.4f},{-1.0 + s * math.sin(a):.4f},{w * (t - 1.0):.4f},0.8,0,{w}\n")
        f.write(f"{3.6:.4f},r,0,0,0,0,0,0\n")


def test_fspin_report_finds_the_zero_of_the_bend(tmp_path):
    for k, (lat, w) in enumerate([(0.1, 3.0), (0.04, -3.0), (0.0, 3.0), (0.1, -3.0)]):
        fspin_log(tmp_path / f"00000{k}_fspin_0.8_90_{w}_{lat}_-1.1_-1.0_2.2.csv", lat, w)
    out = run("fspin_report.py", tmp_path)
    line = [l for l in out.splitlines() if l.startswith("DECIDE")][0]
    assert "set latency to 0.05" in line, out


def step_log(path, v=1.5, decel=3.0):
    """forward along +x at v, command off at t = 2, constant deceleration"""
    with open(path, "w") as f:
        f.write(HEAD); x = 0.0; vv = 0.0; t = 0.0
        while t < 4.0:
            on = 0.5 <= t < 2.0
            vv = min(v, vv + 3.0 / 60) if on else max(0.0, vv - decel / 60)
            x += vv / 60; f.write(f"{t:.4f},r,{x:.4f},0.0,0.0,{v if on else 0.0},0,0\n"); t += 1 / 60


def test_steps_report_stop_distance_and_decision(tmp_path):
    step_log(tmp_path / "000001_step_fwd_1.5_-1.35_-1.15_90_1.6.csv")
    out = run("steps_report.py", tmp_path)
    stop = float(out.split(", stop ")[1].split()[0])
    assert 0.35 <= stop <= 0.38, out          # v^2 / 2a = 0.375 m from the cut; the first sample after it is 1/60 s on
    assert "DECIDE: keep speed_cap 1.5" in out
    out = run("steps_report.py", tmp_path, "--stop-ok", "0.3")
    assert "DECIDE: speed_cap 1.3" in out, out                      # 1.5 x sqrt(0.3 / stop) = 1.34-1.36


def ram_log(path, exit_deg, v=1.0):
    """robot along +x at v hits a ball at x = 0; the ball leaves at exit_deg off the heading at 1.2 v"""
    a = math.radians(exit_deg)
    with open(path, "w") as f:
        f.write(HEAD); t = 0.0
        while t < 3.0:
            xr = -0.6 + v * max(0.0, t - 0.5) if t < 1.1 else 0.0
            if t < 1.1: bx, by = (0.0, 0.0) if xr < -0.09 else (xr + 0.09, 0.0)
            else:
                s = 1.2 * v * (t - 1.1); bx, by = 0.0 + 0.09 + 0.0 + s * math.cos(a) + 0.0, s * math.sin(a)
            cmd = v if 0.5 <= t < 1.1 else 0.0
            f.write(f"{t:.4f},r,{xr:.4f},0.0,0.0,{cmd},0,0\n{t + 0.001:.4f},b,{bx:.4f},{by:.4f},0.0,{cmd},0,0\n"); t += 1 / 60


def test_rams_report_exit_angle_mean_and_sd(tmp_path):
    for k, e in enumerate((-2.0, 0.0, 2.0, 4.0)):
        ram_log(tmp_path / f"00000{k}_ram_1.0_0_0.6_0.2.csv", e)
    out = run("rams_report.py", tmp_path)
    line = [l for l in out.splitlines() if l.startswith("DECIDE")][0]
    assert "mean +1.0" in line and "sd 2.6" in line, out


@pytest.mark.skipif(os.environ.get("FSPIN_ROS") != "1", reason="needs ROS 2; FSPIN_ROS=1 runs it on a loopback-only domain")
def test_fspin_runs_against_the_fake_field(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("ROS_DISCOVERY_SERVER", "ROS_SUPER_CLIENT", "FASTRTPS_DEFAULT_PROFILES_FILE")}
    env.update(ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="231", FSPIN_DRY="1", SHORT="1", START="0.5,0.5,0")
    fake = subprocess.Popen([sys.executable, os.path.join(HERE, "fake_field.py"), "blue"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        out = subprocess.run([os.path.join(SOLO, "fspin_runs.sh"), "blue", str(tmp_path)], env=env, capture_output=True, text=True, timeout=120).stdout
    finally:
        fake.terminate(); fake.wait(5)
    line = [l for l in out.splitlines() if l.startswith("DECIDE")][0]
    z = float(line.split("set latency to ")[1].split()[0].rstrip(";"))
    assert 0.04 <= z <= 0.07, out                                    # the fake's straightest latency: VIS + NET = 0.055
