"""Gyro + wheel fusion checks against a synthetic robot (no ROS, no hardware).

Run from the package directory:  python3 -m pytest test/test_odom_ekf.py -v
"""
import math

import numpy as np
import pytest

from robot_control.odom_demo import metrics, run
from robot_control.odom_ekf import OdomEKF
from robot_control.sim_sensors import SIGMA_WHEEL, SLIPS, STALE, STUCK, SyntheticRun


@pytest.fixture(scope='module', params=[0, 1, 2])
def result(request):
    return run(seed=request.param)


def test_tracks_gyro_bias_while_driving(result):
    before_stuck = result['t'] < STUCK[0]
    err = abs(result['bias'][before_stuck][-1] - result['true_bias'][before_stuck][-1])
    assert err < 0.005


def test_heading_drifts_far_less_than_current_method(result):
    m = metrics(result)
    current, fused = m['heading error just before stuck (deg)']
    assert fused < 5.0
    assert fused < 0.25 * current


def test_omega_better_than_current_method(result):
    current, fused = metrics(result)['omega RMS (rad/s)']
    assert fused < 0.7 * current


def test_slipping_wheel_is_caught(result):
    wheels_only, fused = metrics(result)['linear velocity RMS, one wheel slipping (m/s)']
    assert fused < 0.6 * wheels_only
    wheel, start, end, _extra = SLIPS[0]
    assert result['ekf'].slips[wheel] >= 0.8 * (end - start) * 50


def test_normal_driving_not_worse_than_wheels(result):
    wheels_only, fused = metrics(result)['linear velocity RMS, normal (m/s)']
    assert fused < 1.1 * wheels_only


def test_stale_wheel_uses_remaining_three():
    sim = SyntheticRun(seed=0)
    ekf = OdomEKF(sim.J, sigma_wheel=SIGMA_WHEEL)
    wheel, start, end = STALE[0]
    used_during = []
    for kind, t, z in sim.events():
        if kind == 'imu':
            ekf.update_gyro(t, z)
        else:
            used = ekf.update_wheels(t, z)
            if start <= t < end:
                used_during.append(used)
    assert used_during and all(len(u) == 3 and wheel not in u for u in used_during)


def test_recovers_from_wrong_velocity():
    """A confidently wrong velocity is pulled back by the wheels."""
    sim = SyntheticRun(seed=0, duration=14.0)
    ekf = OdomEKF(sim.J, sigma_wheel=SIGMA_WHEEL)
    corrupted = False
    for kind, t, z in sim.events():
        if kind == 'imu':
            ekf.update_gyro(t, z)
            continue
        if not corrupted and t >= 11.0:
            ekf.x[3] += 2.0                       # 2 m/s off
            ekf.P[3, 3] = ekf.P[4, 4] = 1e-6      # and sure of it
            corrupted = True
        ekf.update_wheels(t, z)
    k = sim.index(14.0)
    assert abs(ekf.x[3] - sim.vel[k, 0]) < 0.05


def test_stationary_estimates_zero():
    sim = SyntheticRun(seed=3, duration=9.0)  # power-on idle only
    ekf = OdomEKF(sim.J, sigma_wheel=SIGMA_WHEEL)
    for kind, t, z in sim.events():
        (ekf.update_gyro if kind == 'imu' else ekf.update_wheels)(t, z)
    assert np.all(np.abs(ekf.x[3:6]) < 0.02)
    assert math.hypot(ekf.x[0], ekf.x[1]) < 0.02
