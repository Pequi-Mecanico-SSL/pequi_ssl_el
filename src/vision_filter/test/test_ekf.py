"""EKF checks against synthetic ground truth (no ROS, no simulator).

Run from the package directory:  python3 -m pytest test -v
"""
import math

import numpy as np
import pytest

from vision_filter.ekf import RobotEKF, wrap

VISION_HZ = 60.0
SIGMA_XY = 0.005
SIGMA_TH = 0.03


def body_velocity(t):
    """Smooth figure-eight-ish body velocity profile (vx, vy, omega)."""
    return (0.8 * math.sin(0.7 * t) + 0.4,
            0.5 * math.cos(1.3 * t),
            1.5 * math.sin(0.9 * t))


def ground_truth(duration, velocity=body_velocity, dt=0.001, start=(0.0, 0.0, 0.0)):
    """Integrate the true trajectory at 1 kHz, sampled at the vision rate."""
    x, y, th = start
    samples = []
    next_sample = 0.0
    t = 0.0
    while t <= duration + 1e-9:
        vx, vy, w = velocity(t)
        if t >= next_sample - 1e-9:
            samples.append((t, np.array([x, y, th, vx, vy, w])))
            next_sample += 1.0 / VISION_HZ
        c, s = math.cos(th), math.sin(th)
        x += (vx * c - vy * s) * dt
        y += (vx * s + vy * c) * dt
        th = wrap(th + w * dt)
        t += dt
    return samples


def noisy(truth, rng):
    return np.array([truth[0] + rng.normal(0, SIGMA_XY),
                     truth[1] + rng.normal(0, SIGMA_XY),
                     wrap(truth[2] + rng.normal(0, SIGMA_TH))])


def make_filter(**kw):
    return RobotEKF(sigma_xy=SIGMA_XY, sigma_theta=SIGMA_TH, **kw)


def run(ekf, samples, rng, settle=1.0):
    """Feed noisy vision; return raw and filtered errors after settling."""
    raw_err, pos_err, th_err, vel_err = [], [], [], []
    for t, truth in samples:
        z = noisy(truth, rng)
        ekf.update(t, z)
        if t < settle:
            continue
        raw_err.append(np.hypot(*(z[:2] - truth[:2])))
        pos_err.append(np.hypot(*(ekf.x[:2] - truth[:2])))
        th_err.append(abs(wrap(ekf.x[2] - truth[2])))
        vel_err.append(np.linalg.norm(ekf.x[3:5] - truth[3:5]))
    rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
    return rms(raw_err), rms(pos_err), rms(th_err), rms(vel_err)


def test_smooths_position_and_estimates_velocity():
    rng = np.random.default_rng(0)
    raw, pos, th, vel = run(make_filter(), ground_truth(20.0), rng)
    print(f'raw pos {raw*1000:.1f} mm, filtered {pos*1000:.1f} mm, '
          f'theta {math.degrees(th):.2f} deg, velocity {vel:.3f} m/s')
    assert pos < 0.75 * raw
    assert th < SIGMA_TH
    assert vel < 0.15


def test_rejects_single_outlier():
    rng = np.random.default_rng(1)
    ekf = make_filter()
    samples = ground_truth(3.0)
    for k, (t, truth) in enumerate(samples):
        z = noisy(truth, rng)
        if k == 120:
            z[0] += 1.0  # ghost detection 1 m away
            assert ekf.update(t, z) == 'rejected'
        else:
            ekf.update(t, z)
        assert np.hypot(*(ekf.x[:2] - truth[:2])) < 0.05
    assert ekf.n_rejected == 1 and ekf.n_resets == 0


def test_snaps_to_teleported_robot():
    rng = np.random.default_rng(2)
    ekf = make_filter()
    samples = ground_truth(4.0)
    jump = np.array([1.0, -0.5, 0.0, 0, 0, 0])
    for k, (t, truth) in enumerate(samples):
        if k >= 120:
            truth = truth + jump  # robot picked up and put down elsewhere
        ekf.update(t, noisy(truth, rng))
        if k == 120 + ekf.max_rejects + 30:
            assert np.hypot(*(ekf.x[:2] - truth[:2])) < 0.05
    assert ekf.n_resets == 1


def test_coasts_through_dropout():
    rng = np.random.default_rng(3)
    ekf = make_filter()
    samples = ground_truth(4.0)
    last_seen = None
    for t, truth in samples:
        if t >= 3.0:
            break  # vision lost for the next 150 ms
        ekf.update(t, noisy(truth, rng))
        last_seen = truth
    t_end, truth_end = next((t, s) for t, s in samples if t >= 3.15)
    x, _ = ekf.predict(t_end)
    predicted_err = np.hypot(*(x[:2] - truth_end[:2]))
    stale_err = np.hypot(*(last_seen[:2] - truth_end[:2]))
    print(f'after 150 ms gap: predicted {predicted_err*1000:.1f} mm, '
          f'last raw pose {stale_err*1000:.1f} mm off')
    assert predicted_err < 0.06
    assert predicted_err < 0.25 * stale_err


def test_out_of_order_matches_in_order():
    samples = ground_truth(5.0)
    measurements = [(t, noisy(s, np.random.default_rng(k)))
                    for k, (t, s) in enumerate(samples)]
    swapped = list(measurements)
    for k in range(1, len(swapped) - 1, 3):  # every third pair arrives swapped
        swapped[k], swapped[k + 1] = swapped[k + 1], swapped[k]

    a, b = make_filter(), make_filter()
    for t, z in measurements:
        a.update(t, z)
    for t, z in swapped:
        b.update(t, z)
    assert b.n_dropped_late == 0
    assert np.allclose(a.x, b.x, atol=1e-6)


def test_heading_wraps_while_spinning():
    rng = np.random.default_rng(4)
    spin = lambda t: (0.3, 0.0, 6.0)  # ~1 turn/s, crosses +-pi constantly
    ekf = make_filter()
    omegas = []
    for t, truth in ground_truth(6.0, spin):
        ekf.update(t, noisy(truth, rng))
        if t > 1.0:
            omegas.append(ekf.x[5])
            assert abs(wrap(ekf.x[2] - truth[2])) < 3 * SIGMA_TH
    assert abs(np.mean(omegas) - 6.0) < 0.1
    assert np.std(omegas) < 0.6
    assert ekf.n_resets == 0 and ekf.n_rejected == 0


def test_cmd_vel_speeds_up_velocity_tracking():
    """After a velocity step, a filter told the command converges faster."""
    lag = 0.05  # robot follows the command with a short lag

    def step(t):
        return (1.5, 0.0, 0.0) if t >= 1.0 else (0.0, 0.0, 0.0)

    def robot(t):
        vx = 1.5 * (1.0 - math.exp(-(t - 1.0) / lag)) if t >= 1.0 else 0.0
        return (vx, 0.0, 0.0)

    samples = ground_truth(1.6, robot)

    def vx_error_at(ekf, t_check):
        rng = np.random.default_rng(5)
        for t, truth in samples:
            ekf.update(t, noisy(truth, rng))
            if t >= t_check:
                return abs(ekf.x[3] - truth[3])

    blind = make_filter()
    told = make_filter(cmd_tau=lag)
    told.add_command(0.0, 0.0, 0.0, 0.0)
    for k in range(0, 160):
        t = k * 0.01
        told.add_command(t, *step(t))
    e_blind = vx_error_at(blind, 1.1)
    e_told = vx_error_at(told, 1.1)
    print(f'vx error 100 ms after step: blind {e_blind:.3f}, with cmd_vel {e_told:.3f}')
    assert e_told < 0.5 * e_blind


@pytest.mark.parametrize('seed', range(5))
def test_stays_consistent(seed):
    """Normalized error stays near its expected value (filter not overconfident)."""
    rng = np.random.default_rng(100 + seed)
    ekf = make_filter()
    nees = []
    for t, truth in ground_truth(10.0):
        ekf.update(t, noisy(truth, rng))
        if t > 1.0:
            e = ekf.x[:3] - truth[:3]
            e[2] = wrap(e[2])
            nees.append(e @ np.linalg.inv(ekf.P[:3, :3]) @ e)
    # expected 3 for a consistent filter; allow a generous band
    assert 0.5 < np.mean(nees) < 9.0
