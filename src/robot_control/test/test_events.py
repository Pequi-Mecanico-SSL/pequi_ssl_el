"""robot_events checks against a synthetic robot with known events (no ROS).

Run from the package directory:  python3 -m pytest test/test_events.py -v
"""
import numpy as np
import pytest

from robot_control.event_detector import EventDetector, Hysteresis
from robot_control.events_demo import detect, match
from robot_control.sim_events import EVENTS, SKID, EventRun

MAX_DELAY = {'impact': 0.03, 'skid': 0.1, 'stall': 0.3, 'push': 0.2}


@pytest.fixture(scope='module', params=range(10))
def result(request):
    return match(detect(seed=request.param))


def test_every_event_detected_in_time(result):
    rows, _false = result
    for name, start, _end, found, _ended in rows:
        assert found is not None, f'{name} at {start} missed'
        assert found['t'] - start <= MAX_DELAY[name], f'{name} late: {found["t"] - start:.3f}s'


def test_every_lasting_event_ends(result):
    rows, _false = result
    for name, start, end, _found, ended in rows:
        if start != end:
            assert ended is not None, f'{name} never ended'
            assert ended['t'] - end < 0.6


def test_no_false_alarms(result):
    _rows, false_alarms = result
    assert false_alarms == []


def test_impact_details(result):
    rows, _false = result
    wall, side = [r[3] for r in rows if r[0] == 'impact']
    assert wall['kind'] == 'linear' and wall['saturated']
    # wall straight ahead: robot thrown backwards
    assert abs(abs(wall['direction_deg']) - 180.0) < 20.0
    assert side['kind'] == 'rotational' and side['ang_accel'] > 200.0


def test_skid_names_the_right_wheel_pair(result):
    rows, _false = result
    skid = next(r[3] for r in rows if r[0] == 'skid')
    assert SKID[0] in skid['candidates']
    assert len(skid['candidates']) <= 2


def test_ambiguous_pairs_come_from_geometry():
    det = EventDetector(EventRun().J)
    assert det.candidates == [[0, 3], [1, 2], [1, 2], [0, 3]]


def test_hysteresis():
    h = Hysteresis(on_s=0.1, off_s=0.2)
    assert h.update(0.00, True) is None
    assert h.update(0.05, False) is None      # blip resets the timer
    assert h.update(0.10, True) is None
    assert h.update(0.20, True) == 'start'
    assert h.update(0.30, False) is None
    assert h.update(0.50, False) == 'end'


def test_scenario_is_what_the_tests_assume():
    assert [e[0] for e in EVENTS] == ['impact', 'stall', 'skid', 'impact', 'push']


def _detector():
    return EventDetector(EventRun().J)


def test_shove_while_standing_is_push_not_stall():
    """Commanded still, shoved sideways at 0.6 m/s with the motors at the
    current limit: a push, never a stall (found in the simulator)."""
    det = _detector()
    J = EventRun().J
    events = []
    for k in range(100):                  # 2 s at 50 Hz
        t = k * 0.02
        det.wheel_setpoints(t, np.zeros(4))
        v = np.array([0.0, 0.6, 0.0]) if t >= 1.0 else np.zeros(3)
        speeds = J @ v
        currents = np.full(4, 1.3) if t >= 1.0 else np.zeros(4)
        events += det.wheels(t, speeds, currents)
    kinds = [(e['event'], e.get('state')) for e in events]
    assert ('push', 'start') in kinds
    assert not any(e['event'] == 'stall' for e in events)


def test_spin_damping_does_not_end_a_push():
    """velocity_controller damps a shove-induced spin with small rotation
    setpoints; the push must continue."""
    det = _detector()
    J = EventRun().J
    events = []
    for k in range(150):
        t = k * 0.02
        spin_damping = J @ np.array([0.0, 0.0, -0.5]) if t >= 1.2 else np.zeros(4)
        det.wheel_setpoints(t, spin_damping)
        v = np.array([0.0, 0.4, 0.3]) if t >= 1.0 else np.zeros(3)
        events += det.wheels(t, J @ v, np.zeros(4))
    pushes = [e['state'] for e in events if e['event'] == 'push']
    assert pushes == ['start']


def test_blocked_robot_is_stall():
    """Commanded forward 1 m/s, wheels not turning, current saturated."""
    det = _detector()
    J = EventRun().J
    events = []
    for k in range(50):
        t = k * 0.02
        det.wheel_setpoints(t, J @ np.array([1.0, 0.0, 0.0]))
        events += det.wheels(t, np.zeros(4), np.full(4, 1.3))
    assert ('stall', 'start') in [(e['event'], e.get('state')) for e in events]
