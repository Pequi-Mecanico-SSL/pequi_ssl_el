"""Offline demo: robot_events on a synthetic robot with known events
(no ROS, no simulator, no hardware).

    ros2 run robot_control events_demo [--seed 0]

Feeds sim_events.py through the odometry EKF and the event detector, then
prints each true event next to what was detected, and any false alarm.
"""
import argparse

from .event_detector import EventDetector
from .odom_ekf import OdomEKF
from .sim_events import (CURRENT_LIMIT, EVENTS, SIGMA_GYRO, SIGMA_WHEEL, SKID, EventRun)


def detect(seed=0, **detector_kwargs):
    run = EventRun(seed=seed)
    ekf = OdomEKF(run.J, sigma_gyro=SIGMA_GYRO, sigma_wheel=SIGMA_WHEEL)
    det = EventDetector(run.J, current_limit=CURRENT_LIMIT, sigma_wheel=SIGMA_WHEEL,
                        **detector_kwargs)
    events = []
    for kind, t, data in run.stream():
        if kind == 'imu':
            gyro, ax, ay = data
            ekf.update_gyro(t, gyro)
            events += det.imu(t, gyro, (ax, ay, 9.81))
        else:
            speeds, currents, setpoints = data
            det.wheel_setpoints(t, setpoints)
            events += det.wheels(t, speeds, currents)
            ekf.update_wheels(t, speeds)
            events += det.residuals(t, ekf.last_residuals)
    return events


def match(events, tolerance=0.35, end_tolerance=0.6):
    """Pair each true event with detections; return (rows, false_alarms)."""
    used = set()
    rows = []
    for name, start, end in EVENTS:
        found = [k for k, e in enumerate(events)
                 if e['event'] == name and k not in used
                 and (name != 'skid' or SKID[0] in e['candidates'])
                 and e.get('state', 'start') == 'start'
                 and start - 0.05 <= e['t'] <= start + tolerance]
        ends = [k for k, e in enumerate(events)
                if e['event'] == name and e.get('state') == 'end'
                and (name != 'skid' or SKID[0] in e['candidates'])
                and end - 0.05 <= e['t'] <= end + end_tolerance]
        used.update(found[:1] + ends[:1])
        rows.append((name, start, end,
                     events[found[0]] if found else None,
                     events[ends[0]] if ends else None))
    false_alarms = [e for k, e in enumerate(events) if k not in used]
    return rows, false_alarms


def fmt(e):
    extra = {k: v for k, v in e.items() if k not in ('event', 'state', 't')}
    return ' '.join(f'{k}={v}' for k, v in extra.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rows, false_alarms = match(detect(args.seed))
    print('\n=== robot_events on a synthetic robot (30 s) ===')
    print(f'{"true event":10s} {"at":>6s}   {"detected":>9s} {"delay":>7s}   {"ended":>6s}   details')
    for name, start, end, found, ended in rows:
        if found is None:
            print(f'{name:10s} {start:6.2f}   {"MISSED":>9s}')
            continue
        end_txt = f'{ended["t"]:6.2f}' if ended else ('   n/a' if start == end else ' never')
        print(f'{name:10s} {start:6.2f}   {found["t"]:9.2f} {1000 * (found["t"] - start):5.0f}ms'
              f'   {end_txt}   {fmt(found)}')
    print(f'false alarms: {len(false_alarms)}')
    for e in false_alarms:
        print(f'  {e["t"]:6.2f} {e["event"]} {e.get("state", "")} {fmt(e)}')


if __name__ == '__main__':
    main()
