#!/usr/bin/env python3
"""Write the series.py plans of the left-half sessions (tools/solo/sessions/plans/left_*.jsonl): every test inside the
home area x -2.2..-0.10, y -1.45..1.45 (field frame, the left half of the field), 0.25 m inside the lines except the
coverage pass. Lanes and spots were picked on the camera coverage map (validate.py checks them against it):
  lane Y1  x = -1.35 along y, y -1.15..+1.15: ramped runs at the 1.5 m/s cap (0.66 m to reach 1.5 at 1.7 m/s2, about
           0.9 m held, about 0.6 m to stop inside the box); forward and sideways are the same lane with another heading
  lane Y2  x = -0.85 along y: the command jumps (no ramp)
  spot S1  (-0.95, +0.10): spins and the gyro-scale turns, in a 0.8 m square
    python3 tools/solo/sessions/make_left_plans.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "plans")
HOME = "-2.2,-0.10,-1.45,1.45"
Y1, Y2 = -1.35, -0.85
BOX_Y1 = f"{Y1 - 0.4:.2f},{Y1 + 0.4:.2f},-1.25,1.25"
BOX_Y2 = f"{Y2 - 0.4:.2f},{Y2 + 0.4:.2f},-1.25,1.25"
S1 = (-0.95, 0.10); BOX_S1 = f"{S1[0] - 0.4:.2f},{S1[0] + 0.4:.2f},{S1[1] - 0.4:.2f},{S1[1] + 0.4:.2f}"
SURF = "venue_carpet"


def step(label, axis, v, x, y0, up, L, box):
    d = 90 if up else -90
    return dict(label=label, test="step", args=[axis, v, x, y0 if up else -y0, d, L], surface=SURF, box=box, settle=1.0)


def plans():
    P = {}
    up, dn = True, False
    P["left_A_ramped"] = [
        step("sentinel fwd 1.0 up (Y1)", "fwd", 1.0, Y1, -1.15, up, 1.2, BOX_Y1),
        step("sentinel fwd 1.0 down (Y1)", "fwd", 1.0, Y1, -1.15, dn, 1.2, BOX_Y1),
        step("ramp fwd 1.5 up (Y1)", "fwd", 1.5, Y1, -1.15, up, 1.6, BOX_Y1),
        step("ramp fwd 1.5 down (Y1)", "fwd", 1.5, Y1, -1.15, dn, 1.6, BOX_Y1),
        step("ramp side 1.5 up (Y1)", "side", 1.5, Y1, -1.15, up, 1.6, BOX_Y1),
        step("ramp side 1.5 down (Y1)", "side", 1.5, Y1, -1.15, dn, 1.6, BOX_Y1),
        step("ramp fwd 1.5 up repeat (Y1)", "fwd", 1.5, Y1, -1.15, up, 1.6, BOX_Y1),
        step("ramp fwd 1.5 down repeat (Y1)", "fwd", 1.5, Y1, -1.15, dn, 1.6, BOX_Y1),
        step("ramp side 1.5 up repeat (Y1)", "side", 1.5, Y1, -1.15, up, 1.6, BOX_Y1),
        step("ramp side 1.5 down repeat (Y1)", "side", 1.5, Y1, -1.15, dn, 1.6, BOX_Y1),
    ]
    P["left_B_jumps"] = [
        step("jump fwd 1.0 up (Y2)", "fwd", 1.0, Y2, -1.15, up, 1.2, BOX_Y2),
        step("jump fwd 1.0 down (Y2)", "fwd", 1.0, Y2, -1.15, dn, 1.2, BOX_Y2),
        step("jump fwd 1.5 up (Y2)", "fwd", 1.5, Y2, -1.15, up, 1.6, BOX_Y2),
        step("jump fwd 1.5 down (Y2)", "fwd", 1.5, Y2, -1.15, dn, 1.6, BOX_Y2),
        step("jump side 1.5 up (Y2)", "side", 1.5, Y2, -1.15, up, 1.6, BOX_Y2),
        step("jump side 1.5 down (Y2)", "side", 1.5, Y2, -1.15, dn, 1.6, BOX_Y2),
    ]
    at = [S1[0], S1[1], 0]
    P["left_C_spins"] = [dict(label=l, test="spin", at=at, args=a, surface=SURF, box=BOX_S1) for l, a in (
        ("spin 6", [6.0]), ("spin -6", [-6.0]), ("spin 10", [10.0]), ("spin -10", [-10.0]), ("spin 14", [14.0]),
        ("gyro scale +6 for 6 s", [6.0, 6.0]), ("gyro scale -6 for 6 s", [-6.0, 6.0]))]
    P["left_E_coverage"] = [dict(label=l, test="sweep", args=a, box=HOME) for l, a in (
        ("-y edge rows (left half)", [-1.9, -0.35, -1.35, -1.15, 2, 0.5]),
        ("+y edge rows (left half)", [-1.9, -0.35, 1.05, 1.3, 2, 0.5]),
        ("-x end and defense area", [-2.15, -1.8, -1.2, 1.2, 5, 0.4]),
        ("centre spot, left part", [-0.55, -0.2, -0.55, 0.25, 3, 0.4]))]
    pts = [(-0.6, -0.9, 45), (-1.6, 0.2, 135), (-0.7, 0.9, -90), (-1.7, -1.0, 0), (-1.0, 0.0, 180), (-0.5, -0.3, 90),
           (-1.5, 0.95, -45), (-1.2, -0.6, 30), (-0.45, 0.5, -135), (-1.8, 0.6, 90), (-0.9, -1.1, 180), (-1.3, 0.45, -90)]
    P["left_F_goto_chain"] = [dict(label=f"goto {k}", test="goto", args=list(p), box="-1.95,-0.35,-1.2,1.2") for k, p in enumerate(pts)]
    P["left_G_sentinel"] = P["left_A_ramped"][:2]
    return P


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, lines in plans().items():
        with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
            for e in lines: f.write(json.dumps(e) + "\n")
        print(f"{name}.jsonl: {len(lines)} tests")
