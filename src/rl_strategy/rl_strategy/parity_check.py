#!/usr/bin/env python3
"""Parity harness: PolicyAdapter vs the reference deploy script in the team's
RL repository (Pequi-Mecanico-SSL/RL, deploy_policy_grsim.py).

Two gates, both must pass before the adapter is used on hardware:

1. Glue parity (scale=1): adapter configured with a 9x6 "real" field (so the
   scale is the identity) must produce, for an identical synthetic match
   sequence, the same commands as a reference pipeline assembled from the RL
   repo's own modules (frame_to_observations, InferenceModel,
   InferenceBetaDist, normalized_action_to_grsim) driven exactly like
   deploy_policy_grsim.GrSimVisionController.step does.

2. Scale closure (real scale): the same match expressed in real meters
   (training-frame coords / scale) through an adapter configured with the real field must
   yield reference translational commands divided by the scale, identical
   angular commands.

Runs without ROS, in any venv with numpy+torch+protobuf:
    python3 parity_check.py --ref /path/to/RL \
        --weights /path/to/policy_state.portable.npz
"""
import argparse
import importlib
import os
import sys

import numpy as np
import torch

TOL = 1e-5
N_STEPS = 60
POLICY_L, POLICY_W = 9.0, 6.0


def synthetic_frame(k):
    """Deterministic scripted 3v3 + ball state, in training-frame units (9x6, degrees)."""
    t = k / 30.0
    frame = {"robots_blue": {}, "robots_yellow": {}, "ball": None}
    for i in range(3):
        frame["robots_blue"][f"robot_{i}"] = [
            -2.0 + 0.8 * np.sin(t + i), (i - 1) * 1.2 + 0.3 * np.cos(t * 1.3 + i),
            np.rad2deg(0.5 * t + i)
        ]
        frame["robots_yellow"][f"robot_{i}"] = [
            2.0 - 0.8 * np.sin(t * 0.7 + i), (1 - i) * 1.2 - 0.3 * np.cos(t + i),
            np.rad2deg(np.pi - 0.4 * t + i)
        ]
    frame["ball"] = [1.5 * np.sin(t * 0.9), 0.8 * np.cos(t * 1.1)]
    return frame


def reference_run(ref_repo, weights_path, team):
    """Drive the RL repo's own modules exactly as deploy_policy_grsim does."""
    sys.path.insert(0, ref_repo)                       # deploy + protos + scripts pkg
    sys.path.insert(0, os.path.join(ref_repo, "scripts"))  # sim2real as top-level

    # reference implementation from the RL repo
    from sim2real.state_to_obs import frame_to_observations
    from scripts.model.model_inferece import InferenceModel
    from scripts.model.action_dists_inferece import InferenceBetaDist
    deploy = importlib.import_module("deploy_policy_grsim")

    # The reference deploy script's _load_model logic, fed with the portable
    # weights export.
    model = InferenceModel(input_size=77 * 8, output_size=8)
    with np.load(weights_path) as npz:
        raw_weights = {name: npz[name] for name in npz.files}
    weights_dict = {}
    for layer_name, weights in raw_weights.items():
        split = layer_name.split(".")
        if "_logits" == split[0] or "_value_branch" == split[0]:
            new_layer_name = split[0] + "." + split[-1]
        else:
            if len(split) >= 3:
                new_layer_name = split[0] + "." + str(int(split[1]) * 2) + "." + split[-1]
            else:
                new_layer_name = layer_name
        weights_dict[new_layer_name] = torch.tensor(np.asarray(weights))
    model.load_state_dict(weights_dict, strict=True)
    model.eval()

    stacked = {
        **{f"blue_{i}": np.zeros(77 * 8, dtype=np.float32) for i in range(3)},
        **{f"yellow_{i}": np.zeros(77 * 8, dtype=np.float32) for i in range(3)},
    }
    last_actions = {
        **{f"blue_{i}": np.zeros(4, dtype=np.float32) for i in range(3)},
        **{f"yellow_{i}": np.zeros(4, dtype=np.float32) for i in range(3)},
    }

    per_step = []
    for k in range(N_STEPS):
        frame = synthetic_frame(k)
        obs = frame_to_observations(frame, last_actions, stacked, steps=k)
        stacked.update(obs)

        model_inputs, names = [], []
        for name in sorted(stacked.keys()):
            if team in name:
                model_inputs.append(stacked[name])
                names.append(name)
        x = torch.tensor(np.array(model_inputs, dtype=np.float32))
        with torch.no_grad():
            out, _ = model(x)
        signal = [-1, 1, -1, 1] if team == "yellow" else [1, 1, 1, 1]
        dist = InferenceBetaDist(out, signal=signal)
        actions = dist.deterministic_sample().detach().numpy()

        cmds = {}
        for i, name in enumerate(names):
            last_actions.update({name: actions[i].tolist()})  # deploy stores lists
            idx = int(name.split("_")[1])
            theta_deg = frame[f"robots_{team}"][f"robot_{idx}"][2]
            cmds[idx] = deploy.normalized_action_to_grsim(actions[i], theta_deg)
        per_step.append(cmds)
    return per_step


def _purge_reference_modules(ref_repo):
    """Make sure the adapter imports its own vendored code, not the reference's.

    Both the reference repo and the vendored copy expose a top-level
    ``sim2real`` module; without this purge the adapter would silently bind to
    the reference's copy and gate 1 would prove nothing about the vendoring.
    """
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in ("sim2real", "model", "scripts", "deploy_policy_grsim"):
            del sys.modules[name]
    for p in (ref_repo, os.path.join(ref_repo, "scripts")):
        while p in sys.path:
            sys.path.remove(p)


def adapter_run(weights_path, team, field_length, field_width, coord_scale,
             ref_repo=None):
    """Drive our PolicyAdapter on the same match, coordinates / coord_scale."""
    if ref_repo:
        _purge_reference_modules(ref_repo)
    from rl_strategy.adapter import PolicyAdapter

    adapter = PolicyAdapter(
        weights_path=weights_path, team=team,
        field_length_m=field_length, field_width_m=field_width,
    )
    per_step = []
    for k in range(N_STEPS):
        frame = synthetic_frame(k)
        world = {"blue": {}, "yellow": {}, "ball": None}
        for color in ("blue", "yellow"):
            for i in range(3):
                x, y, th_deg = frame[f"robots_{color}"][f"robot_{i}"]
                world[color][i] = (x / coord_scale, y / coord_scale,
                                   np.deg2rad(th_deg))
        world["ball"] = (frame["ball"][0] / coord_scale,
                         frame["ball"][1] / coord_scale)
        per_step.append(adapter.step(world))
    return per_step


def compare(label, ref_steps, adapter_steps, vel_scale):
    """adapter translational cmd must equal reference / vel_scale."""
    worst = 0.0
    for k, (ref, got) in enumerate(zip(ref_steps, adapter_steps)):
        assert ref.keys() == got.keys(), f"{label} step {k}: robot sets differ"
        for idx in ref:
            rt, rn, ra, rk = ref[idx]
            gt, gn, ga, gk = got[idx]
            errs = [abs(rt / vel_scale - gt), abs(rn / vel_scale - gn),
                    abs(ra - ga), abs(rk / vel_scale - gk)]
            worst = max(worst, *errs)
            if max(errs) > TOL:
                raise AssertionError(
                    f"{label} step {k} robot {idx}: "
                    f"ref/s=({rt/vel_scale:.6f},{rn/vel_scale:.6f},{ra:.6f},{rk/vel_scale:.6f}) "
                    f"adapter=({gt:.6f},{gn:.6f},{ga:.6f},{gk:.6f})"
                )
    print(f"PASS {label}: {len(ref_steps)} steps, worst abs err {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="checkout of the RL repo at the commit pinned in vendored/VENDOR.md")
    ap.add_argument("--weights", required=True, help="portable weights npz (see vendored/VENDOR.md)")
    ap.add_argument("--team", default="blue", choices=["blue", "yellow"])
    ap.add_argument("--field-length", type=float, default=3.3)
    ap.add_argument("--field-width", type=float, default=2.0)
    args = ap.parse_args()

    # our package importable when run from the repo checkout
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, pkg_root)

    ref_repo = os.path.abspath(args.ref)
    ref = reference_run(ref_repo, args.weights, args.team)

    # Gate 1: identity scale; the adapter is told the field is 9x6.
    z1 = adapter_run(args.weights, args.team, POLICY_L, POLICY_W, coord_scale=1.0,
                  ref_repo=ref_repo)
    compare("glue-parity (scale=1)", ref, z1, vel_scale=1.0)

    # Gate 2: real field; the scale mapping must close exactly.
    s = POLICY_L / args.field_length
    z2 = adapter_run(args.weights, args.team, args.field_length, args.field_width,
                  coord_scale=s)
    compare(f"scale-closure (scale={s:.3f})", ref, z2, vel_scale=s)

    print("ALL PARITY GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
