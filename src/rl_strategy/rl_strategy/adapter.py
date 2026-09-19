"""Adapter between the vendored RL policy and the real-world ROS stack.

Callers of this module use real-world SI in the real field frame: meters,
radians, right-handed (x toward the +x goal, y left, theta CCW).

Internally, the policy works in its training frame: the 9x6 training field
(``vendored/sim2real/config.py``: FIELD_LENGTH=9, FIELD_WIDTH=6), degrees,
normalized actions. This module owns the entire mapping:

    in : real pose (m, rad)  --x scale-->  policy pose (training-frame m, deg)
    out: policy velocity     --/ scale-->  real body-frame velocity (m/s, rad/s)

with a single isotropic scale, so geometry (angles, aspect) is preserved.
The default scale is 1.0 (real meters are fed unchanged); when derived it
is 9.0 / real_field_length, and the real field then occupies a centered
band of the training field. Angular velocity is scale-invariant and passes
through.

All glue decisions follow the reference deploy script in the team's RL
repository (Pequi-Mecanico-SSL/RL, ``deploy_policy_grsim.py``; "the deploy"
below). parity_check.py asserts action-level equivalence against it. The
vendored inference code is in ``vendored/`` (see vendored/VENDOR.md); this is
the only module allowed to import it.
"""
import numpy as np
import torch

from . import vendored  # noqa: F401  (installs the vendored import shim)
from model.model_inferece import InferenceModel
from model.action_dists_inferece import InferenceBetaDist
from sim2real.state_to_obs import frame_to_observations
from sim2real.config import FIELD_LENGTH as POLICY_FIELD_LENGTH
from sim2real.config import MAX_V as POLICY_MAX_V
from sim2real.config import MAX_W as POLICY_MAX_W

OBS_SIZE = 77
N_STACK = 8
ACTION_SIZE = 4
# The deploy applies a fixed 3 m/s kick for any positive kick action
# (policy units). Our robots have no kicker yet; the node ignores this field.
POLICY_KICK_SPEED = 3.0

# Training kickoff formation in training-frame coordinates (the RL repo's
# reset_grsim_positions.py / config.yaml init_pos). Used to detect the
# episode-reset signature the same way deploy_policy_grsim.is_kickoff_formation
# does: ball at center + all six robots on their marks never happens in open
# play, so it doubles as a synchronized reset signal with no side channel.
KICKOFF_POLICY_FRAME = {
    "blue": [(-1.5, 0.0), (-2.0, 1.0), (-2.0, -1.0)],
    "yellow": [(1.5, 0.0), (2.0, 1.0), (2.0, -1.0)],
}
KICKOFF_BALL_TOL = 0.15   # training-frame meters, as in the deploy
KICKOFF_ROBOT_TOL = 0.25  # training-frame meters, as in the deploy


def load_model(weights_path, device="cpu"):
    """Build InferenceModel and load a portable ``.npz`` weights export.

    The npz holds one entry per RLlib layer tensor (see extract_weights.py).
    Layer-name remap copied verbatim from the deploy's ``_load_model``
    (RLlib ``_hidden_layers.<i>._model.0.<p>`` -> Sequential ``_hidden_layers.<i*2>.<p>``).
    """
    model = InferenceModel(input_size=OBS_SIZE * N_STACK, output_size=2 * ACTION_SIZE)

    with np.load(weights_path) as npz:
        raw_weights = {name: npz[name] for name in npz.files}

    weights_dict = {}
    for layer_name, weights in raw_weights.items():
        split = layer_name.split(".")
        if "_logits" == split[0] or "_value_branch" == split[0]:
            new_layer_name = split[0] + "." + split[-1]
        else:
            if len(split) >= 3:
                layer_idx = int(split[1])
                new_layer_name = split[0] + "." + str(layer_idx * 2) + "." + split[-1]
            else:
                new_layer_name = layer_name
        weights_dict[new_layer_name] = torch.tensor(np.asarray(weights))

    model.load_state_dict(weights_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _global_normalized_to_body(action, theta_degrees):
    """Policy-normalized global action -> body-frame policy velocities.

    Verbatim math from the deploy's ``normalized_action_to_grsim``
    (which hardcodes 1.5 m/s and 10.0 rad/s, identical to the vendored
    config's MAX_V / MAX_W, used here so the constants stay tied to the policy).
    Returns (v_forward, v_left, v_angular, kick) in policy units.
    """
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    global_velocity = action[:2] * POLICY_MAX_V
    speed = np.linalg.norm(global_velocity)
    if speed > POLICY_MAX_V:
        global_velocity *= POLICY_MAX_V / speed

    theta = np.deg2rad(theta_degrees)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    tangent = global_velocity[0] * cos_theta + global_velocity[1] * sin_theta
    normal = -global_velocity[0] * sin_theta + global_velocity[1] * cos_theta
    kick = POLICY_KICK_SPEED if action[3] > 0.0 else 0.0
    return tangent, normal, action[2] * POLICY_MAX_W, kick


def default_placeholders(field_length_m, field_width_m, color):
    """Benign real-frame poses for robots vision can't see.

    Parked spread across the team's own half (blue defends -x, yellow +x),
    facing midfield: stationary teammates away from the play.
    """
    sx = -1.0 if color == "blue" else 1.0
    theta = 0.0 if color == "blue" else np.pi
    return {
        i: (sx * 0.35 * field_length_m, dy * 0.25 * field_width_m, theta)
        for i, dy in ((0, 0.0), (1, 1.0), (2, -1.0))
    }


def kickoff_placeholders(scale, color):
    """Real-frame placeholders on the training spawn marks (KICKOFF_POLICY_FRAME/scale).

    For the solo real-robot setup: the one seen robot plays its slot, every
    unseen slot sits exactly where the policy expects robots at episode start,
    so the kickoff-reset check reduces to "ball centered + the real robot
    near its own mark". With scale=1 the marks are at the training-field
    coordinates (e.g. (-2.0, +-1.0) m), which may lie outside a small real
    field; that is intended, the policy only ever sees them as inputs.
    """
    theta = 0.0 if color == "blue" else np.pi
    return {
        i: (kx / scale, ky / scale, theta)
        for i, (kx, ky) in enumerate(KICKOFF_POLICY_FRAME[color])
    }


class PolicyAdapter:
    """Real-SI world in -> real-SI body-frame commands out."""

    def __init__(
        self,
        weights_path,
        team="blue",
        field_length_m=3.3,
        field_width_m=2.0,
        n_robots_blue=3,
        n_robots_yellow=3,
        action_mode="mean",
        action_seed=0,
        device="cpu",
        placeholders=None,
        placeholder_mode="parked",  # "parked" | "kickoff" (training spawn marks)
        episode_reset=True,
        scale=None,
    ):
        if action_mode not in ("mean", "sample"):
            raise ValueError(f"invalid action_mode: {action_mode!r}")
        if team not in ("blue", "yellow"):
            raise ValueError(f"invalid team: {team!r}")

        self.team = team
        self.n_robots = {"blue": n_robots_blue, "yellow": n_robots_yellow}
        self.action_mode = action_mode
        self.device = torch.device(device)
        torch.manual_seed(action_seed)

        # Coordinate scale between the real field and the training frame.
        # scale=1 (feed real meters) is the default: with scale != 1 robot and
        # ball sizes are scaled in the policy's view, so the trained
        # ball-contact distance no longer matches the physical one and
        # ball-contact behaviors are expected to degrade. Keep scale=1 unless a policy
        # is trained for scaled worlds. scale=None derives the geometric field
        # mapping (9.0/field_length) for experiments and the parity harness.
        self.scale = scale if scale is not None else POLICY_FIELD_LENGTH / field_length_m

        # Real-frame poses used for robots vision doesn't see.
        if placeholders is None:
            maker = (kickoff_placeholders if placeholder_mode == "kickoff"
                     else lambda s, c: default_placeholders(field_length_m, field_width_m, c))
            placeholders = {color: maker(self.scale, color)
                        for color in ("blue", "yellow")}
        self.placeholders = placeholders

        self.model = load_model(weights_path, device=device)

        # Kickoff-signature episode reset (mirrors deploy episode_reset mode).
        # Inert unless the full 6-robot formation + centered ball appears, so
        # it is safe to leave on for partial-field setups.
        self.episode_reset = episode_reset
        self._kickoff_latched = False
        self.did_kickoff_reset = False  # observability: set by the last step()

        self.step_count = 0
        self.stacked_obs = {
            **{f"blue_{i}": np.zeros(OBS_SIZE * N_STACK, dtype=np.float32)
               for i in range(n_robots_blue)},
            **{f"yellow_{i}": np.zeros(OBS_SIZE * N_STACK, dtype=np.float32)
               for i in range(n_robots_yellow)},
        }
        self.last_actions = {
            **{f"blue_{i}": np.zeros(ACTION_SIZE, dtype=np.float32)
               for i in range(n_robots_blue)},
            **{f"yellow_{i}": np.zeros(ACTION_SIZE, dtype=np.float32)
               for i in range(n_robots_yellow)},
        }
        self._last_ball = (0.0, 0.0)

    def reset(self):
        """Reset episode state (obs stack, last actions, step counter)."""
        self.step_count = 0
        for k in self.stacked_obs:
            self.stacked_obs[k] = np.zeros(OBS_SIZE * N_STACK, dtype=np.float32)
        for k in self.last_actions:
            self.last_actions[k] = np.zeros(ACTION_SIZE, dtype=np.float32)

    # ---- frame conversion --------------------------------------------------

    def _to_policy_frame(self, world):
        """Real-SI world -> the deploy's exact frame dict, in policy units."""
        s = self.scale
        frame = {"robots_blue": {}, "robots_yellow": {}, "ball": None}

        for color in ("blue", "yellow"):
            seen = world.get(color) or {}
            for i in range(self.n_robots[color]):
                x, y, theta = seen.get(i) or self.placeholders[color][i]
                frame[f"robots_{color}"][f"robot_{i}"] = [
                    x * s, y * s, np.rad2deg(theta)
                ]

        ball = world.get("ball")
        if ball is not None:
            self._last_ball = (float(ball[0]), float(ball[1]))
        bx, by = self._last_ball
        frame["ball"] = [bx * s, by * s]
        return frame

    def _is_kickoff_formation(self, frame):
        """Deploy's is_kickoff_formation, in training-frame coordinates."""
        bx, by = frame["ball"]
        if (bx * bx + by * by) ** 0.5 > KICKOFF_BALL_TOL:
            return False
        for color in ("blue", "yellow"):
            for i, (kx, ky) in enumerate(KICKOFF_POLICY_FRAME[color]):
                x, y, _ = frame[f"robots_{color}"][f"robot_{i}"]
                if ((x - kx) ** 2 + (y - ky) ** 2) ** 0.5 > KICKOFF_ROBOT_TOL:
                    return False
        return True

    def step(self, world):
        """Run one inference step.

        world (real SI, right-handed):
            {"blue":   {robot_id: (x_m, y_m, theta_rad), ...},   # seen robots only
             "yellow": {robot_id: (x_m, y_m, theta_rad), ...},
             "ball":   (x_m, y_m) or None}

        Returns {robot_id: (v_forward, v_left, v_angular, kick_speed)} for the
        adapter's team, in real SI body frame (m/s, m/s, rad/s, m/s).
        """
        frame = self._to_policy_frame(world)

        # Synchronized episode reset on the kickoff signature (latched, like
        # the deploy: reset once per formation appearance).
        self.did_kickoff_reset = False
        if self.episode_reset:
            if self._is_kickoff_formation(frame):
                if not self._kickoff_latched:
                    self._kickoff_latched = True
                    self.reset()
                    self.did_kickoff_reset = True
            else:
                self._kickoff_latched = False

        # Obs + stack: the vendored observation builder, called exactly as the deploy does.
        observations = frame_to_observations(
            frame, self.last_actions, self.stacked_obs, steps=self.step_count
        )
        self.stacked_obs.update(observations)

        # Model forward for this team, mirroring deploy _compute_actions.
        model_inputs, robot_names = [], []
        for robot_name in sorted(self.stacked_obs.keys()):
            if self.team in robot_name:
                model_inputs.append(self.stacked_obs[robot_name])
                robot_names.append(robot_name)

        model_input = torch.tensor(
            np.array(model_inputs, dtype=np.float32)
        ).to(self.device)
        if not torch.isfinite(model_input).all():
            raise ValueError("observation contains NaN/Inf")

        with torch.no_grad():
            model_output, _value = self.model(model_input)
        if not torch.isfinite(model_output).all():
            raise ValueError("logits contain NaN/Inf")

        signal = [-1, 1, -1, 1] if self.team == "yellow" else [1, 1, 1, 1]
        distribution = InferenceBetaDist(model_output, signal=signal)
        if self.action_mode == "mean":
            actions_tensor = distribution.deterministic_sample()
        else:
            actions_tensor = distribution.sample()
        if not torch.isfinite(actions_tensor).all():
            raise ValueError("actions contain NaN/Inf")
        actions = actions_tensor.detach().cpu().numpy()

        # Convert to body-frame policy velocities, then map back to real SI:
        # divide translational speeds by the scale. Angular is scale-invariant.
        commands = {}
        for i, robot_name in enumerate(robot_names):
            self.last_actions[robot_name] = np.asarray(
                actions[i], dtype=np.float32
            )
            _, idx = robot_name.split("_")
            idx = int(idx)
            theta_deg = frame[f"robots_{self.team}"][f"robot_{idx}"][2]
            tangent, normal, angular, kick = _global_normalized_to_body(
                actions[i], theta_deg
            )
            commands[idx] = (
                tangent / self.scale,
                normal / self.scale,
                angular,
                kick / self.scale,
            )

        self.step_count += 1
        return commands
