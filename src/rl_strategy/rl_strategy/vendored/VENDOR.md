# Vendored RL inference code: provenance and update procedure

This directory contains **verbatim copies** of the inference code used by
the reference deploy script (`deploy_policy_grsim.py`) in the team's RL
repository. Only `rl_strategy/adapter.py` calls into it.

## Source

- Repo: https://github.com/Pequi-Mecanico-SSL/RL.git
- Commit: `8d3e142c7386ec14972c069c4d9bf19933620417`

| File here | Source path in RL repo |
|---|---|
| `sim2real/config.py` | `scripts/sim2real/config.py` |
| `sim2real/utils.py` | `scripts/sim2real/utils.py` |
| `sim2real/state_to_obs.py` | `scripts/sim2real/state_to_obs.py` |
| `model/model_inferece.py` | `scripts/model/model_inferece.py` |
| `model/action_dists_inferece.py` | `scripts/model/action_dists_inferece.py` |
| `sim2real/__init__.py`, `model/__init__.py` | same (empty) |

The RL repository keeps these files in lockstep with its training code;
`deploy_policy_grsim.py` there is the reference usage.

## Weights

`weights/policy_state.portable.npz` (package root) is a weights-only export of
RLlib checkpoint `PPO_selfplay_rec/PPO_Soccer_baseline_2025-03-16/checkpoint_000003`
(`policies/policy_blue/policy_state.pkl`), the checkpoint used by the RL
repository's grSim deployment. Produced by `rl_strategy/extract_weights.py`
(tolerant unpickler; no Ray install required). Stored as `.npz` (one entry
per RLlib layer tensor, original names) rather than the RL repository's pickle
format because pickles are not portable across numpy major versions.
16 tensors, 531,709 params.

## Rules

1. **Never edit files in this directory.** They must stay byte-identical to
   the pinned commit so parity against the reference deploy script is meaningful.
2. Only `rl_strategy/adapter.py` may import from this package.
3. To update: re-copy the files from the new RL commit, update the commit
   hash above, re-export weights if the checkpoint moved, and re-run
   `rl_strategy/parity_check.py` before trusting the result.
