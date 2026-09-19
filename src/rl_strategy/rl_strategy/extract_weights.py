#!/usr/bin/env python3
"""Extract portable weights from a raw RLlib checkpoint, no Ray required.

Raw RLlib ``policy_state.pkl`` files reference ``ray.cloudpickle`` and embed
pickled code objects from the training Python version, so they cannot be
unpickled on a normal machine. Only ``state["weights"]`` (plain numpy
arrays) is needed. This tool unpickles tolerantly:

- remaps ``ray.cloudpickle.*`` -> the real ``cloudpickle`` package;
- stubs cloudpickle's ``_builtin_type`` for code/function types so foreign
  code objects reconstruct as inert placeholders instead of crashing;
- absorbs any other unresolvable class as an inert placeholder.

Numpy arrays reconstruct through numpy's own pickle paths and are untouched.

The output is a ``.npz`` archive (one entry per layer tensor). Unlike a
pickle, npz is portable across numpy major versions: a pickle written under
numpy 2.x fails to load under numpy 1.x (``No module named 'numpy._core'``),
e.g. when the export environment and the ROS container carry different
numpy majors.

Usage:
    python3 extract_weights.py <raw_policy_state.pkl> <out.portable.npz>

Requires: numpy, cloudpickle (pip). Run once per new checkpoint.
"""
import pickle
import sys
import types

import numpy as np


class Inert:
    """Absorbs any pickle reconstruction that is not needed."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return Inert()

    def __setstate__(self, s):
        pass


def _safe_builtin_type(name):
    if name in ("CodeType", "FunctionType", "LambdaType", "MethodType"):
        return Inert()  # callable; swallows constructor args
    t = getattr(types, name, None)
    return t if t is not None else Inert()


def _guarded(fn):
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception:
            return Inert()

    return wrapper


class TolerantRLlibUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("ray.cloudpickle"):
            module = module.replace("ray.cloudpickle", "cloudpickle", 1)
        if module.startswith("cloudpickle") and name == "_builtin_type":
            return _safe_builtin_type
        try:
            obj = super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return Inert
        if module.startswith("cloudpickle") and callable(obj):
            return _guarded(obj)
        return obj


def extract(src_path, dst_path):
    with open(src_path, "rb") as f:
        state = TolerantRLlibUnpickler(f).load()
    weights = {
        name: np.asarray(value).copy() for name, value in state["weights"].items()
    }
    n_params = int(sum(v.size for v in weights.values()))
    np.savez(dst_path, **weights)
    return len(weights), n_params


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    n_tensors, n_params = extract(sys.argv[1], sys.argv[2])
    print(f"exported {n_tensors} tensors, {n_params} params -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
