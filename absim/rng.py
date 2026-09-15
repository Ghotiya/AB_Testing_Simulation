"""Reproducible seeding.

One master seed. Each (study, config) gets its own child stream whose spawn key is a
stable hash of the config, so any single cell can be re-run in isolation and reproduce
exactly, independent of which other cells were run or in what order.
"""

import hashlib
import json

import numpy as np

MASTER_SEED = 20260912


def _spawn_key(study: str, config: dict) -> tuple[int, ...]:
    blob = json.dumps({"study": study, "config": config}, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode()).digest()
    return tuple(int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4))


def make_rng(study: str, **config) -> np.random.Generator:
    seq = np.random.SeedSequence(MASTER_SEED, spawn_key=_spawn_key(study, config))
    return np.random.Generator(np.random.PCG64(seq))
