"""Trial simulation: a readable scalar reference and the vectorized production path.

Looks are nested on one cumulative data stream: each trial draws per-interval
successes Binomial(dn, p) and cumulative-sums them, so look k sees exactly the data of
look k-1 plus the new interval. Drawing fresh samples per look would decorrelate the
looks and silently produce a wrong peeking curve.

Both paths consume the generator in the same order (per trial: all control increments,
then all treatment increments), so a loop of run_single_trial() on one generator
reproduces run_trials() exactly. tests/test_simulate.py enforces this.
"""

import math
from dataclasses import dataclass

import numpy as np

from absim.stats import two_proportion_ztest


def look_schedule(n_per_arm: int, num_looks: int) -> np.ndarray:
    """Cumulative per-arm sample size at each look: n_k = round(k N / K), half up."""
    k = np.arange(1, num_looks + 1, dtype=np.int64)
    return (2 * k * n_per_arm + num_looks) // (2 * num_looks)


@dataclass(frozen=True)
class TrialBatch:
    """Results of n_trials simulated experiments, each tested at every look."""

    n: np.ndarray  # (K,) cumulative per-arm sample size at each look
    x_c: np.ndarray  # (T, K) cumulative control conversions
    x_t: np.ndarray  # (T, K) cumulative treatment conversions
    z: np.ndarray  # (T, K)
    p: np.ndarray  # (T, K)
    delta_hat: np.ndarray  # (T, K)

    @property
    def n_trials(self) -> int:
        return self.x_c.shape[0]

    @property
    def num_looks(self) -> int:
        return self.n.shape[0]

    def select_looks(self, n_values) -> "TrialBatch":
        """The same trials, tested only at the looks with these cumulative sizes.

        Simulating once on the union of several look schedules and selecting each schedule
        from it gives common random numbers across schedules.
        """
        idx = np.searchsorted(self.n, n_values)
        if not np.array_equal(self.n[idx], n_values):
            raise ValueError("requested looks are not in this batch")
        return TrialBatch(
            n=self.n[idx],
            x_c=self.x_c[:, idx],
            x_t=self.x_t[:, idx],
            z=self.z[:, idx],
            p=self.p[:, idx],
            delta_hat=self.delta_hat[:, idx],
        )


def union_look_schedule(n_per_arm: int, look_counts) -> np.ndarray:
    """Sorted union of the look schedules for several K, so one stream serves all of them."""
    return np.unique(np.concatenate([look_schedule(n_per_arm, K) for K in look_counts]))


def run_trials(
    rng: np.random.Generator,
    n_trials: int,
    p_c: float,
    p_t: float,
    n_per_arm: int,
    num_looks: int = 1,
) -> TrialBatch:
    """Simulate n_trials experiments with true rates p_c, p_t, tested at num_looks looks."""
    return run_trials_at_looks(rng, n_trials, p_c, p_t, look_schedule(n_per_arm, num_looks))


def run_trials_at_looks(
    rng: np.random.Generator,
    n_trials: int,
    p_c: float,
    p_t: float,
    looks,
) -> TrialBatch:
    """Simulate n_trials experiments tested at the given increasing cumulative per-arm sizes."""
    n = np.asarray(looks, dtype=np.int64)
    if n.ndim != 1 or n[0] <= 0 or np.any(np.diff(n) <= 0):
        raise ValueError("looks must be strictly increasing positive sizes")
    num_looks = n.shape[0]
    increments = np.diff(n, prepend=0)
    probs = np.array([p_c, p_t])[None, :, None]
    draws = rng.binomial(increments[None, None, :], probs, size=(n_trials, 2, num_looks))
    counts = np.cumsum(draws, axis=2)
    x_c, x_t = counts[:, 0, :], counts[:, 1, :]
    z, p, delta_hat = two_proportion_ztest(x_c, n, x_t, n)
    return TrialBatch(n=n, x_c=x_c, x_t=x_t, z=z, p=p, delta_hat=delta_hat)


def run_single_trial(
    rng: np.random.Generator,
    p_c: float,
    p_t: float,
    n_per_arm: int,
    num_looks: int = 1,
) -> list[dict]:
    """One experiment, written for reading rather than speed. Returns one row per look."""
    looks = [int(n_k) for n_k in look_schedule(n_per_arm, num_looks)]

    cumulative = {}
    for arm, rate in (("control", p_c), ("treatment", p_t)):
        total, prev_n, path = 0, 0, []
        for n_k in looks:
            total += int(rng.binomial(n_k - prev_n, rate))  # only the new interval's users
            prev_n = n_k
            path.append(total)
        cumulative[arm] = path

    rows = []
    for k, n_k in enumerate(looks):
        x_c, x_t = cumulative["control"][k], cumulative["treatment"][k]
        delta_hat = x_t / n_k - x_c / n_k
        pooled = (x_c + x_t) / (2 * n_k)
        se = math.sqrt(pooled * (1 - pooled) * (2 / n_k))
        if se == 0:
            z, p = 0.0, 1.0
        else:
            z = delta_hat / se
            p = math.erfc(abs(z) / math.sqrt(2))  # two-sided normal tail
        rows.append({"n": n_k, "x_c": x_c, "x_t": x_t, "delta_hat": delta_hat, "z": z, "p": p})
    return rows
