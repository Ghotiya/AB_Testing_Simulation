"""Interim peeking with stop-at-first-significance, with and without correction.

The experiment is tested at every look and stops at the first look whose p-value is
below the per-look threshold. A trial that never crosses runs to the full horizon N
, so the fixed-horizon test is exactly the K = 1 case.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from absim.simulate import TrialBatch, run_trials
from absim.theory import pocock_constant

CORRECTIONS = (None, "bonferroni", "pocock")


def per_look_alpha(num_looks: int, alpha: float, correction: str | None) -> float:
    """Nominal p-value threshold applied at every look.

    bonferroni: alpha / K, which ignores the correlation between nested looks (conservative).
    pocock: the constant boundary calibrated so the overall rate is alpha for equally spaced
    looks, the reference for how much Bonferroni over-corrects.
    """
    if correction is None:
        return alpha
    if correction == "bonferroni":
        return alpha / num_looks
    if correction == "pocock":
        return float(2 * norm.sf(pocock_constant(num_looks, alpha)))
    raise ValueError(f"unknown correction {correction!r}; expected one of {CORRECTIONS}")


@dataclass(frozen=True)
class StoppingOutcome:
    alpha_per_look: float
    rejected: np.ndarray  # (T,) bool: crossed at some look
    stop_index: np.ndarray  # (T,) index of the first crossing, or the last look if none
    n_at_stop: np.ndarray  # (T,) per-arm sample size when the experiment ended
    delta_hat_at_stop: np.ndarray  # (T,) effect estimate reported at the stopping point
    delta_hat_final: np.ndarray  # (T,) estimate the same trial would give at the full horizon


def apply_stopping(batch: TrialBatch, alpha: float = 0.05, correction: str | None = None) -> StoppingOutcome:
    a = per_look_alpha(batch.num_looks, alpha, correction)
    crossed = batch.p < a
    rejected = crossed.any(axis=1)
    stop_index = np.where(rejected, crossed.argmax(axis=1), batch.num_looks - 1)
    rows = np.arange(batch.n_trials)
    return StoppingOutcome(
        alpha_per_look=a,
        rejected=rejected,
        stop_index=stop_index,
        n_at_stop=batch.n[stop_index],
        delta_hat_at_stop=batch.delta_hat[rows, stop_index],
        delta_hat_final=batch.delta_hat[:, -1],
    )


def run_with_peeking(
    num_looks: int,
    correction: str | None = None,
    *,
    rng: np.random.Generator,
    n_trials: int,
    p_c: float,
    p_t: float,
    n_per_arm: int,
    alpha: float = 0.05,
) -> StoppingOutcome:
    """Simulate n_trials experiments peeked at num_looks equally spaced looks."""
    batch = run_trials(rng, n_trials, p_c, p_t, n_per_arm, num_looks)
    return apply_stopping(batch, alpha, correction)
