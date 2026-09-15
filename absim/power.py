"""Power of the pooled two-proportion z-test: closed form, sample size, and simulated sweeps."""

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from absim.exact import exact_rejection_probability
from absim.mc import rate_estimate
from absim.rng import make_rng
from absim.simulate import run_trials


def closed_form_power(p_c, p_t, n_per_arm, alpha: float = 0.05):
    """Normal-approximation power of the two-sided pooled z-test, equal arms.

    The critical value uses the pooled SE under H0 (matching the test statistic) and the
    spread of delta_hat uses the unpooled SE under H1. Both rejection tails are included,
    so power at p_t == p_c is exactly alpha.
    """
    p_c, p_t, n = np.asarray(p_c, float), np.asarray(p_t, float), np.asarray(n_per_arm, float)
    z_crit = norm.ppf(1 - alpha / 2)
    p_bar = (p_c + p_t) / 2
    se0 = np.sqrt(2 * p_bar * (1 - p_bar) / n)
    se1 = np.sqrt((p_c * (1 - p_c) + p_t * (1 - p_t)) / n)
    delta = p_t - p_c
    return norm.cdf((delta - z_crit * se0) / se1) + norm.cdf((-delta - z_crit * se0) / se1)


def required_sample_size(
    mde: float, power: float = 0.80, alpha: float = 0.05, p_baseline: float = 0.10
) -> int:
    """Smallest per-arm n whose closed-form power reaches `power`.

    `mde` is the absolute difference p_t - p_c (e.g. 0.01 for 0.10 -> 0.11).
    Solved by root-finding on closed_form_power itself rather than a textbook
    rearrangement, so the calculator and the power curve cannot disagree.
    """
    p_t = p_baseline + mde

    def gap(n):
        return float(closed_form_power(p_baseline, p_t, n, alpha)) - power

    hi = 10.0
    while gap(hi) < 0:
        hi *= 2
    return math.ceil(brentq(gap, 1.0, hi, xtol=1e-6))


def power_sweep(
    effect_sizes,
    sample_sizes,
    *,
    p_c: float = 0.10,
    alpha: float = 0.05,
    n_trials: int = 20_000,
    study: str = "study3",
    with_exact: bool = True,
) -> pd.DataFrame:
    """Empirical power on a grid, next to closed-form (and optionally exact) power.

    `effect_sizes` are relative lifts (0.10 means p_t = 1.10 * p_c); `sample_sizes` are
    per-arm n. Each cell has its own seed from (study, lift, n), so any cell re-runs alone.
    """
    rows = []
    for lift in effect_sizes:
        p_t = p_c * (1 + lift)
        for n in sample_sizes:
            rng = make_rng(study, relative_lift=float(lift), n_per_arm=int(n))
            batch = run_trials(rng, n_trials, p_c, p_t, int(n))
            est = rate_estimate(batch.p[:, 0] < alpha)
            row = {
                "relative_lift": lift,
                "p_c": p_c,
                "p_t": p_t,
                "abs_delta": p_t - p_c,
                "n_per_arm": int(n),
                "power_sim": est["rate"],
                "se": est["se"],
                "ci_lo": est["lo"],
                "ci_hi": est["hi"],
                "power_closed_form": float(closed_form_power(p_c, p_t, n, alpha)),
                "n_trials": n_trials,
            }
            if with_exact:
                row["power_exact"] = exact_rejection_probability(int(n), p_c, p_t, alpha)
            rows.append(row)
    return pd.DataFrame(rows)
