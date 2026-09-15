"""Winner's curse: how much the effect reported at a significant stopping point overstates
the true effect.

A "declared winner" is a trial that stopped significant with delta_hat > 0, the result a
business ships and celebrates. The all-significant variant (either sign) is also reported,
because at low power a real share of significant results point the wrong way.

Two biases are in play and are kept apart:
- selection: at a fixed horizon (K = 1), conditioning on significance alone selects
  large estimates;
- optional stopping: peeking adds early stopping points where estimates are noisiest.
The incremental cost of peeking is bias(K) - bias(K = 1), on common random numbers.
"""

import math

import numpy as np
from scipy.stats import norm

from absim.mc import mean_estimate
from absim.peeking import StoppingOutcome, run_with_peeking


def _mean_columns(prefix: str, values: np.ndarray, true_delta: float) -> dict:
    if values.size < 2:
        return {f"{prefix}_{k}": math.nan for k in ("mean", "se", "lo", "hi", "rel_bias", "rel_lo", "rel_hi")}
    est = mean_estimate(values)
    rel = {k: math.nan for k in ("rel_bias", "rel_lo", "rel_hi")}
    if true_delta > 0:
        rel = {
            "rel_bias": est["mean"] / true_delta - 1,
            "rel_lo": est["lo"] / true_delta - 1,
            "rel_hi": est["hi"] / true_delta - 1,
        }
    return {
        f"{prefix}_mean": est["mean"],
        f"{prefix}_se": est["se"],
        f"{prefix}_lo": est["lo"],
        f"{prefix}_hi": est["hi"],
        **{f"{prefix}_{k}": v for k, v in rel.items()},
    }


def summarize_winners(outcome: StoppingOutcome, true_delta: float) -> dict:
    """Bias of reported effects among declared winners and among all significant trials."""
    d = outcome.delta_hat_at_stop
    significant = outcome.rejected
    winners = significant & (d > 0)
    n_sig = int(significant.sum())
    return {
        "share_significant": float(significant.mean()),
        "share_winners": float(winners.mean()),
        "n_winners": int(winners.sum()),
        "wrong_sign_share_of_significant": float((significant & (d < 0)).sum() / n_sig) if n_sig else math.nan,
        "mean_n_at_stop_winners": float(outcome.n_at_stop[winners].mean()) if winners.any() else math.nan,
        # the estimate reported when the test was stopped and a winner declared
        **_mean_columns("winners", d[winners], true_delta),
        # all significant trials, sign kept
        **_mean_columns("significant", d[significant], true_delta),
        # the same winners' estimates had the test run to the full horizon
        **_mean_columns("winners_full_horizon", outcome.delta_hat_final[winners], true_delta),
    }


def estimate_winners_curse_bias(
    num_looks: int,
    true_delta: float,
    *,
    rng: np.random.Generator,
    n_trials: int,
    p_c: float,
    n_per_arm: int,
    alpha: float = 0.05,
    correction: str | None = None,
) -> dict:
    """Simulate peeked experiments with a true absolute effect and summarize the winner's curse."""
    outcome = run_with_peeking(
        num_looks, correction, rng=rng, n_trials=n_trials, p_c=p_c, p_t=p_c + true_delta, n_per_arm=n_per_arm, alpha=alpha
    )
    return summarize_winners(outcome, true_delta)


def truncated_normal_winner_mean(p_c: float, p_t: float, n_per_arm: int, alpha: float = 0.05) -> float:
    """Normal-approximation E[delta_hat | z > z_crit] at a fixed horizon, the textbook
    explanation of selection bias: delta + SE1 * phi(a) / (1 - Phi(a)), a = (z_crit SE0 - delta) / SE1."""
    z_crit = norm.ppf(1 - alpha / 2)
    p_bar = (p_c + p_t) / 2
    se0 = math.sqrt(2 * p_bar * (1 - p_bar) / n_per_arm)
    se1 = math.sqrt((p_c * (1 - p_c) + p_t * (1 - p_t)) / n_per_arm)
    delta = p_t - p_c
    a = (z_crit * se0 - delta) / se1
    return delta + se1 * norm.pdf(a) / norm.sf(a)


def bootstrap_conditional_mean_difference(
    values_a: np.ndarray, mask_a: np.ndarray, values_b: np.ndarray, mask_b: np.ndarray,
    rng: np.random.Generator, n_boot: int = 400,
) -> dict:
    """SE and percentile CI of mean(values_a | mask_a) - mean(values_b | mask_b) when both are
    computed on the same trials (correlated), by Poisson bootstrap over trials."""
    va, vb = np.where(mask_a, values_a, 0.0), np.where(mask_b, values_b, 0.0)
    ma, mb = mask_a.astype(float), mask_b.astype(float)
    point = va.sum() / ma.sum() - vb.sum() / mb.sum()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        w = rng.poisson(1.0, size=va.size)
        diffs[i] = (w @ va) / (w @ ma) - (w @ vb) / (w @ mb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "se": float(diffs.std(ddof=1)), "lo": float(lo), "hi": float(hi)}
