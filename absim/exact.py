"""Exact (enumerated) distribution of the z-test p-value under H0, equal arms.

The p-value of a test on counts is discrete: in particular every tie x_t == x_c gives
z = 0, p = 1, an atom of mass sum_k pmf(k)^2 (0.0077 at n = 14,800, p = 0.10). A KS test
of simulated p-values against U(0,1) detects that atom at 100k trials even when the
simulator is perfect. The right reference for validating the simulator is this exact
distribution; its departure from U(0,1) is a property of the test.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, kstwo

from absim.stats import two_proportion_ztest


@dataclass(frozen=True)
class DiscreteDistribution:
    support: np.ndarray  # sorted p-values (with repeats)
    cdf: np.ndarray  # cumulative probability at each support point

    def cdf_right(self, v):
        """P(p <= v)."""
        i = np.searchsorted(self.support, v, side="right")
        return np.where(i > 0, self.cdf[np.maximum(i - 1, 0)], 0.0)

    def cdf_left(self, v):
        """P(p < v)."""
        i = np.searchsorted(self.support, v, side="left")
        return np.where(i > 0, self.cdf[np.maximum(i - 1, 0)], 0.0)


def exact_null_pvalue_distribution(n_per_arm: int, p: float, width_sd: float = 20.0) -> DiscreteDistribution:
    """Enumerate (x_c, x_t) within width_sd binomial SDs of the mean (all of it for small n)."""
    sd = np.sqrt(n_per_arm * p * (1 - p))
    lo = max(0, int(np.floor(n_per_arm * p - width_sd * sd)))
    hi = min(n_per_arm, int(np.ceil(n_per_arm * p + width_sd * sd)))
    k = np.arange(lo, hi + 1)
    pmf = binom.pmf(k, n_per_arm, p)
    x_c, x_t = np.meshgrid(k, k, indexing="ij")
    pvals = two_proportion_ztest(x_c, n_per_arm, x_t, n_per_arm).p.ravel()
    weights = np.outer(pmf, pmf).ravel()
    order = np.argsort(pvals, kind="stable")
    return DiscreteDistribution(support=pvals[order], cdf=np.cumsum(weights[order]))


def exact_rejection_probability(
    n_per_arm: int, p_c: float, p_t: float, alpha: float = 0.05, width_sd: float = 12.0
) -> float:
    """Exact P(p < alpha) for the pooled z-test at a single look, by enumerating both arms.

    Under p_t == p_c this is the exact size; otherwise the exact power. Unlike
    closed_form_power it involves no normal approximation, so simulation should match it
    at every n, while the closed form may drift at small n.
    """

    def support(p):
        sd = np.sqrt(n_per_arm * p * (1 - p))
        lo = max(0, int(np.floor(n_per_arm * p - width_sd * sd)))
        hi = min(n_per_arm, int(np.ceil(n_per_arm * p + width_sd * sd)))
        k = np.arange(lo, hi + 1)
        return k, binom.pmf(k, n_per_arm, p)

    k_c, pmf_c = support(p_c)
    k_t, pmf_t = support(p_t)
    total = 0.0
    for x_c, w_c in zip(k_c, pmf_c):  # one row of the grid at a time keeps memory flat
        reject = two_proportion_ztest(x_c, n_per_arm, k_t, n_per_arm).p < alpha
        total += w_c * pmf_t[reject].sum()
    return float(total)


def exact_winner_summary(
    n_per_arm: int, p_c: float, p_t: float, alpha: float = 0.05, width_sd: float = 12.0
) -> dict:
    """Exact fixed-horizon P(declared winner) and E[delta_hat | declared winner].

    "Declared winner" = significant and delta_hat > 0. This is the exact value of
    the selection-only winner's curse at K = 1, with no normal approximation.
    """

    def support(p):
        sd = np.sqrt(n_per_arm * p * (1 - p))
        lo = max(0, int(np.floor(n_per_arm * p - width_sd * sd)))
        hi = min(n_per_arm, int(np.ceil(n_per_arm * p + width_sd * sd)))
        k = np.arange(lo, hi + 1)
        return k, binom.pmf(k, n_per_arm, p)

    k_c, pmf_c = support(p_c)
    k_t, pmf_t = support(p_t)
    mass, first_moment = 0.0, 0.0
    for x_c, w_c in zip(k_c, pmf_c):
        res = two_proportion_ztest(x_c, n_per_arm, k_t, n_per_arm)
        win = (res.p < alpha) & (res.delta_hat > 0)
        w = w_c * pmf_t[win]
        mass += w.sum()
        first_moment += (w * res.delta_hat[win]).sum()
    return {"p_winner": float(mass), "mean_delta_hat_winners": float(first_moment / mass) if mass > 0 else float("nan")}


def ks_against(sample, dist: DiscreteDistribution) -> dict:
    """Two-sided KS distance between a sample and a discrete distribution (right and left limits).

    The Kolmogorov p-value is conservative for discrete distributions, so a pass is valid.
    """
    s = np.sort(np.asarray(sample, dtype=np.float64))
    n = s.size
    u = np.unique(s)
    e_right = np.searchsorted(s, u, side="right") / n
    e_left = np.searchsorted(s, u, side="left") / n
    d = max(np.abs(e_right - dist.cdf_right(u)).max(), np.abs(e_left - dist.cdf_left(u)).max())
    return {"statistic": float(d), "pvalue": float(kstwo.sf(d, n))}
