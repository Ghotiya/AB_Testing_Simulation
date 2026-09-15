"""Pooled two-proportion z-test, vectorized over trials and looks."""

from typing import NamedTuple

import numpy as np
from scipy.stats import norm


class ZTestResult(NamedTuple):
    z: np.ndarray
    p: np.ndarray
    delta_hat: np.ndarray  # treatment rate minus control rate


def two_proportion_ztest(x_c, n_c, x_t, n_t) -> ZTestResult:
    """Two-sided pooled z-test of H0: p_t == p_c. Inputs broadcast against each other.

    SE = sqrt(p_pool (1 - p_pool) (1/n_c + 1/n_t)), no continuity correction.
    When p_pool is 0 or 1 the SE is 0 and z is undefined; that look is reported as
    z = 0, p = 1 (not significant), never NaN.
    """
    x_c = np.asarray(x_c, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    n_c = np.asarray(n_c, dtype=np.float64)
    n_t = np.asarray(n_t, dtype=np.float64)

    delta_hat = x_t / n_t - x_c / n_c
    pooled = (x_c + x_t) / (n_c + n_t)
    var = pooled * (1.0 - pooled) * (1.0 / n_c + 1.0 / n_t)

    degenerate = var <= 0.0
    se = np.sqrt(np.where(degenerate, 1.0, var))
    z = np.where(degenerate, 0.0, delta_hat / se)
    p = np.where(degenerate, 1.0, 2.0 * norm.sf(np.abs(z)))
    return ZTestResult(z=z, p=p, delta_hat=delta_hat)
