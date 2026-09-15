"""Large-sample reference for repeated significance tests (Armitage, McPherson & Rowe 1969).

As n grows, the z-statistics at nested looks behave like a standard Brownian motion W
observed at information fractions t_k = n_k / N, with Z_k = W(t_k) / sqrt(t_k). The
probability of crossing |Z_k| >= c_k at some look is computed by recursive numerical
integration: propagate the density of W on a grid, remove the mass beyond each look's
boundary, and convolve with the Gaussian increment to the next look.

This is independent of the simulator (no binomial draws, no z-test on counts), so
agreement between the two validates the simulated peeking curve.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.stats import norm


def gaussian_crossing_probability(info_fractions, z_crit, grid_step: float = 5e-4, half_width: float = 8.0) -> float:
    """P(|Z_k| >= z_crit_k at some look) under H0 for looks at the given information fractions."""
    t = np.asarray(info_fractions, dtype=np.float64)
    c = np.broadcast_to(np.asarray(z_crit, dtype=np.float64), t.shape)
    if np.any(np.diff(t) <= 0) or t[0] <= 0:
        raise ValueError("information fractions must be strictly increasing and positive")

    x = np.arange(-half_width, half_width + grid_step / 2, grid_step)
    density = None
    prev_t = 0.0
    crossed = 0.0
    for t_k, c_k in zip(t, c):
        sd = np.sqrt(t_k - prev_t)
        if density is None:
            density = norm.pdf(x, scale=sd)
        else:
            m = min(int(np.ceil(10 * sd / grid_step)), x.size // 2)
            kernel = norm.pdf(np.arange(-m, m + 1) * grid_step, scale=sd) * grid_step
            density = fftconvolve(density, kernel, mode="same")
        outside = np.abs(x) >= c_k * np.sqrt(t_k)
        crossed += density[outside].sum() * grid_step
        density[outside] = 0.0
        prev_t = t_k
    return float(crossed)


@lru_cache(maxsize=None)
def pocock_constant(num_looks: int, alpha: float = 0.05) -> float:
    """Pocock (1977) boundary: the single critical |z| used at all K equally spaced looks so the
    overall false-positive rate is exactly alpha. Unlike Bonferroni it accounts for the positive
    correlation between nested looks, so it lies between z_(alpha/2) and z_(alpha/(2K))."""
    if num_looks == 1:
        return float(norm.ppf(1 - alpha / 2))
    t = np.arange(1, num_looks + 1) / num_looks
    lo = norm.ppf(1 - alpha / 2)
    hi = norm.ppf(1 - alpha / (2 * num_looks))
    return float(brentq(lambda c: gaussian_crossing_probability(t, c) - alpha, lo, hi, xtol=1e-6))
