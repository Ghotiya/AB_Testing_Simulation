"""Monte Carlo uncertainty helpers. Every reported rate carries its MC standard error."""

import math

import numpy as np
from scipy.stats import norm


def rate_estimate(hits, conf: float = 0.95) -> dict:
    """Rate of a boolean array with its MC standard error and normal CI."""
    hits = np.asarray(hits, dtype=bool)
    n = hits.size
    rate = hits.mean()
    se = math.sqrt(rate * (1 - rate) / n)
    half = norm.ppf(0.5 + conf / 2) * se
    return {"rate": float(rate), "se": se, "lo": float(rate - half), "hi": float(rate + half), "n_trials": n}


def mean_estimate(values, conf: float = 0.95) -> dict:
    """Mean of a sample with its standard error and normal CI."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    mean = values.mean()
    se = values.std(ddof=1) / math.sqrt(n)
    half = norm.ppf(0.5 + conf / 2) * se
    return {"mean": float(mean), "se": float(se), "lo": float(mean - half), "hi": float(mean + half), "n": n}
