import numpy as np
import pytest
from scipy.stats import binom

from absim.exact import exact_null_pvalue_distribution, ks_against
from absim.rng import make_rng
from absim.simulate import run_trials


def test_exact_distribution_is_a_distribution_with_the_tie_atom():
    n, p = 500, 0.1
    dist = exact_null_pvalue_distribution(n, p)
    assert dist.cdf[-1] == pytest.approx(1.0, abs=1e-12)
    tie_mass = (binom.pmf(np.arange(n + 1), n, p) ** 2).sum()
    assert 1 - dist.cdf_left(1.0) == pytest.approx(tie_mass, rel=1e-9)


def test_simulated_small_n_null_matches_exact_distribution():
    n, p = 200, 0.02
    sim = run_trials(make_rng("test_exact"), 50_000, p, p, n).p[:, 0]
    dist = exact_null_pvalue_distribution(n, p)
    assert ks_against(sim, dist)["pvalue"] > 0.01
    # and a wrong reference must be rejected, so the check can fail
    assert ks_against(sim, exact_null_pvalue_distribution(n, 0.05))["pvalue"] < 1e-6
