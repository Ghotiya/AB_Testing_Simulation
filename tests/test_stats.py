"""G0: absim.stats agrees with statsmodels, and degenerate looks never produce NaN."""

import warnings

import numpy as np
import pytest
from statsmodels.stats.proportion import proportions_ztest

from absim.stats import two_proportion_ztest


def _random_inputs(rng, m):
    """m inputs: a third uniform counts, a third realistic ~10% rates, a third edge cases."""
    n_c = rng.integers(1, 50_001, m)
    n_t = rng.integers(1, 50_001, m)
    x_c = rng.integers(0, n_c + 1)
    x_t = rng.integers(0, n_t + 1)

    third = m // 3
    realistic = slice(third, 2 * third)
    x_c[realistic] = rng.binomial(n_c[realistic], 0.10)
    x_t[realistic] = rng.binomial(n_t[realistic], 0.11)

    edge = np.arange(2 * third, m)
    small = edge[::5]  # tiny samples, where degenerate pooled rates are common
    n_c[small] = rng.integers(1, 6, small.size)
    n_t[small] = rng.integers(1, 6, small.size)
    x_c[small] = rng.integers(0, n_c[small] + 1)
    x_t[small] = rng.integers(0, n_t[small] + 1)
    x_c[edge[1::5]] = 0  # control has no successes
    x_t[edge[2::5]] = n_t[edge[2::5]]  # treatment is all successes
    x_c[edge[3::5]], x_t[edge[3::5]] = 0, 0  # pooled rate 0
    x_c[edge[4::5]], x_t[edge[4::5]] = n_c[edge[4::5]], n_t[edge[4::5]]  # pooled rate 1
    return x_c, n_c, x_t, n_t


def test_matches_statsmodels_on_randomized_inputs():
    rng = np.random.default_rng(0)
    x_c, n_c, x_t, n_t = _random_inputs(rng, 10_000)
    ours = two_proportion_ztest(x_c, n_c, x_t, n_t)

    n_degenerate = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels divides by zero on degenerate inputs
        for i in range(x_c.size):
            z_ref, p_ref = proportions_ztest([x_t[i], x_c[i]], [n_t[i], n_c[i]])
            pooled = (x_c[i] + x_t[i]) / (n_c[i] + n_t[i])
            if pooled in (0.0, 1.0):
                n_degenerate += 1
                assert np.isnan(p_ref)
                assert ours.p[i] == 1.0 and ours.z[i] == 0.0
            else:
                assert abs(ours.p[i] - p_ref) < 1e-10, i
                assert abs(ours.z[i] - z_ref) < 1e-10, i
    assert n_degenerate > 500  # the edge cases were actually exercised


@pytest.mark.parametrize("x_c,x_t", [(0, 0), (100, 100)])
def test_degenerate_pooled_rate_is_not_significant(x_c, x_t):
    res = two_proportion_ztest(x_c, 100, x_t, 100)
    assert res.p == 1.0 and res.z == 0.0


def test_degenerate_inputs_raise_no_warnings_and_no_nan():
    x_c = np.array([0, 5, 0, 7])
    x_t = np.array([0, 5, 3, 7])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = two_proportion_ztest(x_c, 7, x_t, 7)
    assert not np.isnan(res.p).any() and not np.isnan(res.z).any()
    np.testing.assert_array_equal(res.p[[0, 3]], [1.0, 1.0])


def test_sign_convention_is_treatment_minus_control():
    res = two_proportion_ztest(100, 1000, 150, 1000)
    assert res.delta_hat == pytest.approx(0.05) and res.z > 0
