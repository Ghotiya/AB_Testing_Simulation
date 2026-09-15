import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from absim.config import LOOK_COUNTS
from absim.peeking import apply_stopping, per_look_alpha, run_with_peeking
from absim.rng import make_rng
from absim.simulate import TrialBatch, look_schedule, run_trials, run_trials_at_looks, union_look_schedule
from absim.theory import gaussian_crossing_probability, pocock_constant

Z = norm.ppf(0.975)


def _batch(p):
    """A hand-built batch with the given (T, K) p-values."""
    p = np.asarray(p, dtype=float)
    T, K = p.shape
    zeros = np.zeros((T, K))
    delta = np.tile(np.arange(1, K + 1) / 100, (T, 1))
    return TrialBatch(n=np.arange(1, K + 1) * 100, x_c=zeros, x_t=zeros, z=zeros, p=p, delta_hat=delta)


def test_stops_at_first_crossing_and_runs_to_horizon_otherwise():
    out = apply_stopping(_batch([[0.5, 0.01, 0.001], [0.5, 0.2, 0.3], [0.04, 0.9, 0.9]]), alpha=0.05)
    assert out.rejected.tolist() == [True, False, True]
    assert out.stop_index.tolist() == [1, 2, 0]
    assert out.n_at_stop.tolist() == [200, 300, 100]
    np.testing.assert_allclose(out.delta_hat_at_stop, [0.02, 0.03, 0.01])
    np.testing.assert_allclose(out.delta_hat_final, [0.03, 0.03, 0.03])


def test_bonferroni_threshold():
    assert per_look_alpha(14, 0.05, "bonferroni") == pytest.approx(0.05 / 14)
    out = apply_stopping(_batch([[0.03, 0.03]]), alpha=0.05, correction="bonferroni")
    assert not out.rejected[0]  # 0.03 < 0.05 but not < 0.025
    assert apply_stopping(_batch([[0.03, 0.03]]), alpha=0.05).rejected[0]
    with pytest.raises(ValueError):
        per_look_alpha(3, 0.05, "obrien-fleming")


@pytest.mark.parametrize("K,published", [(2, 2.178), (3, 2.289), (4, 2.361), (5, 2.413), (10, 2.555)])
def test_pocock_constant_matches_published_table(K, published):
    assert abs(pocock_constant(K) - published) < 0.003  # Pocock (1977), two-sided alpha = 0.05


def test_pocock_threshold_lies_between_uncorrected_and_bonferroni():
    assert per_look_alpha(1, 0.05, "pocock") == pytest.approx(0.05)
    for K in (2, 5, 14, 28):
        assert 0.05 / K < per_look_alpha(K, 0.05, "pocock") < 0.05


def test_single_look_is_the_fixed_horizon_test():
    batch = run_trials(make_rng("test_k1"), 20_000, 0.1, 0.1, 14_800, 1)
    out = apply_stopping(batch)
    np.testing.assert_array_equal(out.rejected, batch.p[:, 0] < 0.05)
    assert np.all(out.n_at_stop == 14_800)


def test_run_with_peeking_matches_manual_composition():
    kwargs = dict(n_trials=2_000, p_c=0.1, p_t=0.11, n_per_arm=14_800)
    out = run_with_peeking(14, "bonferroni", rng=make_rng("test_rwp"), **kwargs)
    manual = apply_stopping(run_trials(make_rng("test_rwp"), 2_000, 0.1, 0.11, 14_800, 14), 0.05, "bonferroni")
    np.testing.assert_array_equal(out.rejected, manual.rejected)
    np.testing.assert_array_equal(out.delta_hat_at_stop, manual.delta_hat_at_stop)


def test_selecting_from_union_stream_equals_retesting_those_counts():
    union = union_look_schedule(14_800, LOOK_COUNTS)
    full = run_trials_at_looks(make_rng("test_union"), 500, 0.1, 0.1, union)
    for K in LOOK_COUNTS:
        sub = full.select_looks(look_schedule(14_800, K))
        assert sub.n.tolist() == look_schedule(14_800, K).tolist()
        np.testing.assert_array_equal(sub.x_c[:, -1], full.x_c[:, -1])
    with pytest.raises(ValueError):
        full.select_looks([123])


def test_gaussian_theory_single_look_is_alpha():
    assert abs(gaussian_crossing_probability([1.0], Z) - 0.05) < 1e-4


def test_gaussian_theory_two_looks_matches_bivariate_normal():
    rho = np.sqrt(0.5)
    inside = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([Z, Z], lower_limit=[-Z, -Z])
    assert abs(gaussian_crossing_probability([0.5, 1.0], Z) - (1 - inside)) < 2e-4


@pytest.mark.parametrize("K,published", [(2, 0.083), (3, 0.107), (5, 0.142), (10, 0.193), (20, 0.248)])
def test_gaussian_theory_matches_armitage_table(K, published):
    t = np.arange(1, K + 1) / K
    assert abs(gaussian_crossing_probability(t, Z) - published) < 1.5e-3  # table is rounded to 3 dp
