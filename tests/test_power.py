import pytest

from absim.exact import exact_null_pvalue_distribution, exact_rejection_probability
from absim.power import closed_form_power, power_sweep, required_sample_size


def test_power_under_null_is_alpha():
    assert closed_form_power(0.10, 0.10, 14_800, 0.05) == pytest.approx(0.05, abs=1e-12)


def test_design_sample_size_matches_decision_log():
    assert required_sample_size(0.01, 0.80, 0.05, 0.10) == 14_751


def test_required_sample_size_is_the_smallest_sufficient_n():
    for mde in (0.005, 0.01, 0.02, 0.03):
        n = required_sample_size(mde)
        assert closed_form_power(0.10, 0.10 + mde, n) >= 0.80
        assert closed_form_power(0.10, 0.10 + mde, n - 1) < 0.80


def test_exact_rejection_under_null_equals_exact_size():
    dist = exact_null_pvalue_distribution(500, 0.10)
    assert exact_rejection_probability(500, 0.10, 0.10) == pytest.approx(float(dist.cdf_left(0.05)), abs=1e-9)


def test_exact_power_close_to_closed_form_at_large_n():
    assert exact_rejection_probability(14_800, 0.10, 0.11) == pytest.approx(0.8013, abs=0.003)


def test_power_sweep_grid_and_reproducibility():
    a = power_sweep([0.0, 0.1], [500, 2_000], n_trials=2_000, with_exact=False)
    b = power_sweep([0.1], [2_000], n_trials=2_000, with_exact=False)
    assert len(a) == 4 and set(a.n_per_arm) == {500, 2_000}
    # a single cell re-run alone reproduces the same number
    assert a[(a.relative_lift == 0.1) & (a.n_per_arm == 2_000)].power_sim.item() == b.power_sim.item()
