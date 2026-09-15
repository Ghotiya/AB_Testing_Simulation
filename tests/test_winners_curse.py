import math

import numpy as np
import pytest

from absim.exact import exact_winner_summary
from absim.peeking import StoppingOutcome
from absim.rng import make_rng
from absim.winners_curse import (
    bootstrap_conditional_mean_difference,
    estimate_winners_curse_bias,
    summarize_winners,
    truncated_normal_winner_mean,
)


def _outcome(rejected, d_stop, d_final, n_stop):
    return StoppingOutcome(
        alpha_per_look=0.05,
        rejected=np.array(rejected),
        stop_index=np.zeros(len(rejected), dtype=int),
        n_at_stop=np.array(n_stop),
        delta_hat_at_stop=np.array(d_stop, dtype=float),
        delta_hat_final=np.array(d_final, dtype=float),
    )


def test_summary_conditions_on_significant_and_positive():
    out = _outcome(
        rejected=[True, True, True, False],
        d_stop=[0.03, 0.01, -0.02, 0.05],
        d_final=[0.015, 0.012, -0.01, 0.04],
        n_stop=[100, 300, 200, 400],
    )
    s = summarize_winners(out, true_delta=0.01)
    assert s["n_winners"] == 2 and s["share_winners"] == 0.5
    assert s["wrong_sign_share_of_significant"] == pytest.approx(1 / 3)
    assert s["winners_mean"] == pytest.approx(0.02)
    assert s["winners_rel_bias"] == pytest.approx(1.0)  # 0.02 reported vs 0.01 true
    assert s["significant_mean"] == pytest.approx((0.03 + 0.01 - 0.02) / 3)
    assert s["winners_full_horizon_mean"] == pytest.approx(0.0135)
    assert s["mean_n_at_stop_winners"] == 200


def test_relative_bias_undefined_under_null():
    s = summarize_winners(_outcome([True, True], [0.01, 0.02], [0.0, 0.0], [1, 1]), true_delta=0.0)
    assert s["winners_mean"] == pytest.approx(0.015) and math.isnan(s["winners_rel_bias"])


def test_truncated_normal_theory_matches_exact_enumeration_at_design_n():
    for p_t in (0.104, 0.11, 0.114):
        exact = exact_winner_summary(14_800, 0.10, p_t)["mean_delta_hat_winners"]
        assert truncated_normal_winner_mean(0.10, p_t, 14_800) == pytest.approx(exact, rel=0.01)


def test_fixed_horizon_winners_match_exact_enumeration():
    s = estimate_winners_curse_bias(1, 0.004, rng=make_rng("test_wc"), n_trials=40_000, p_c=0.10, n_per_arm=14_800)
    exact = exact_winner_summary(14_800, 0.10, 0.104)
    assert abs(s["share_winners"] - exact["p_winner"]) < 4 * math.sqrt(exact["p_winner"] * (1 - exact["p_winner"]) / 40_000)
    assert abs(s["winners_mean"] - exact["mean_delta_hat_winners"]) < 4 * s["winners_se"]


def test_bootstrap_difference_of_identical_conditional_means_is_zero():
    rng = np.random.default_rng(1)
    v = rng.normal(size=5_000)
    m = v > 0
    res = bootstrap_conditional_mean_difference(v, m, v, m, rng, n_boot=50)
    assert res["diff"] == 0 and res["se"] == 0


def test_bootstrap_se_close_to_analytic_for_independent_halves():
    rng = np.random.default_rng(2)
    v = rng.normal(size=40_000)
    a = np.arange(v.size) % 2 == 0
    res = bootstrap_conditional_mean_difference(v, a, v, ~a, rng, n_boot=300)
    analytic = math.sqrt(2 / 20_000)
    assert res["se"] == pytest.approx(analytic, rel=0.15)
