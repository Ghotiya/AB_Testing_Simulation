"""G0: scalar/vectorized agreement, nested looks, look schedule, seeding."""

import numpy as np
import pytest

from absim.config import LOOK_COUNTS
from absim.rng import make_rng
from absim.simulate import look_schedule, run_single_trial, run_trials


def test_look_schedule():
    assert look_schedule(14_800, 1).tolist() == [14_800]
    assert look_schedule(14_800, 14)[0] == 1_057
    assert look_schedule(14_800, 28)[0] == 529
    for K in LOOK_COUNTS:
        n = look_schedule(14_800, K)
        assert n[-1] == 14_800 and np.all(np.diff(n) > 0)
        expected = [int(np.floor(k * 14_800 / K + 0.5)) for k in range(1, K + 1)]
        assert n.tolist() == expected


@pytest.mark.parametrize(
    "p_c,p_t,n_per_arm,num_looks",
    [(0.10, 0.10, 14_800, 1), (0.10, 0.11, 14_800, 14), (0.02, 0.02, 200, 28), (0.5, 0.6, 1_000, 7)],
)
def test_scalar_reference_matches_vectorized(p_c, p_t, n_per_arm, num_looks):
    config = dict(p_c=p_c, p_t=p_t, n=n_per_arm, K=num_looks)
    batch = run_trials(make_rng("test_agreement", **config), 300, p_c, p_t, n_per_arm, num_looks)

    rng = make_rng("test_agreement", **config)
    rows = [run_single_trial(rng, p_c, p_t, n_per_arm, num_looks) for _ in range(300)]

    def col(key):
        return np.array([[r[key] for r in trial] for trial in rows])

    np.testing.assert_array_equal(col("x_c"), batch.x_c)
    np.testing.assert_array_equal(col("x_t"), batch.x_t)
    np.testing.assert_array_equal(col("n")[0], batch.n)
    for key in ("delta_hat", "z", "p"):
        np.testing.assert_allclose(col(key), getattr(batch, key), rtol=0, atol=1e-12)


def test_looks_are_nested_on_one_stream():
    N, K, T = 14_800, 28, 20_000
    batch = run_trials(make_rng("test_nested"), T, 0.10, 0.10, N, K)
    increments = np.diff(batch.n, prepend=0)
    for x in (batch.x_c, batch.x_t):
        steps = np.diff(x, axis=1, prepend=0)
        assert np.all(steps >= 0) and np.all(steps <= increments)

    # Nested looks are correlated: corr(x at look 1, x at look K) = sqrt(n_1 / N) = 0.189.
    # Independent fresh draws per look would give 0.
    corr = np.corrcoef(batch.x_c[:, 0], batch.x_c[:, -1])[0, 1]
    assert abs(corr - np.sqrt(batch.n[0] / N)) < 0.03


def test_final_look_has_the_right_marginal():
    N, T = 14_800, 50_000
    batch = run_trials(make_rng("test_marginal"), T, 0.10, 0.11, N, 5)
    for x, p in ((batch.x_c[:, -1], 0.10), (batch.x_t[:, -1], 0.11)):
        assert abs(x.mean() - N * p) < 5 * np.sqrt(N * p * (1 - p) / T)
        assert abs(x.var() / (N * p * (1 - p)) - 1) < 0.03


def test_degenerate_early_looks_do_not_produce_nan():
    batch = run_trials(make_rng("test_degenerate"), 5_000, 0.02, 0.02, 200, 28)
    empty = (batch.x_c + batch.x_t) == 0
    assert empty.any()
    assert not np.isnan(batch.p).any()
    assert np.all(batch.p[empty] == 1.0)


def test_seeding_is_reproducible_per_config():
    a = make_rng("s", K=14, effect=0.01).random(5)
    b = make_rng("s", K=14, effect=0.01).random(5)
    c = make_rng("s", K=14, effect=0.02).random(5)
    d = make_rng("other", K=14, effect=0.01).random(5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c) and not np.array_equal(a, d)
