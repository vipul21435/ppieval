from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from ppieval import (
    Estimate,
    classical_mean,
    estimate_all,
    judge_only_mean,
    optimal_lambda,
    optimal_lambda_from_moments,
    ppi_mean,
    ppi_pp_mean,
    variance_from_moments,
)
from ppieval.estimators import normal_quantile


def biased_binary_sample(
    rng: np.random.Generator,
    n: int,
    n_unl: int,
    p: float = 0.7,
    false_positive: float = 0.4,
    false_negative: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0/1 human labels and a judge that says "pass" too often."""
    total = n + n_unl
    y_all = (rng.random(total) < p).astype(float)
    u = rng.random(total)
    f_all = np.where(y_all == 1.0, u >= false_negative, u < false_positive).astype(float)
    return y_all[:n], f_all[:n], f_all[n:]


def noisy_rating_sample(
    rng: np.random.Generator, n: int, n_unl: int, slope: float = 0.8, noise: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous human ratings and a judge with an offset, a slope and noise."""
    total = n + n_unl
    y_all = rng.normal(3.0, 1.0, total)
    f_all = 0.5 + slope * y_all + rng.normal(0.0, noise, total)
    return y_all[:n], f_all[:n], f_all[n:]


# --- closed forms ---------------------------------------------------------------------------


def test_lambda_zero_is_the_classical_estimate() -> None:
    y, f, fu = noisy_rating_sample(np.random.default_rng(1), 40, 400)
    classical = classical_mean(y, level=0.9)
    at_zero = ppi_mean(y, f, fu, lam=0.0, level=0.9)
    assert at_zero.estimate == classical.estimate
    assert at_zero.se == classical.se
    assert (at_zero.lo, at_zero.hi) == (classical.lo, classical.hi)
    assert at_zero.lam == 0.0


def test_lambda_one_is_the_ppi_estimate() -> None:
    y, f, fu = noisy_rating_sample(np.random.default_rng(2), 50, 700)
    got = ppi_mean(y, f, fu)
    rectifier = f - y
    expected = fu.mean() - rectifier.mean()
    expected_var = fu.var(ddof=1) / fu.size + rectifier.var(ddof=1) / y.size
    assert got.method == "ppi"
    assert got.lam == 1.0
    assert got.estimate == pytest.approx(expected, rel=1e-12)
    assert got.se == pytest.approx(math.sqrt(expected_var), rel=1e-12)


def test_classical_interval_matches_scipy() -> None:
    y = np.random.default_rng(3).normal(2.0, 1.5, 37)
    got = classical_mean(y, level=0.9)
    lo, hi = stats.norm.interval(0.9, loc=y.mean(), scale=stats.sem(y))
    assert got.estimate == pytest.approx(y.mean(), rel=1e-12)
    assert got.se == pytest.approx(stats.sem(y), rel=1e-12)
    assert (got.lo, got.hi) == pytest.approx((lo, hi), rel=1e-9)


@pytest.mark.parametrize("p", [0.005, 0.025, 0.2, 0.5, 0.9, 0.975, 0.9995])
def test_normal_quantile_matches_scipy(p: float) -> None:
    assert normal_quantile(p) == pytest.approx(stats.norm.ppf(p), abs=1e-9)


def test_perfect_judge_moments_give_the_full_sample_variance() -> None:
    sigma2, n, n_unl = 0.21, 80, 920
    lam = optimal_lambda_from_moments(sigma2, sigma2, n, n_unl)
    assert lam == pytest.approx(n_unl / (n + n_unl), rel=1e-12)
    variance = variance_from_moments(sigma2, sigma2, sigma2, n, n_unl, lam)
    assert variance == pytest.approx(sigma2 / (n + n_unl), rel=1e-12)


def test_perfect_judge_estimate_is_the_mean_over_all_rows() -> None:
    rng = np.random.default_rng(4)
    y_all = (rng.random(600) < 0.35).astype(float)
    y, y_rest = y_all[:60], y_all[60:]
    got = ppi_mean(y, y, y_rest, lam=540 / 600)
    assert got.estimate == pytest.approx(y_all.mean(), rel=1e-12)


def test_perfect_judge_ppi_pp_is_close_to_the_full_sample_interval() -> None:
    rng = np.random.default_rng(5)
    y_all = rng.normal(0.0, 2.0, 4400)
    y, y_rest = y_all[:400], y_all[400:]
    got = ppi_pp_mean(y, y, y_rest)
    full = classical_mean(y_all)
    # With f = y the plug-in is N / (n + N) times a ratio of two estimates of the same variance.
    expected_lam = 4000 / 4400 * y.var(ddof=1) / y_all.var(ddof=1)
    assert got.lam == pytest.approx(expected_lam, rel=1e-12)
    assert got.lam == pytest.approx(4000 / 4400, rel=0.1)
    assert got.estimate == pytest.approx(full.estimate, abs=0.5 * full.se)
    assert got.se == pytest.approx(full.se, rel=0.05)


def test_useless_judge_moments_give_the_classical_variance() -> None:
    lam = optimal_lambda_from_moments(1.3, 0.0, 50, 5000)
    assert lam == 0.0
    assert variance_from_moments(0.8, 1.3, 0.0, 50, 5000, lam) == pytest.approx(0.8 / 50)


def test_moment_formulas_with_a_constant_judge_or_no_unlabelled_rows() -> None:
    # A judge with no variance, or no rows for it to add, leaves the classical variance.
    assert optimal_lambda_from_moments(0.0, 0.0, 50, 5000) == 0.0
    assert optimal_lambda_from_moments(1.3, 0.9, 50, 0) == 0.0
    assert variance_from_moments(0.8, 1.3, 0.9, 50, 0, 0.0) == pytest.approx(0.8 / 50)


def test_useless_judge_ppi_pp_approaches_classical_with_many_rows() -> None:
    rng = np.random.default_rng(6)
    y = rng.normal(0.0, 1.0, 20_000)
    f = rng.normal(0.0, 1.0, 20_000)
    fu = rng.normal(0.0, 1.0, 200_000)
    classical = classical_mean(y)
    tuned = ppi_pp_mean(y, f, fu)
    assert tuned.lam == pytest.approx(0.0, abs=0.03)
    assert tuned.se <= classical.se * (1 + 1e-9)
    assert tuned.se == pytest.approx(classical.se, rel=1e-3)
    # PPI without tuning pays for the noise the judge adds.
    assert ppi_mean(y, f, fu).se > 1.3 * classical.se


def test_optimal_lambda_formula() -> None:
    y, f, fu = noisy_rating_sample(np.random.default_rng(7), 64, 512)
    cov = np.cov(y, f, ddof=1)[0, 1]
    var_all = np.concatenate([f, fu]).var(ddof=1)
    expected = cov / ((1 + 64 / 512) * var_all)
    assert optimal_lambda(y, f, fu) == pytest.approx(expected, rel=1e-12)
    assert ppi_pp_mean(y, f, fu).lam == pytest.approx(expected, rel=1e-12)


def test_variance_at_optimal_lambda_closed_form() -> None:
    var_y, var_f, rho, n, n_unl = 2.0, 0.5, 0.6, 120, 880
    cov = rho * math.sqrt(var_y * var_f)
    lam = optimal_lambda_from_moments(var_f, cov, n, n_unl)
    got = variance_from_moments(var_y, var_f, cov, n, n_unl, lam)
    assert got == pytest.approx(var_y / n * (1 - rho**2 * n_unl / (n + n_unl)), rel=1e-12)


# --- properties -----------------------------------------------------------------------------


def test_optimal_lambda_minimises_the_variance() -> None:
    rng = np.random.default_rng(8)
    for _ in range(200):
        var_y, var_f = rng.uniform(0.1, 5.0, 2)
        cov = rng.uniform(-1.0, 1.0) * math.sqrt(var_y * var_f)
        n = int(rng.integers(2, 500))
        n_unl = int(rng.integers(1, 5000))
        lam = optimal_lambda_from_moments(var_f, cov, n, n_unl)
        best = variance_from_moments(var_y, var_f, cov, n, n_unl, lam)
        assert best <= var_y / n + 1e-15
        for step in (-0.5, -0.01, 0.01, 0.5):
            other = variance_from_moments(var_y, var_f, cov, n, n_unl, lam + step)
            assert best < other


def test_a_constant_judge_bias_does_not_move_ppi() -> None:
    y, f, fu = biased_binary_sample(np.random.default_rng(9), 80, 800)
    for lam in (1.0, 0.4, -0.3):
        base = ppi_mean(y, f, fu, lam=lam)
        shifted = ppi_mean(y, f + 10.0, fu + 10.0, lam=lam)
        assert shifted.estimate == pytest.approx(base.estimate, abs=1e-9)
        assert shifted.se == pytest.approx(base.se, rel=1e-9)
    assert ppi_pp_mean(y, f + 10.0, fu + 10.0).estimate == pytest.approx(
        ppi_pp_mean(y, f, fu).estimate, abs=1e-9
    )
    # The judge-only estimate moves by the whole bias.
    moved = judge_only_mean(np.concatenate([f, fu]) + 10.0).estimate
    assert moved == pytest.approx(judge_only_mean(np.concatenate([f, fu])).estimate + 10.0)


def test_estimates_follow_a_shift_and_scale_of_the_labels() -> None:
    y, f, fu = noisy_rating_sample(np.random.default_rng(10), 70, 300)
    base = ppi_pp_mean(y, f, fu)
    moved = ppi_pp_mean(2.5 * y + 4.0, f, fu)
    assert moved.estimate == pytest.approx(2.5 * base.estimate + 4.0, rel=1e-9)
    assert moved.se == pytest.approx(2.5 * base.se, rel=1e-9)
    assert moved.lam == pytest.approx(2.5 * base.lam, rel=1e-9)


def test_negatively_correlated_judge_gets_a_negative_lambda() -> None:
    rng = np.random.default_rng(11)
    y_all = (rng.random(1000) < 0.6).astype(float)
    y, y_rest = y_all[:100], y_all[100:]
    tuned = ppi_pp_mean(y, 1.0 - y, 1.0 - y_rest)
    assert tuned.lam is not None and tuned.lam < -0.7
    assert tuned.se < 0.5 * classical_mean(y).se
    exact = ppi_mean(y, 1.0 - y, 1.0 - y_rest, lam=-900 / 1000)
    assert exact.estimate == pytest.approx(y_all.mean(), rel=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_ppi_pp_interval_is_not_wider_than_human_only(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(30, 200))
    n_unl = int(rng.integers(200, 2000))
    y, f, fu = biased_binary_sample(rng, n, n_unl, p=float(rng.uniform(0.2, 0.8)))
    assert ppi_pp_mean(y, f, fu).width <= classical_mean(y).width * (1 + 1e-12)


# --- simulated coverage ---------------------------------------------------------------------


def test_simulated_coverage_with_a_biased_judge() -> None:
    """2,000 draws, n = 100 labelled, N = 1,000 unlabelled, nominal level 95 %.

    With 2,000 draws the standard error of a coverage estimate near 0.95 is
    about 0.005, so the valid estimators must land inside 0.95 +/- 0.02.
    """
    rng = np.random.default_rng(2026)
    truth, reps = 0.7, 2000
    hits = {"human-only": 0, "judge-only": 0, "ppi": 0, "ppi++": 0}
    widths = {name: 0.0 for name in hits}
    for _ in range(reps):
        y, f, fu = biased_binary_sample(rng, 100, 1000, p=truth)
        for est in estimate_all(y, f, fu):
            hits[est.method] += est.lo <= truth <= est.hi
            widths[est.method] += est.width / reps
    coverage = {name: count / reps for name, count in hits.items()}
    for name in ("human-only", "ppi", "ppi++"):
        assert 0.93 <= coverage[name] <= 0.97, (name, coverage[name])
    assert coverage["judge-only"] < 0.05
    assert widths["ppi++"] < widths["human-only"]
    assert widths["ppi++"] <= widths["ppi"]


# --- behaviour at the edges -----------------------------------------------------------------


def test_estimate_all_order_and_counts() -> None:
    y, f, fu = biased_binary_sample(np.random.default_rng(12), 30, 270)
    got = estimate_all(y, f, fu, level=0.8)
    assert [e.method for e in got] == ["human-only", "judge-only", "ppi", "ppi++"]
    assert all(isinstance(e, Estimate) and e.level == 0.8 for e in got)
    assert [(e.n_human, e.n_judge) for e in got] == [(30, 0), (0, 300), (30, 300), (30, 300)]
    assert [e.lam is None for e in got] == [True, True, False, False]


def test_without_unlabelled_rows_ppi_pp_is_the_classical_estimate() -> None:
    y, f, _ = biased_binary_sample(np.random.default_rng(13), 50, 0)
    got = estimate_all(y, f, [])
    assert [e.method for e in got] == ["human-only", "judge-only", "ppi++"]
    assert got[2].lam == 0.0
    assert (got[2].estimate, got[2].se) == (got[0].estimate, got[0].se)
    with pytest.raises(ValueError, match="without a human label"):
        ppi_mean(y, f, [])


def test_constant_judge_gets_lambda_zero() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    assert optimal_lambda(y, np.ones(5), np.ones(20)) == 0.0
    assert ppi_pp_mean(y, np.ones(5), np.ones(20)).estimate == pytest.approx(0.6)


def test_to_dict_includes_the_width() -> None:
    est = classical_mean([0.0, 1.0, 1.0, 1.0])
    as_dict = est.to_dict()
    assert as_dict["method"] == "human-only"
    assert as_dict["width"] == pytest.approx(est.hi - est.lo)
    assert as_dict["lam"] is None


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: classical_mean([1.0]), "at least 2 labelled rows"),
        (lambda: classical_mean([1.0, float("nan")]), "non-finite"),
        (lambda: classical_mean([[1.0, 2.0]]), "one-dimensional"),
        (lambda: classical_mean([1.0, 2.0], level=1.0), "level"),
        (lambda: judge_only_mean([0.5]), "at least 2 judge scores"),
        (lambda: ppi_mean([1.0, 0.0], [1.0], [1.0, 0.0]), "same length"),
        (lambda: ppi_mean([1.0], [1.0], [1.0, 0.0]), "at least 2 labelled rows"),
        (lambda: ppi_mean([1.0, 0.0], [1.0, 0.0], [1.0, 0.0], lam=float("inf")), "finite"),
        (lambda: optimal_lambda([1.0, 0.0], [1.0, float("inf")], [1.0, 0.0]), "non-finite"),
        (lambda: variance_from_moments(1.0, 1.0, 0.5, 0, 10, 0.0), "n must be at least 1"),
        (lambda: variance_from_moments(1.0, 1.0, 0.5, 5, -1, 0.0), "must not be negative"),
        (lambda: variance_from_moments(1.0, 1.0, 0.5, 5, 0, 1.0), "lam must be 0"),
        (lambda: normal_quantile(0.0), "strictly between 0 and 1"),
    ],
)
def test_invalid_input_is_rejected(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
