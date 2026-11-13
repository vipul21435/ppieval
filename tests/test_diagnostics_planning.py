"""Diagnostics, planning and sampling."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from ppieval.diagnostics import diagnose, ks_two_sample
from ppieval.estimators import (
    classical_mean,
    normal_quantile,
    optimal_lambda_from_moments,
    ppi_pp_mean,
    variance_from_moments,
)
from ppieval.planning import (
    effective_sample_size,
    effective_sample_size_from_correlation,
    labels_needed,
    make_plan,
    sample_to_label,
)


@pytest.mark.parametrize("seed", range(6))
def test_ks_statistic_and_p_value_match_scipy(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, 150) if seed % 2 else (rng.random(150) < 0.5).astype(float)
    b = rng.normal(0.3, 1.0, 400) if seed % 2 else (rng.random(400) < 0.62).astype(float)
    statistic, pvalue = ks_two_sample(a, b)
    reference = stats.ks_2samp(a, b, method="asymp")
    assert statistic == pytest.approx(reference.statistic, abs=1e-12)
    limit = stats.kstwobign.sf(statistic * math.sqrt(150 * 400 / 550))
    assert pvalue == pytest.approx(limit, abs=1e-9)
    assert pvalue == pytest.approx(reference.pvalue, abs=0.03)


def test_ks_edge_cases() -> None:
    assert ks_two_sample([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == (0.0, 1.0)
    with pytest.raises(ValueError, match="at least one value"):
        ks_two_sample([], [1.0])


def test_diagnostics_on_a_hand_worked_binary_table() -> None:
    y = np.array([1.0] * 40 + [0.0] * 60)
    f = np.array([1.0] * 36 + [0.0] * 4 + [1.0] * 12 + [0.0] * 48)
    fu = np.array([1.0] * 240 + [0.0] * 260)
    got = diagnose(y, f, fu)
    assert (got.n_labelled, got.n_unlabelled) == (100, 500)
    assert got.agreement == pytest.approx(0.84)
    assert got.sensitivity == pytest.approx(0.9)
    assert got.specificity == pytest.approx(0.8)
    assert got.correlation == pytest.approx(np.corrcoef(y, f)[0, 1])
    reference = classical_mean(f - y)
    assert got.rectifier.method == "rectifier"
    assert got.rectifier.estimate == pytest.approx(0.08)
    assert (got.rectifier.lo, got.rectifier.hi) == (reference.lo, reference.hi)
    assert got.ks_statistic == pytest.approx(0.0, abs=1e-12)
    assert got.warnings == ()
    assert got.to_dict()["rectifier"]["method"] == "rectifier"


def test_diagnostics_warn_about_a_labelled_subset_picked_by_judge_score() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 2000)
    f = y + rng.normal(0.0, 0.5, 2000)
    top = np.argsort(f)[-200:]
    rest = np.setdiff1d(np.arange(2000), top)
    picked = diagnose(y[top], f[top], f[rest])
    assert picked.ks_pvalue is not None and picked.ks_pvalue < 1e-6
    assert any("may not be a random subset" in text for text in picked.warnings)
    assert picked.agreement is None and picked.sensitivity is None
    random = diagnose(y[:200], f[:200], f[200:])
    assert random.warnings == ()


def test_diagnostics_other_warnings() -> None:
    small = diagnose([1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [])
    assert small.correlation is None and small.ks_statistic is None
    assert any("constant" in text for text in small.warnings)
    assert any("fewer than 30" in text for text in small.warnings)
    y = np.array([1.0, 0.0] * 20)
    assert any("not positively correlated" in t for t in diagnose(y, 1.0 - y, y).warnings)


def test_effective_sample_size_closed_forms() -> None:
    assert effective_sample_size_from_correlation(100, 900, 0.0) == 100
    assert effective_sample_size_from_correlation(100, 900, 1.0) == pytest.approx(1000)
    assert effective_sample_size_from_correlation(100, 900, 0.8) == pytest.approx(
        100 / (1 - 0.64 * 0.9)
    )
    assert effective_sample_size_from_correlation(5, 0, 1.0) == 5
    with pytest.raises(ValueError):
        effective_sample_size_from_correlation(0, 10, 0.5)


def test_effective_sample_size_from_data() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 1.0, 5000)
    f = 0.8 * y + rng.normal(0.0, 0.6, 5000)
    got = effective_sample_size(y[:500], f[:500], f[500:])
    ratio = classical_mean(y[:500]).se / ppi_pp_mean(y[:500], f[:500], f[500:]).se
    assert got == pytest.approx(500 * ratio**2)
    rho = np.corrcoef(y[:500], f[:500])[0, 1]
    assert got == pytest.approx(effective_sample_size_from_correlation(500, 4500, rho), rel=0.1)
    assert effective_sample_size([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0]) == math.inf


def test_labels_needed_solves_the_variance_formula() -> None:
    with_judge, human_only = labels_needed(0.1, 0.25, 0.8, 10_000)
    z = normal_quantile(0.975)
    assert human_only == math.ceil(0.25 * (2 * z / 0.1) ** 2) == 385
    assert with_judge is not None and with_judge < human_only

    def width(n: int) -> float:
        cov = 0.8 * 0.25
        lam = optimal_lambda_from_moments(0.25, cov, n, 10_000 - n)
        return 2 * z * math.sqrt(variance_from_moments(0.25, 0.25, cov, n, 10_000 - n, lam))

    assert width(with_judge) <= 0.1 < width(with_judge - 1)
    assert labels_needed(0.1, 0.25, 0.0, 10_000) == (385, 385)
    assert labels_needed(0.01, 0.25, 0.9, 1000) == (None, math.ceil(0.25 * (2 * z / 0.01) ** 2))
    # With a perfect judge the variance is var_y / total_rows whatever n is: the minimum, 2.
    assert labels_needed(0.1, 0.25, 1.0, 400)[0] == 2
    # A target that only labelling every row reaches.
    assert labels_needed(0.098, 0.25, 0.0, 400) == (400, 400)
    with pytest.raises(ValueError):
        labels_needed(0.0, 0.25, 0.5, 100)


def test_make_plan_reports_current_widths_and_targets() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.6).astype(float)
    f = np.where(rng.random(2000) < 0.85, y, 1 - y)
    plan = make_plan(y[:200], f[:200], f[200:], target_width=0.08)
    assert (plan.n_labelled, plan.n_unlabelled) == (200, 1800)
    assert plan.width_ppi_pp < plan.width_human_only
    assert plan.effective_n > 200
    assert plan.labels_with_judge is not None and plan.labels_with_judge < plan.labels_human_only
    assert plan.to_dict()["target_width"] == 0.08
    flat = make_plan([1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0])
    assert flat.correlation == 0.0 and flat.labels_with_judge is None


def test_uniform_sample() -> None:
    judge = np.random.default_rng(4).random(500)
    eligible = np.arange(500) >= 100
    got = sample_to_label(judge, 80, seed=3, eligible=eligible)
    assert got.selected.sum() == 80 and not got.selected[:100].any()
    assert np.all(got.probability[100:] == 0.2) and np.all(got.probability[:100] == 0.0)
    assert np.all(got.weight[got.selected] == 5.0)
    assert np.allclose(got.weight[eligible & ~got.selected], 1.25)
    assert np.isnan(got.weight[:100]).all()
    again = sample_to_label(judge, 80, seed=3, eligible=eligible)
    assert np.array_equal(again.selected, got.selected)


@pytest.mark.parametrize("allocation", ["proportional", "equal"])
def test_stratified_sample_weights_add_up_to_the_pool(allocation: str) -> None:
    judge = np.random.default_rng(5).normal(0.0, 1.0, 1000)
    got = sample_to_label(judge, 100, method="stratified", n_strata=4, allocation=allocation)
    assert got.selected.sum() == 100
    counts = np.bincount(got.stratum[got.selected])
    assert list(counts) == [25, 25, 25, 25]
    assert got.weight[got.selected].sum() == pytest.approx(1000)
    assert np.allclose(got.probability, counts[got.stratum] / np.bincount(got.stratum)[got.stratum])


def test_equal_allocation_oversamples_a_rare_judge_score() -> None:
    judge = np.array([0.0] * 100 + [1.0] * 900)
    got = sample_to_label(judge, 150, method="stratified", allocation="equal", seed=1)
    assert got.selected[:100].sum() == 75 and got.selected[100:].sum() == 75
    assert got.weight[got.selected].sum() == pytest.approx(1000)
    # A stratum cannot give more rows than it has.
    most = sample_to_label(judge, 400, method="stratified", allocation="equal")
    assert most.selected[:100].sum() == 100 and most.selected[100:].sum() == 300
    with pytest.raises(ValueError, match="size must be between"):
        sample_to_label(judge, 1001)
    with pytest.raises(ValueError, match="method is uniform or stratified"):
        sample_to_label(judge, 10, method="quota")
    with pytest.raises(ValueError, match="one entry per row"):
        sample_to_label(judge, 10, eligible=[True])
