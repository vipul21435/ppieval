"""Stratified PPI, Rogan-Gladen, the bootstrap, weights and clusters."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ppieval.estimators import (
    bootstrap_interval,
    classical_mean,
    estimate_all,
    normal_quantile,
    optimal_lambda,
    ppi_mean,
    ppi_pp_mean,
    rogan_gladen_mean,
    stratified_ppi_mean,
)


def binary_sample(
    rng: np.random.Generator, n: int, n_unl: int, sens: float = 0.9, spec: float = 0.7
):
    total = n + n_unl
    y = (rng.random(total) < 0.6).astype(float)
    draw = rng.random(total)
    f = np.where(y == 1.0, draw < sens, draw >= spec).astype(float)
    return y[:n], f[:n], f[n:]


def rating_sample(rng: np.random.Generator, n: int, n_unl: int):
    total = n + n_unl
    y = rng.normal(3.0, 1.0, total)
    f = 0.4 + 0.9 * y + rng.normal(0.0, 0.5, total)
    return y[:n], f[:n], f[n:]


# --- stratified PPI ---------------------------------------------------------------------------


def test_stratified_with_a_binary_judge_is_the_post_stratified_mean() -> None:
    y, f, fu = binary_sample(np.random.default_rng(1), 200, 1800)
    got = stratified_ppi_mean(y, f, fu)
    total = 2000
    shares, means, within = [], [], []
    for value in (0.0, 1.0):
        labels = y[f == value]
        shares.append((labels.size + np.sum(fu == value)) / total)
        means.append(labels.mean())
        within.append(labels.var(ddof=1) / labels.size)
    w, m, v = np.array(shares), np.array(means), np.array(within)
    estimate = np.sum(w * m)
    variance = np.sum(w**2 * v) + np.sum(w * (m - estimate) ** 2) / total
    assert got.method == "ppi-stratified"
    assert got.estimate == pytest.approx(estimate, rel=1e-12)
    assert got.se == pytest.approx(math.sqrt(variance), rel=1e-12)


def test_one_stratum_is_ppi_pp() -> None:
    y, f, fu = rating_sample(np.random.default_rng(2), 80, 700)
    one = stratified_ppi_mean(y, f, fu, n_strata=1)
    tuned = ppi_pp_mean(y, f, fu)
    assert (one.estimate, one.se) == pytest.approx((tuned.estimate, tuned.se), rel=1e-12)
    # Too few labels for more than one stratum: the strata are merged into one.
    merged = stratified_ppi_mean(y[:15], f[:15], fu, n_strata=5, min_labelled=20)
    assert merged.estimate == pytest.approx(ppi_pp_mean(y[:15], f[:15], fu).estimate, rel=1e-12)


def test_stratified_coverage_on_a_judge_whose_error_depends_on_its_score() -> None:
    rng = np.random.default_rng(3)
    hits, reps, truth = 0, 600, 3.0
    for _ in range(reps):
        y = rng.normal(truth, 1.0, 1200)
        f = y + rng.normal(0.0, 0.2 + 0.4 * (y > truth), 1200)
        est = stratified_ppi_mean(y[:150], f[:150], f[150:])
        hits += est.lo <= truth <= est.hi
    assert 0.92 <= hits / reps <= 0.98


def test_stratified_rejects_bad_settings() -> None:
    y, f, fu = rating_sample(np.random.default_rng(4), 30, 100)
    with pytest.raises(ValueError, match="n_strata"):
        stratified_ppi_mean(y, f, fu, n_strata=0)


# --- Rogan-Gladen -----------------------------------------------------------------------------


def test_rogan_gladen_hand_worked_table() -> None:
    # 40 human 1s, judge says 1 on 36; 60 human 0s, judge says 1 on 12; 1,000 unlabelled, 500 ones.
    y = np.array([1.0] * 40 + [0.0] * 60)
    f = np.array([1.0] * 36 + [0.0] * 4 + [1.0] * 12 + [0.0] * 48)
    fu = np.array([1.0] * 500 + [0.0] * 500)
    got = rogan_gladen_mean(y, f, fu)
    sens, spec, p = 0.9, 0.8, 0.5
    prevalence = (p + spec - 1) / (sens + spec - 1)
    variance = (
        p * (1 - p) / 1000 + prevalence**2 * sens * (1 - sens) / 40
        + (1 - prevalence) ** 2 * spec * (1 - spec) / 60
    ) / (sens + spec - 1) ** 2
    assert got.estimate == pytest.approx(3 / 7, rel=1e-12)
    assert got.estimate == pytest.approx(prevalence, rel=1e-12)
    assert got.se == pytest.approx(math.sqrt(variance), rel=1e-12)
    z = normal_quantile(0.975)
    assert (got.lo, got.hi) == pytest.approx((prevalence - z * got.se, prevalence + z * got.se))


def test_rogan_gladen_is_truncated_to_the_unit_interval() -> None:
    y = np.array([1.0] * 40 + [0.0] * 60)
    f = np.array([1.0] * 36 + [0.0] * 4 + [1.0] * 12 + [0.0] * 48)
    got = rogan_gladen_mean(y, f, np.array([1.0] * 95 + [0.0] * 5))
    assert got.estimate == 1.0 and got.hi == 1.0 and got.lo < 1.0


@pytest.mark.parametrize(
    ("y", "f", "fu", "message"),
    [
        ([1.0, 0.0, 1.0], [0.9, 0.1, 1.0], [1.0, 0.0], "0/1 labels"),
        ([1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0], "both classes"),
        ([1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0], "no better than chance"),
    ],
)
def test_rogan_gladen_rejects_what_it_cannot_correct(y, f, fu, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rogan_gladen_mean(y, f, fu)


# --- bootstrap --------------------------------------------------------------------------------


def test_bootstrap_agrees_with_the_clt_interval_and_is_seeded() -> None:
    y, f, fu = rating_sample(np.random.default_rng(5), 200, 800)
    clt = ppi_pp_mean(y, f, fu)
    boot = bootstrap_interval(ppi_pp_mean, y, f, fu, n_boot=1500, seed=1)
    assert boot.method == "ppi++ (bootstrap)"
    assert boot.estimate == clt.estimate
    assert boot.se == pytest.approx(clt.se, rel=0.1)
    assert (boot.lo, boot.hi) == pytest.approx((clt.lo, clt.hi), abs=0.25 * clt.se)
    again = bootstrap_interval(ppi_pp_mean, y, f, fu, n_boot=1500, seed=1)
    assert (again.lo, again.hi) == (boot.lo, boot.hi)
    other = bootstrap_interval(ppi_pp_mean, y, f, fu, n_boot=1500, seed=2)
    assert (other.lo, other.hi) != (boot.lo, boot.hi)


def test_cluster_bootstrap_is_wider_when_rows_repeat_within_clusters() -> None:
    y, f, fu = rating_sample(np.random.default_rng(6), 60, 240)
    ids, ids_u = np.repeat(np.arange(60), 4), np.repeat(np.arange(60, 300), 4)
    yy, ff, ffu = np.repeat(y, 4), np.repeat(f, 4), np.repeat(fu, 4)
    rows = bootstrap_interval(ppi_pp_mean, yy, ff, ffu, n_boot=400, seed=0)
    groups = bootstrap_interval(
        ppi_pp_mean, yy, ff, ffu, n_boot=400, seed=0, clusters=ids, clusters_unlabelled=ids_u
    )
    assert groups.se > 1.6 * rows.se
    with pytest.raises(ValueError, match="clusters are needed"):
        bootstrap_interval(ppi_pp_mean, yy, ff, ffu, n_boot=50, clusters=ids)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_interval(ppi_pp_mean, y, f, fu, n_boot=5)


def test_bootstrap_fails_loudly_when_the_estimator_keeps_failing() -> None:
    def broken(_y, _f, _fu, calls=[0]):  # noqa: B006
        calls[0] += 1
        if calls[0] > 1:
            raise ValueError("no")
        return classical_mean(_y)

    with pytest.raises(ValueError, match="more than a tenth"):
        bootstrap_interval(broken, [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0], n_boot=20)


# --- clusters ---------------------------------------------------------------------------------


def test_repeating_rows_inside_clusters_does_not_shrink_the_clustered_interval() -> None:
    y = np.random.default_rng(7).normal(0.0, 1.0, 50)
    repeated, ids = np.repeat(y, 3), np.repeat(np.arange(50), 3)
    assert classical_mean(repeated, clusters=ids).se == pytest.approx(classical_mean(y).se)
    assert classical_mean(repeated).se < 0.6 * classical_mean(y).se
    # One cluster per row is the ordinary variance.
    assert classical_mean(y, clusters=np.arange(50)).se == pytest.approx(classical_mean(y).se)


def test_clustered_ppi_matches_the_plain_one_on_duplicated_data() -> None:
    y, f, fu = rating_sample(np.random.default_rng(8), 120, 480)
    ids, ids_u = np.arange(120), np.arange(120, 600)
    plain = ppi_mean(y, f, fu, lam=0.6)
    one_per_row = ppi_mean(y, f, fu, lam=0.6, clusters=ids, clusters_unlabelled=ids_u)
    assert one_per_row.estimate == pytest.approx(plain.estimate, rel=1e-12)
    assert one_per_row.se == pytest.approx(plain.se, rel=0.01)
    doubled = ppi_pp_mean(
        np.repeat(y, 2), np.repeat(f, 2), np.repeat(fu, 2),
        clusters=np.repeat(ids, 2), clusters_unlabelled=np.repeat(ids_u, 2),
    )
    # Close, not equal: G / (G - 1) replaces n / (n - 1), and with clusters lambda minimises
    # the estimated variance, which takes the judge's variance from the labelled rows.
    assert doubled.se == pytest.approx(ppi_pp_mean(y, f, fu).se, rel=0.05)
    assert doubled.lam == pytest.approx(ppi_pp_mean(y, f, fu).lam, rel=0.25)
    with pytest.raises(ValueError, match="clusters are needed"):
        ppi_mean(y, f, fu, clusters=ids)
    with pytest.raises(ValueError, match="clusters are needed"):
        optimal_lambda(y, f, fu, clusters=ids)
    with pytest.raises(ValueError, match="at least 2 clusters"):
        classical_mean(y, clusters=["a"] * 120)
    with pytest.raises(ValueError, match="one entry per row"):
        classical_mean(y, clusters=["a", "b"])


def test_estimate_all_with_clusters_keeps_the_methods_that_support_them() -> None:
    y, f, fu = binary_sample(np.random.default_rng(9), 60, 300)
    ids, ids_u = np.arange(60) // 2, np.arange(300) // 2
    got = estimate_all(y, f, fu, clusters=ids, clusters_unlabelled=ids_u)
    assert [e.method for e in got] == ["human-only", "judge-only", "ppi", "ppi++"]
    boot = estimate_all(y, f, fu, n_boot=30, seed=1)
    assert [e.method for e in boot][6:] == [
        f"{name} (bootstrap)"
        for name in ("human-only", "ppi", "ppi++", "ppi-stratified", "rogan-gladen")
    ]


# --- weights ----------------------------------------------------------------------------------


def test_equal_weights_change_nothing() -> None:
    y, f, fu = rating_sample(np.random.default_rng(10), 70, 400)
    for lam in (1.0, 0.5):
        plain = ppi_mean(y, f, fu, lam=lam)
        weighted = ppi_mean(
            y, f, fu, lam=lam, weights=np.full(70, 7.0), weights_unlabelled=np.full(400, 0.3)
        )
        assert weighted.estimate == pytest.approx(plain.estimate, rel=1e-12)
        assert weighted.se == pytest.approx(plain.se, rel=1e-12)
    assert classical_mean(y, weights=np.full(70, 2.0)).se == pytest.approx(classical_mean(y).se)
    with pytest.raises(ValueError, match="must be positive"):
        classical_mean(y, weights=np.zeros(70))
    with pytest.raises(ValueError, match="one entry per row"):
        classical_mean(y, weights=np.ones(3))
    with pytest.raises(ValueError, match="bootstrap is not available with weights"):
        estimate_all(y, f, fu, weights=np.ones(70), n_boot=20)


def test_weights_remove_the_bias_of_labelling_by_judge_score() -> None:
    """Rows the judge fails are labelled six times as often as rows it passes."""
    rng = np.random.default_rng(11)
    truth, reps = 0.6, 500
    hits = {"weighted human-only": 0, "weighted ppi++": 0}
    naive_error = 0.0
    for _ in range(reps):
        total = 3000
        y = (rng.random(total) < truth).astype(float)
        draw = rng.random(total)
        f = np.where(y == 1.0, draw < 0.9, draw >= 0.7).astype(float)
        chance = np.where(f == 0.0, 0.30, 0.05)
        lab = rng.random(total) < chance
        w, wu = 1.0 / chance[lab], 1.0 / (1.0 - chance[~lab])
        human = classical_mean(y[lab], weights=w)
        tuned = ppi_pp_mean(y[lab], f[lab], f[~lab], weights=w, weights_unlabelled=wu)
        hits["weighted human-only"] += human.lo <= truth <= human.hi
        hits["weighted ppi++"] += tuned.lo <= truth <= tuned.hi
        naive_error += (classical_mean(y[lab]).estimate - truth) / reps
    assert naive_error < -0.15
    for name, count in hits.items():
        assert 0.92 <= count / reps <= 0.98, (name, count / reps)
