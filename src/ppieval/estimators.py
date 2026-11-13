"""Estimators of a mean from a few human labels and many judge scores.

Notation, following the prediction-powered inference papers:

``y``
    human labels on the ``n`` labelled rows.
``f``
    judge scores on the same ``n`` rows.
``f_unlabelled``
    judge scores on the ``N`` rows that have no human label.

All estimators target the mean of the human label over the population the rows
were drawn from. A proportion (accuracy, win rate) is the mean of a 0/1 label.

The one-parameter family at the centre is the PPI++ mean estimator
(Angelopoulos, Duchi and Zrnic 2023, Example 6.1)::

    theta(lam) = mean(y) + lam * (mean(f_unlabelled) - mean(f))

``lam = 0`` is the classical sample mean of the human labels and ``lam = 1`` is
the prediction-powered estimator of Angelopoulos, Bates, Fannjiang, Jordan and
Zrnic (2023). The two terms are independent sample means, so the variance is::

    Var(y - lam * f) / n + lam**2 * Var(f) / N

and the interval is the normal-approximation (CLT) interval built from it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Estimate",
    "bootstrap_interval",
    "classical_mean",
    "estimate_all",
    "judge_only_mean",
    "normal_quantile",
    "optimal_lambda",
    "optimal_lambda_from_moments",
    "ppi_mean",
    "ppi_pp_mean",
    "rogan_gladen_mean",
    "stratified_ppi_mean",
    "variance_from_moments",
]

FloatArray = NDArray[np.float64]
Estimator = Callable[[FloatArray, FloatArray, FloatArray], "Estimate"]


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its standard error and confidence interval.

    ``n_human`` is the number of human labels the estimate used and ``n_judge``
    the number of judge scores; ``lam`` is the tuning parameter, where there is one.
    """

    method: str
    estimate: float
    se: float
    lo: float
    hi: float
    level: float
    n_human: int
    n_judge: int
    lam: float | None = None

    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.hi - self.lo

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Plain dictionary, with the interval width included."""
        out: dict[str, float | int | str | None] = asdict(self)
        out["width"] = self.width
        return out


def normal_quantile(p: float) -> float:
    """Quantile function of the standard normal distribution."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be strictly between 0 and 1, got {p!r}")
    return NormalDist().inv_cdf(p)


def _as_1d(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains missing or non-finite values")
    return arr


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be strictly between 0 and 1, got {level!r}")


def _check_labelled(y: FloatArray, f: FloatArray) -> None:
    if y.shape != f.shape:
        raise ValueError(f"y and f must have the same length, got {y.shape[0]} and {f.shape[0]}")
    if y.shape[0] < 2:
        raise ValueError("at least 2 labelled rows are needed to estimate a variance")


def _weights(values: ArrayLike | None, size: int, name: str) -> FloatArray | None:
    if values is None:
        return None
    arr = _as_1d(values, name)
    if arr.shape[0] != size:
        raise ValueError(f"{name} must have one entry per row, got {arr.shape[0]} for {size}")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive")
    return arr


def _clusters(values: Sequence[Any] | None, size: int, name: str) -> list[str] | None:
    if values is None:
        return None
    out = [str(value) for value in values]
    if len(out) != size:
        raise ValueError(f"{name} must have one entry per row, got {len(out)} for {size}")
    return out


def _influence(z: FloatArray, w: FloatArray | None) -> tuple[float, FloatArray]:
    """Weighted (Hajek) mean of ``z`` and each row's linearised contribution to it."""
    if w is None:
        mean = float(np.mean(z))
        return mean, (z - mean) / z.shape[0]
    total = float(np.sum(w))
    mean = float(np.sum(w * z) / total)
    return mean, w * (z - mean) / total


def _cluster_sums(parts: Sequence[tuple[FloatArray, list[str]]]) -> FloatArray:
    labels = np.asarray([label for _, ids in parts for label in ids])
    values = np.concatenate([infl for infl, _ in parts])
    _, codes = np.unique(labels, return_inverse=True)
    return np.bincount(codes, weights=values)


def _variance(parts: Sequence[tuple[FloatArray, list[str] | None]]) -> float:
    """Variance of a sum of independent means, from their influence values.

    Without clusters each part contributes ``sum(influence**2) * m / (m - 1)``,
    which is the sample variance over ``m`` for an unweighted mean. With
    clusters the influence values are summed within each cluster first and
    the factor is ``G / (G - 1)`` for ``G`` clusters.
    """
    if all(ids is None for _, ids in parts):
        return float(
            sum(np.sum(infl**2) * infl.shape[0] / (infl.shape[0] - 1) for infl, _ in parts)
        )
    clustered = [(infl, ids) for infl, ids in parts if ids is not None]
    if len(clustered) != len(parts):
        raise ValueError("clusters are needed for the labelled and the unlabelled rows")
    sums = _cluster_sums(clustered)
    groups = sums.shape[0]
    if groups < 2:
        raise ValueError("at least 2 clusters are needed to estimate a variance")
    return float(np.sum(sums**2) * groups / (groups - 1))


def _interval(
    method: str,
    estimate: float,
    variance: float,
    level: float,
    n_human: int,
    n_judge: int,
    lam: float | None = None,
) -> Estimate:
    se = math.sqrt(max(variance, 0.0))
    z = normal_quantile(0.5 + level / 2.0)
    return Estimate(
        method=method,
        estimate=estimate,
        se=se,
        lo=estimate - z * se,
        hi=estimate + z * se,
        level=level,
        n_human=n_human,
        n_judge=n_judge,
        lam=lam,
    )


def classical_mean(
    y: ArrayLike,
    level: float = 0.95,
    *,
    weights: ArrayLike | None = None,
    clusters: Sequence[Any] | None = None,
) -> Estimate:
    """Human-only estimate: the sample mean of the human labels.

    The variance is the sample variance (``n - 1`` denominator) over ``n``.
    ``weights`` are inverse-probability weights for rows that were not chosen
    uniformly (the mean is then the weighted mean and the variance its
    linearisation); ``clusters`` gives a cluster-robust variance.
    """
    _check_level(level)
    y_arr = _as_1d(y, "y")
    n = y_arr.shape[0]
    if n < 2:
        raise ValueError("at least 2 labelled rows are needed to estimate a variance")
    mean, infl = _influence(y_arr, _weights(weights, n, "weights"))
    variance = _variance([(infl, _clusters(clusters, n, "clusters"))])
    return _interval("human-only", mean, variance, level, n, 0)


def judge_only_mean(
    f_all: ArrayLike, level: float = 0.95, *, clusters: Sequence[Any] | None = None
) -> Estimate:
    """Naive estimate: the mean of the judge's scores, taken as if they were labels.

    Shown for contrast only. The interval describes the judge's mean score, so
    it does not cover the human-judged mean when the judge is biased, and it
    gets narrower, not better, as rows are added.
    """
    _check_level(level)
    f_arr = _as_1d(f_all, "f_all")
    total = f_arr.shape[0]
    if total < 2:
        raise ValueError("at least 2 judge scores are needed to estimate a variance")
    mean, infl = _influence(f_arr, None)
    variance = _variance([(infl, _clusters(clusters, total, "clusters"))])
    return _interval("judge-only", mean, variance, level, 0, total)


def ppi_mean(
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    lam: float = 1.0,
    level: float = 0.95,
    method: str | None = None,
    *,
    weights: ArrayLike | None = None,
    weights_unlabelled: ArrayLike | None = None,
    clusters: Sequence[Any] | None = None,
    clusters_unlabelled: Sequence[Any] | None = None,
) -> Estimate:
    """Prediction-powered estimate of the mean with a fixed tuning parameter.

    ``theta = mean(y) + lam * (mean(f_unlabelled) - mean(f))``, with variance
    ``var(y - lam * f) / n + lam**2 * var(f_unlabelled) / N``.

    ``lam = 1`` (the default) is the original prediction-powered estimator,
    ``mean(f_unlabelled) - mean(f - y)``: the judge's mean on the unlabelled
    rows minus the rectifier, the judge's average error on the labelled rows.
    ``lam = 0`` is the classical estimate and ignores the judge.

    ``weights`` and ``weights_unlabelled`` are inverse-probability weights for
    labelled rows that were not chosen uniformly: one over the probability of
    being labelled for a labelled row, one over the probability of being left
    unlabelled for the others. Both means become weighted means. ``clusters``
    and ``clusters_unlabelled`` give a cluster-robust variance; a cluster may
    hold rows of both kinds.
    """
    _check_level(level)
    y_arr = _as_1d(y, "y")
    f_arr = _as_1d(f, "f")
    fu_arr = _as_1d(f_unlabelled, "f_unlabelled")
    _check_labelled(y_arr, f_arr)
    if not math.isfinite(lam):
        raise ValueError(f"lam must be finite, got {lam!r}")
    n = y_arr.shape[0]
    n_unl = fu_arr.shape[0]
    lam = float(lam)
    name = method if method is not None else "ppi"
    w = _weights(weights, n, "weights")
    ids = _clusters(clusters, n, "clusters")

    estimate, infl = _influence(y_arr - lam * f_arr, w)
    if lam == 0.0:
        return _interval(name, estimate, _variance([(infl, ids)]), level, n, 0, lam)
    if n_unl < 2:
        raise ValueError("at least 2 rows without a human label are needed when lam is not 0")
    wu = _weights(weights_unlabelled, n_unl, "weights_unlabelled")
    ids_u = _clusters(clusters_unlabelled, n_unl, "clusters_unlabelled")
    judge_mean, infl_u = _influence(fu_arr, wu)
    variance = _variance([(infl, ids), (lam * infl_u, ids_u)])
    return _interval(name, estimate + lam * judge_mean, variance, level, n, n + n_unl, lam)


def optimal_lambda(
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    weights_unlabelled: ArrayLike | None = None,
    clusters: Sequence[Any] | None = None,
    clusters_unlabelled: Sequence[Any] | None = None,
) -> float:
    """Plug-in estimate of the variance-minimising tuning parameter.

    ``cov(y, f) / ((1 + n / N) * var(f))``, where the covariance comes from
    the labelled rows and the variance of the judge's scores from all rows,
    labelled and unlabelled (PPI++, Example 6.1). The value is not clipped:
    for a mean the variance is a convex quadratic in ``lam`` on the whole real
    line, and a negative value is the right answer for a judge that is
    negatively correlated with the human label.

    With weights or clusters the value is instead the minimiser of the
    estimated (weighted, cluster-robust) variance used by :func:`ppi_mean`.

    Returns 0 when there are fewer than 2 unlabelled rows or when the judge's
    scores are constant, because the judge then carries no usable information.
    """
    y_arr = _as_1d(y, "y")
    f_arr = _as_1d(f, "f")
    fu_arr = _as_1d(f_unlabelled, "f_unlabelled")
    _check_labelled(y_arr, f_arr)
    n = y_arr.shape[0]
    n_unl = fu_arr.shape[0]
    if n_unl < 2:
        return 0.0
    if all(v is None for v in (weights, weights_unlabelled, clusters, clusters_unlabelled)):
        var_f = float(np.var(np.concatenate([f_arr, fu_arr]), ddof=1))
        if var_f <= 0.0:
            return 0.0
        cov_yf = float(np.cov(y_arr, f_arr, ddof=1)[0, 1])
        return cov_yf / ((1.0 + n / n_unl) * var_f)

    w = _weights(weights, n, "weights")
    wu = _weights(weights_unlabelled, n_unl, "weights_unlabelled")
    _, a = _influence(y_arr, w)
    _, b = _influence(f_arr, w)
    _, d = _influence(fu_arr, wu)
    ids = _clusters(clusters, n, "clusters")
    ids_u = _clusters(clusters_unlabelled, n_unl, "clusters_unlabelled")
    if ids is None and ids_u is None:
        num, den = float(np.sum(a * b)), float(np.sum(b * b) + np.sum(d * d))
    elif ids is None or ids_u is None:
        raise ValueError("clusters are needed for the labelled and the unlabelled rows")
    else:
        y_sums = _cluster_sums([(a, ids), (np.zeros(n_unl), ids_u)])
        f_sums = _cluster_sums([(b, ids), (-d, ids_u)])
        num, den = float(np.sum(y_sums * f_sums)), float(np.sum(f_sums * f_sums))
    return num / den if den > 0.0 else 0.0


def ppi_pp_mean(
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    level: float = 0.95,
    *,
    weights: ArrayLike | None = None,
    weights_unlabelled: ArrayLike | None = None,
    clusters: Sequence[Any] | None = None,
    clusters_unlabelled: Sequence[Any] | None = None,
) -> Estimate:
    """PPI++ estimate of the mean: :func:`ppi_mean` at the estimated optimal ``lam``.

    The tuning parameter is estimated from the same data by
    :func:`optimal_lambda`. With an uninformative judge it goes to 0 and the
    estimate goes to the classical one; with a perfect judge it goes to
    ``N / (n + N)`` and the estimate goes to the mean over all rows.
    """
    design: dict[str, Any] = {
        "weights": weights,
        "weights_unlabelled": weights_unlabelled,
        "clusters": clusters,
        "clusters_unlabelled": clusters_unlabelled,
    }
    lam = optimal_lambda(y, f, f_unlabelled, **design)
    return ppi_mean(y, f, f_unlabelled, lam=lam, level=level, method="ppi++", **design)


def _strata(pooled: FloatArray, n_strata: int) -> NDArray[np.intp]:
    """Stratum index of every pooled judge score: its distinct value or its quantile bin."""
    distinct = np.unique(pooled)
    if distinct.shape[0] <= n_strata:
        return np.searchsorted(distinct, pooled)
    edges = np.unique(np.quantile(pooled, np.arange(1, n_strata) / n_strata))
    return np.searchsorted(edges, pooled, side="left")


def stratified_ppi_mean(
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    level: float = 0.95,
    n_strata: int = 5,
    min_labelled: int = 10,
) -> Estimate:
    """PPI++ within strata of the judge's score, combined by the strata's shares of all rows.

    Rows are binned by judge score: one stratum per distinct value when there
    are at most ``n_strata`` of them, otherwise ``n_strata`` quantile bins of
    the pooled scores. Neighbouring strata are merged until each holds at
    least ``min_labelled`` labelled rows. Stratum ``k`` gets its own PPI++
    estimate ``theta_k`` with variance ``V_k`` (the human-only mean when the
    judge is constant inside the stratum), and with ``W_k`` the share of all
    ``M = n + N`` rows in the stratum::

        theta = sum(W_k * theta_k)
        variance = sum(W_k**2 * V_k) + sum(W_k * (theta_k - theta)**2) / M

    The second variance term is the price of estimating the shares from the
    rows at hand, as in double sampling for stratification.
    """
    _check_level(level)
    y_arr = _as_1d(y, "y")
    f_arr = _as_1d(f, "f")
    fu_arr = _as_1d(f_unlabelled, "f_unlabelled")
    _check_labelled(y_arr, f_arr)
    if n_strata < 1 or min_labelled < 2:
        raise ValueError("n_strata must be at least 1 and min_labelled at least 2")
    n, n_unl = y_arr.shape[0], fu_arr.shape[0]
    total = n + n_unl
    index = _strata(np.concatenate([f_arr, fu_arr]), n_strata)
    lab, unl = index[:n], index[n:]
    counts = np.bincount(lab, minlength=int(index.max()) + 1)

    groups: list[list[int]] = []
    for stratum in range(counts.shape[0]):
        if groups and sum(int(counts[s]) for s in groups[-1]) < min_labelled:
            groups[-1].append(stratum)
        else:
            groups.append([stratum])
    if len(groups) > 1 and sum(int(counts[s]) for s in groups[-1]) < min_labelled:
        groups[-2].extend(groups.pop())

    shares, means, variances = [], [], []
    for group in groups:
        in_lab, in_unl = np.isin(lab, group), np.isin(unl, group)
        part = ppi_pp_mean(y_arr[in_lab], f_arr[in_lab], fu_arr[in_unl], level=level)
        shares.append((int(np.sum(in_lab)) + int(np.sum(in_unl))) / total)
        means.append(part.estimate)
        variances.append(part.se**2)
    w, m, v = np.asarray(shares), np.asarray(means), np.asarray(variances)
    estimate = float(np.sum(w * m))
    variance = float(np.sum(w * w * v) + np.sum(w * (m - estimate) ** 2) / total)
    return _interval("ppi-stratified", estimate, variance, level, n, total)


def _is_binary(*arrays: FloatArray) -> bool:
    return all(bool(np.all((arr == 0.0) | (arr == 1.0))) for arr in arrays)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def rogan_gladen_mean(
    y: ArrayLike, f: ArrayLike, f_unlabelled: ArrayLike, level: float = 0.95
) -> Estimate:
    """Rogan-Gladen corrected proportion for a 0/1 label and a 0/1 judge.

    With ``p`` the judge's positive rate on the unlabelled rows and the
    judge's sensitivity ``se`` and specificity ``sp`` measured on the
    labelled rows::

        prevalence = (p + sp - 1) / (se + sp - 1)

    The delta-method variance treats the three proportions as independent,
    which they are because ``p`` uses only unlabelled rows::

        (p(1-p)/N + prev**2 * se(1-se)/n1 + (1-prev)**2 * sp(1-sp)/n0) / (se+sp-1)**2

    with ``n1`` and ``n0`` the numbers of labelled rows with human label 1 and
    0. The estimate and the interval are truncated to [0, 1].
    """
    _check_level(level)
    y_arr = _as_1d(y, "y")
    f_arr = _as_1d(f, "f")
    fu_arr = _as_1d(f_unlabelled, "f_unlabelled")
    _check_labelled(y_arr, f_arr)
    if not _is_binary(y_arr, f_arr, fu_arr):
        raise ValueError("the Rogan-Gladen correction needs 0/1 labels and 0/1 judge scores")
    n, n_unl = y_arr.shape[0], fu_arr.shape[0]
    n1 = int(np.sum(y_arr == 1.0))
    n0 = n - n1
    if n1 == 0 or n0 == 0 or n_unl == 0:
        raise ValueError(
            "the Rogan-Gladen correction needs labelled rows of both classes and unlabelled rows"
        )
    sens = float(np.mean(f_arr[y_arr == 1.0]))
    spec = 1.0 - float(np.mean(f_arr[y_arr == 0.0]))
    youden = sens + spec - 1.0
    if youden <= 0.0:
        raise ValueError(
            "the judge is no better than chance on the labelled rows "
            "(sensitivity + specificity <= 1), so the Rogan-Gladen correction is undefined"
        )
    p = float(np.mean(fu_arr))
    raw = (p + spec - 1.0) / youden
    variance = (
        p * (1.0 - p) / n_unl
        + raw * raw * sens * (1.0 - sens) / n1
        + (1.0 - raw) ** 2 * spec * (1.0 - spec) / n0
    ) / (youden * youden)
    est = _interval("rogan-gladen", raw, variance, level, n, n + n_unl)
    return Estimate(
        est.method, _unit(raw), est.se, _unit(est.lo), _unit(est.hi), level, n, n + n_unl
    )


def bootstrap_interval(
    estimator: Estimator,
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
    clusters: Sequence[Any] | None = None,
    clusters_unlabelled: Sequence[Any] | None = None,
) -> Estimate:
    """Percentile bootstrap interval for any estimator in this module.

    Labelled rows (label and judge score together) and unlabelled rows are
    resampled with replacement, separately, and the estimator is recomputed
    each time, tuning parameter included. With clusters, whole clusters are
    resampled instead of rows. Resamples on which the estimator is undefined
    are dropped; more than a tenth of them failing is an error.
    """
    _check_level(level)
    y_arr = _as_1d(y, "y")
    f_arr = _as_1d(f, "f")
    fu_arr = _as_1d(f_unlabelled, "f_unlabelled")
    _check_labelled(y_arr, f_arr)
    if n_boot < 10:
        raise ValueError("n_boot must be at least 10")
    n, n_unl = y_arr.shape[0], fu_arr.shape[0]
    point = estimator(y_arr, f_arr, fu_arr)
    rng = np.random.default_rng(seed)
    ids = _clusters(clusters, n, "clusters")
    ids_u = _clusters(clusters_unlabelled, n_unl, "clusters_unlabelled")
    if (ids is None) != (ids_u is None) and n_unl > 0:
        raise ValueError("clusters are needed for the labelled and the unlabelled rows")
    members: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []
    if ids is not None:
        lab_ids, unl_ids = np.asarray(ids, dtype=str), np.asarray(ids_u or [], dtype=str)
        for label in sorted(set(ids) | set(ids_u or [])):
            members.append((np.flatnonzero(lab_ids == label), np.flatnonzero(unl_ids == label)))

    draws: list[float] = []
    for _ in range(n_boot):
        if members:
            picks = rng.integers(0, len(members), len(members))
            rows = np.concatenate([members[k][0] for k in picks])
            rows_u = np.concatenate([members[k][1] for k in picks])
        else:
            rows = rng.integers(0, n, n)
            rows_u = rng.integers(0, n_unl, n_unl) if n_unl else np.empty(0, dtype=np.intp)
        try:
            draws.append(estimator(y_arr[rows], f_arr[rows], fu_arr[rows_u]).estimate)
        except ValueError:
            continue
    if len(draws) < 0.9 * n_boot:
        raise ValueError("the estimator failed on more than a tenth of the bootstrap resamples")
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return Estimate(
        method=f"{point.method} (bootstrap)",
        estimate=point.estimate,
        se=float(np.std(draws, ddof=1)),
        lo=float(lo),
        hi=float(hi),
        level=level,
        n_human=point.n_human,
        n_judge=point.n_judge,
        lam=point.lam,
    )


def estimate_all(
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    level: float = 0.95,
    *,
    weights: ArrayLike | None = None,
    weights_unlabelled: ArrayLike | None = None,
    clusters: Sequence[Any] | None = None,
    clusters_unlabelled: Sequence[Any] | None = None,
    n_strata: int = 5,
    n_boot: int = 0,
    seed: int = 0,
) -> list[Estimate]:
    """Run every mean estimator that applies to the data.

    Order: human-only, judge-only, PPI (``lam = 1``), PPI++ (estimated
    ``lam``), stratified PPI, Rogan-Gladen. PPI needs at least 2 unlabelled
    rows. The stratified and Rogan-Gladen rows are left out when weights or
    clusters are given, which they do not support, and Rogan-Gladen also
    when the data are not 0/1 or the judge is no better than chance. The
    judge-only row is never weighted. With ``n_boot > 0`` a percentile
    bootstrap row follows for each row except judge-only; the bootstrap is
    not available with weights.
    """
    y_arr = _as_1d(y, "y")
    f_arr = _as_1d(f, "f")
    fu_arr = _as_1d(f_unlabelled, "f_unlabelled")
    _check_labelled(y_arr, f_arr)
    n_unl = fu_arr.shape[0]
    ids = _clusters(clusters, y_arr.shape[0], "clusters")
    ids_u = _clusters(clusters_unlabelled, n_unl, "clusters_unlabelled")
    weighted = weights is not None or weights_unlabelled is not None
    if n_boot > 0 and weighted:
        raise ValueError("the bootstrap is not available with weights")
    design: dict[str, Any] = {
        "weights": weights,
        "weights_unlabelled": weights_unlabelled,
        "clusters": ids,
        "clusters_unlabelled": ids_u,
    }
    all_ids = None if ids is None or (ids_u is None and n_unl > 0) else ids + (ids_u or [])

    out = [
        classical_mean(y_arr, level=level, weights=weights, clusters=ids),
        judge_only_mean(np.concatenate([f_arr, fu_arr]), level=level, clusters=all_ids),
    ]
    resampled: list[Estimator] = [lambda a, _b, _c: classical_mean(a, level=level)]
    if n_unl >= 2:
        out.append(ppi_mean(y_arr, f_arr, fu_arr, lam=1.0, level=level, **design))
        resampled.append(lambda a, b, c: ppi_mean(a, b, c, lam=1.0, level=level))
    out.append(ppi_pp_mean(y_arr, f_arr, fu_arr, level=level, **design))
    resampled.append(lambda a, b, c: ppi_pp_mean(a, b, c, level=level))
    if n_unl >= 2 and not weighted and ids is None and ids_u is None:
        extra: list[Estimator] = [
            lambda a, b, c: stratified_ppi_mean(a, b, c, level=level, n_strata=n_strata),
            lambda a, b, c: rogan_gladen_mean(a, b, c, level=level),
        ]
        for run in extra:
            try:
                out.append(run(y_arr, f_arr, fu_arr))
            except ValueError:
                continue
            resampled.append(run)
    for run in resampled if n_boot > 0 else []:
        out.append(
            bootstrap_interval(run, y_arr, f_arr, fu_arr, level, n_boot, seed, ids, ids_u)
        )
    return out


def _check_counts(n: int, n_unlabelled: int, lam: float) -> None:
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n!r}")
    if n_unlabelled < 0:
        raise ValueError(f"n_unlabelled must not be negative, got {n_unlabelled!r}")
    if lam != 0.0 and n_unlabelled == 0:
        raise ValueError("lam must be 0 when there are no unlabelled rows")


def variance_from_moments(
    var_y: float,
    var_f: float,
    cov_yf: float,
    n: int,
    n_unlabelled: int,
    lam: float,
) -> float:
    """Variance of the estimator with tuning parameter ``lam``, given population moments.

    ``(var_y - 2 * lam * cov_yf + lam**2 * var_f) / n + lam**2 * var_f / N``.
    """
    _check_counts(n, n_unlabelled, lam)
    labelled_term = (var_y - 2.0 * lam * cov_yf + lam * lam * var_f) / n
    if lam == 0.0:
        return labelled_term
    return labelled_term + lam * lam * var_f / n_unlabelled


def optimal_lambda_from_moments(
    var_f: float,
    cov_yf: float,
    n: int,
    n_unlabelled: int,
) -> float:
    """Tuning parameter that minimises :func:`variance_from_moments`.

    ``cov_yf / ((1 + n / N) * var_f)``; 0 when the judge's scores have no variance
    or there are no unlabelled rows.
    """
    _check_counts(n, n_unlabelled, 0.0)
    if var_f <= 0.0 or n_unlabelled == 0:
        return 0.0
    return cov_yf / ((1.0 + n / n_unlabelled) * var_f)
