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

The one-parameter family implemented here is the PPI++ mean estimator
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
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Estimate",
    "classical_mean",
    "estimate_all",
    "judge_only_mean",
    "normal_quantile",
    "optimal_lambda",
    "optimal_lambda_from_moments",
    "ppi_mean",
    "ppi_pp_mean",
    "variance_from_moments",
]

FloatArray = NDArray[np.float64]


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


def classical_mean(y: ArrayLike, level: float = 0.95) -> Estimate:
    """Human-only estimate: the sample mean of the human labels.

    The variance is the sample variance (``n - 1`` denominator) over ``n``.
    """
    _check_level(level)
    y_arr = _as_1d(y, "y")
    n = y_arr.shape[0]
    if n < 2:
        raise ValueError("at least 2 labelled rows are needed to estimate a variance")
    variance = float(np.var(y_arr, ddof=1)) / n
    return _interval("human-only", float(np.mean(y_arr)), variance, level, n, 0)


def judge_only_mean(f_all: ArrayLike, level: float = 0.95) -> Estimate:
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
    variance = float(np.var(f_arr, ddof=1)) / total
    return _interval("judge-only", float(np.mean(f_arr)), variance, level, 0, total)


def ppi_mean(
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    lam: float = 1.0,
    level: float = 0.95,
    method: str | None = None,
) -> Estimate:
    """Prediction-powered estimate of the mean with a fixed tuning parameter.

    ``theta = mean(y) + lam * (mean(f_unlabelled) - mean(f))``, with variance
    ``var(y - lam * f) / n + lam**2 * var(f_unlabelled) / N``.

    ``lam = 1`` (the default) is the original prediction-powered estimator,
    ``mean(f_unlabelled) - mean(f - y)``: the judge's mean on the unlabelled
    rows minus the rectifier, the judge's average error on the labelled rows.
    ``lam = 0`` is the classical estimate and ignores the judge.
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

    rectified = y_arr - lam * f_arr
    estimate = float(np.mean(rectified))
    variance = float(np.var(rectified, ddof=1)) / n
    if lam == 0.0:
        return _interval(name, estimate, variance, level, n, 0, lam)
    if n_unl < 2:
        raise ValueError("at least 2 rows without a human label are needed when lam is not 0")
    estimate += lam * float(np.mean(fu_arr))
    variance += lam * lam * float(np.var(fu_arr, ddof=1)) / n_unl
    return _interval(name, estimate, variance, level, n, n + n_unl, lam)


def optimal_lambda(y: ArrayLike, f: ArrayLike, f_unlabelled: ArrayLike) -> float:
    """Plug-in estimate of the variance-minimising tuning parameter.

    ``cov(y, f) / ((1 + n / N) * var(f))``, where the covariance comes from
    the labelled rows and the variance of the judge's scores from all rows,
    labelled and unlabelled (PPI++, Example 6.1). The value is not clipped:
    for a mean the variance is a convex quadratic in ``lam`` on the whole real
    line, and a negative value is the right answer for a judge that is
    negatively correlated with the human label.

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
    var_f = float(np.var(np.concatenate([f_arr, fu_arr]), ddof=1))
    if var_f <= 0.0:
        return 0.0
    cov_yf = float(np.cov(y_arr, f_arr, ddof=1)[0, 1])
    return cov_yf / ((1.0 + n / n_unl) * var_f)


def ppi_pp_mean(
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    level: float = 0.95,
) -> Estimate:
    """PPI++ estimate of the mean: :func:`ppi_mean` at the estimated optimal ``lam``.

    The tuning parameter is estimated from the same data by
    :func:`optimal_lambda`. With an uninformative judge it goes to 0 and the
    estimate goes to the classical one; with a perfect judge it goes to
    ``N / (n + N)`` and the estimate goes to the mean over all rows.
    """
    lam = optimal_lambda(y, f, f_unlabelled)
    return ppi_mean(y, f, f_unlabelled, lam=lam, level=level, method="ppi++")


def estimate_all(
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    level: float = 0.95,
) -> list[Estimate]:
    """Run every mean estimator on the same data.

    Order: human-only, judge-only, PPI (``lam = 1``), PPI++ (estimated ``lam``).
    The PPI row is left out when fewer than 2 rows lack a human label.
    """
    y_arr = _as_1d(y, "y")
    f_arr = _as_1d(f, "f")
    fu_arr = _as_1d(f_unlabelled, "f_unlabelled")
    _check_labelled(y_arr, f_arr)
    out = [
        classical_mean(y_arr, level=level),
        judge_only_mean(np.concatenate([f_arr, fu_arr]), level=level),
    ]
    if fu_arr.shape[0] >= 2:
        out.append(ppi_mean(y_arr, f_arr, fu_arr, lam=1.0, level=level))
    out.append(ppi_pp_mean(y_arr, f_arr, fu_arr, level=level))
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
