"""Checks on the judge and on the labelled subset before trusting an estimate."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .estimators import Estimate, _as_1d, _check_labelled, classical_mean

__all__ = ["Diagnostics", "diagnose", "ks_two_sample"]


def ks_two_sample(a: ArrayLike, b: ArrayLike) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and its asymptotic p-value.

    The statistic is the largest gap between the two empirical distribution
    functions. The p-value is the Kolmogorov limit ``Q(D * sqrt(n*m/(n+m)))``
    with ``Q(x) = 2 * sum_{j>=1} (-1)**(j-1) * exp(-2 * j**2 * x**2)``; it is
    conservative when the scores have many ties, as 0/1 scores do.
    """
    x = np.sort(_as_1d(a, "a"))
    y = np.sort(_as_1d(b, "b"))
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("both samples need at least one value")
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / x.shape[0]
    cdf_y = np.searchsorted(y, grid, side="right") / y.shape[0]
    statistic = float(np.max(np.abs(cdf_x - cdf_y)))
    scaled = statistic * math.sqrt(x.shape[0] * y.shape[0] / (x.shape[0] + y.shape[0]))
    if scaled < 0.2:
        return statistic, 1.0
    terms = [(-1.0) ** (j - 1) * math.exp(-2.0 * j * j * scaled * scaled) for j in range(1, 101)]
    return statistic, min(1.0, max(0.0, 2.0 * sum(terms)))


@dataclass(frozen=True)
class Diagnostics:
    """Judge-human agreement, the rectifier, and the random-subset check.

    ``agreement`` is the share of labelled rows where judge and human give the
    same value, reported when both use at most 10 distinct values in total;
    ``sensitivity`` and ``specificity`` are reported for 0/1 data. The
    rectifier is the judge's mean error, ``mean(f - y)``, with its interval.
    """

    n_labelled: int
    n_unlabelled: int
    correlation: float | None
    agreement: float | None
    sensitivity: float | None
    specificity: float | None
    rectifier: Estimate
    ks_statistic: float | None
    ks_pvalue: float | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["rectifier"] = self.rectifier.to_dict()
        out["warnings"] = list(self.warnings)
        return out


def diagnose(
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    level: float = 0.95,
    ks_alpha: float = 0.01,
) -> Diagnostics:
    """Describe the judge on the labelled rows and compare labelled with unlabelled rows.

    A warning is raised when the two-sample KS test rejects, at ``ks_alpha``,
    that the judge scores of labelled and unlabelled rows share a
    distribution: the labelled rows then do not look like a uniform random
    subset, which every estimator here assumes unless weights are given.

    Further warnings flag the cases in which the normal intervals are known
    to be too narrow: fewer than 30 labels; 0/1 labels with fewer than 10 in
    the rarer class; skewed labels with fewer than ``25 * skewness**2`` of
    them (a rule of thumb for the mean of a skewed variable); and a judge
    error with zero sample variance on the labelled rows.
    """
    y_arr = _as_1d(y, "y")
    f_arr = _as_1d(f, "f")
    fu_arr = _as_1d(f_unlabelled, "f_unlabelled")
    _check_labelled(y_arr, f_arr)
    n, n_unl = y_arr.shape[0], fu_arr.shape[0]
    warnings: list[str] = []

    correlation = None
    if np.var(y_arr) > 0.0 and np.var(f_arr) > 0.0:
        correlation = float(np.corrcoef(y_arr, f_arr)[0, 1])
        if correlation <= 0.0:
            warnings.append(
                "the judge is not positively correlated with the human labels on the labelled rows"
            )
    else:
        warnings.append("the judge scores or the human labels are constant on the labelled rows")

    values = np.unique(np.concatenate([y_arr, f_arr]))
    agreement = float(np.mean(y_arr == f_arr)) if values.shape[0] <= 10 else None
    sensitivity = specificity = None
    if bool(np.all((values == 0.0) | (values == 1.0))):
        if np.any(y_arr == 1.0):
            sensitivity = float(np.mean(f_arr[y_arr == 1.0]))
        if np.any(y_arr == 0.0):
            specificity = 1.0 - float(np.mean(f_arr[y_arr == 0.0]))

    rectifier = replace(classical_mean(f_arr - y_arr, level=level), method="rectifier")

    ks_statistic = ks_pvalue = None
    if n_unl > 0:
        ks_statistic, ks_pvalue = ks_two_sample(f_arr, fu_arr)
        if ks_pvalue < ks_alpha:
            warnings.append(
                f"judge scores of labelled and unlabelled rows differ (KS statistic "
                f"{ks_statistic:.3f}, p = {ks_pvalue:.2g}): the labelled rows may not be a "
                "random subset, and the estimates are then biased unless weights are given"
            )
    if n < 30:
        warnings.append("fewer than 30 human labels: the normal approximation may be poor")
    if float(np.var(f_arr - y_arr)) == 0.0:
        warnings.append(
            "the judge's error is the same on every labelled row (it agrees with every human "
            "label, or is off by a constant), so the estimated variance of that error is zero "
            "and the ppi and ppi++ intervals are too narrow"
        )
    if bool(np.all((y_arr == 0.0) | (y_arr == 1.0))):
        rare = int(min(np.sum(y_arr == 1.0), np.sum(y_arr == 0.0)))
        if rare < 10:
            warnings.append(
                f"only {rare} of the {n} human labels are in the rarer class: with fewer than "
                "10 the normal intervals are too narrow and cover less often than their level"
            )
    elif float(np.var(y_arr)) > 0.0:
        centred = y_arr - np.mean(y_arr)
        skewness = float(np.mean(centred**3) / np.mean(centred**2) ** 1.5)
        wanted = math.ceil(25.0 * skewness * skewness)
        if n < wanted:
            warnings.append(
                f"the human labels are skewed (skewness {skewness:.2f}): the normal "
                f"approximation wants about 25 x skewness^2 = {wanted} labels and there are {n}"
            )
    return Diagnostics(
        n, n_unl, correlation, agreement, sensitivity, specificity, rectifier,
        ks_statistic, ks_pvalue, tuple(warnings),
    )
