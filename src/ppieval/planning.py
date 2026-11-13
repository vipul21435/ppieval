"""Planning: what the judge is worth, how many labels to buy, and which items to label."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimators import (
    _as_1d,
    _check_labelled,
    _check_level,
    _strata,
    classical_mean,
    normal_quantile,
    ppi_pp_mean,
)

__all__ = [
    "Plan",
    "Sample",
    "effective_sample_size",
    "effective_sample_size_from_correlation",
    "labels_needed",
    "make_plan",
    "sample_to_label",
]


def effective_sample_size(y: ArrayLike, f: ArrayLike, f_unlabelled: ArrayLike) -> float:
    """Number of human labels that would give the PPI++ variance without a judge.

    ``n * var(human-only) / var(ppi++)``, both estimated on the data. The
    difference from ``n`` is how many human labels the judge is worth here.
    """
    classical = classical_mean(y)
    tuned = ppi_pp_mean(y, f, f_unlabelled)
    if tuned.se == 0.0:
        return math.inf
    return classical.n_human * (classical.se / tuned.se) ** 2


def effective_sample_size_from_correlation(n: int, n_unlabelled: int, rho: float) -> float:
    """``n / (1 - rho**2 * N / (n + N))``: the same quantity from the judge-human correlation."""
    if n < 1 or n_unlabelled < 0 or not -1.0 <= rho <= 1.0:
        raise ValueError("need n >= 1, n_unlabelled >= 0 and a correlation in [-1, 1]")
    shrink = 1.0 - rho * rho * n_unlabelled / (n + n_unlabelled)
    return math.inf if shrink <= 0.0 else n / shrink


def labels_needed(
    width: float, var_y: float, rho: float, total_rows: int, level: float = 0.95
) -> tuple[int | None, int]:
    """Human labels needed for an interval of full width ``width``.

    Returns ``(with_judge, human_only)``. With the judge scoring all
    ``total_rows`` rows and ``n`` of them labelled, the PPI++ variance at the
    best tuning parameter is ``var_y * ((1 - rho**2) / n + rho**2 / total_rows)``,
    which is solved for ``n``; ``with_judge`` is ``None`` when even labelling
    every row would leave a wider interval. ``human_only`` is the classical
    ``var_y * (2 z / width)**2`` and ignores ``total_rows``.
    """
    _check_level(level)
    if width <= 0.0 or var_y < 0.0 or total_rows < 2 or not -1.0 <= rho <= 1.0:
        raise ValueError("need width > 0, var_y >= 0, total_rows >= 2 and a correlation in [-1, 1]")
    target = (width / (2.0 * normal_quantile(0.5 + level / 2.0))) ** 2
    human_only = max(2, math.ceil(var_y / target))
    if target < var_y / total_rows:
        return None, human_only
    spare = target - var_y * rho * rho / total_rows
    with_judge = total_rows if spare <= 0.0 else math.ceil(var_y * (1.0 - rho * rho) / spare)
    return min(total_rows, max(2, with_judge)), human_only


@dataclass(frozen=True)
class Plan:
    """Summary used by ``ppieval plan``."""

    n_labelled: int
    n_unlabelled: int
    correlation: float
    var_y: float
    level: float
    width_human_only: float
    width_ppi_pp: float
    effective_n: float
    target_width: float | None = None
    labels_with_judge: int | None = None
    labels_human_only: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def make_plan(
    y: ArrayLike,
    f: ArrayLike,
    f_unlabelled: ArrayLike,
    level: float = 0.95,
    target_width: float | None = None,
) -> Plan:
    """Effective sample size now, and labels needed for ``target_width`` if one is given.

    The label counts assume the judge-human correlation and the label variance
    observed on the labelled rows, and the same total number of rows.
    """
    y_arr = _as_1d(y, "y")
    f_arr = _as_1d(f, "f")
    fu_arr = _as_1d(f_unlabelled, "f_unlabelled")
    _check_labelled(y_arr, f_arr)
    var_y = float(np.var(y_arr, ddof=1))
    constant = var_y == 0.0 or float(np.var(f_arr)) == 0.0
    rho = 0.0 if constant else float(np.corrcoef(y_arr, f_arr)[0, 1])
    classical = classical_mean(y_arr, level=level)
    tuned = ppi_pp_mean(y_arr, f_arr, fu_arr, level=level)
    with_judge = human_only = None
    if target_width is not None:
        total = y_arr.shape[0] + fu_arr.shape[0]
        with_judge, human_only = labels_needed(target_width, var_y, rho, total, level)
    return Plan(
        y_arr.shape[0], fu_arr.shape[0], rho, var_y, level, classical.width, tuned.width,
        effective_sample_size(y_arr, f_arr, fu_arr), target_width, with_judge, human_only,
    )


@dataclass(frozen=True)
class Sample:
    """Rows chosen for human labelling.

    ``probability`` is each eligible row's chance of being chosen (0 for rows
    that were not eligible). ``weight`` is what the estimators take: one over
    that probability for a chosen row, one over its complement for the rest.
    """

    selected: NDArray[np.bool_]
    probability: NDArray[np.float64]
    weight: NDArray[np.float64]
    stratum: NDArray[np.intp]


def _allocate(size: int, counts: NDArray[np.intp], equal: bool) -> NDArray[np.intp]:
    """Split ``size`` over strata, never asking a stratum for more rows than it has."""
    take = np.zeros(counts.shape[0], dtype=np.intp)
    remaining = size
    while remaining > 0:
        room = counts - take
        open_strata = np.flatnonzero(room > 0)
        share = np.ones(open_strata.shape[0]) if equal else counts[open_strata].astype(float)
        quota = np.minimum(np.floor(remaining * share / share.sum()), room[open_strata])
        if quota.sum() == 0:
            order = open_strata[np.argsort(-share, kind="stable")][:remaining]
            take[order] += 1
            remaining -= order.shape[0]
        else:
            take[open_strata] += quota.astype(np.intp)
            remaining -= int(quota.sum())
    return take


def sample_to_label(
    judge: ArrayLike,
    size: int,
    method: str = "uniform",
    n_strata: int = 5,
    allocation: str = "proportional",
    seed: int = 0,
    eligible: ArrayLike | None = None,
) -> Sample:
    """Choose ``size`` rows to send for human labelling.

    ``method="uniform"`` draws them uniformly without replacement.
    ``method="stratified"`` bins the eligible rows by judge score (as
    :func:`ppieval.stratified_ppi_mean` does) and draws within each bin:
    ``allocation="proportional"`` takes the same fraction of every bin, which
    spreads the labels evenly over judge scores, and ``"equal"`` takes the
    same number from every bin, which oversamples rare scores and makes the
    returned weights necessary.
    """
    scores = _as_1d(judge, "judge")
    total = scores.shape[0]
    mask = np.ones(total, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if mask.shape != scores.shape:
        raise ValueError("eligible must have one entry per row")
    pool = np.flatnonzero(mask)
    if not 1 <= size <= pool.shape[0]:
        raise ValueError(f"size must be between 1 and the {pool.shape[0]} eligible rows")
    if method not in {"uniform", "stratified"} or allocation not in {"proportional", "equal"}:
        raise ValueError("method is uniform or stratified; allocation is proportional or equal")
    rng = np.random.default_rng(seed)
    stratum = np.full(total, -1, dtype=np.intp)
    stratum[pool] = _strata(scores[pool], n_strata) if method == "stratified" else 0
    counts = np.bincount(stratum[pool])
    take = _allocate(size, counts, equal=allocation == "equal")

    selected = np.zeros(total, dtype=bool)
    probability = np.zeros(total)
    for index in range(counts.shape[0]):
        rows = pool[stratum[pool] == index]
        if rows.shape[0] == 0:
            continue
        selected[rng.choice(rows, size=int(take[index]), replace=False)] = True
        probability[rows] = take[index] / rows.shape[0]
    weight = np.full(total, np.nan)
    chosen, left = selected, mask & ~selected
    weight[chosen] = 1.0 / probability[chosen]
    weight[left] = 1.0 / (1.0 - probability[left])
    return Sample(selected, probability, weight, stratum)
