"""Coverage and width of every estimator, on simulated data and on replays of real data."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .estimators import Estimate, _as_1d, _check_level, estimate_all

__all__ = ["replay", "simulate"]


class _Tally:
    """Running coverage, width and error of each method over repetitions."""

    def __init__(self, truth: float) -> None:
        self.truth = truth
        self.rows: dict[str, list[float]] = {}

    def add(self, estimates: list[Estimate]) -> None:
        for est in estimates:
            row = self.rows.setdefault(est.method, [0.0, 0.0, 0.0, 0.0, 0.0])
            row[0] += 1.0
            row[1] += float(est.lo <= self.truth <= est.hi)
            row[2] += est.width
            row[3] += est.estimate - self.truth
            row[4] += est.se**2

    def summary(self, n: int) -> list[dict[str, Any]]:
        base = self.rows.get("human-only")
        out = []
        for method, (used, hits, width, error, variance) in self.rows.items():
            row: dict[str, Any] = {
                "method": method,
                "reps_used": int(used),
                "coverage": hits / used,
                "mean_width": width / used,
                "relative_width": None,
                "mean_error": error / used,
                "effective_n": None,
            }
            if base is not None and base[2] > 0.0:
                row["relative_width"] = (width / used) / (base[2] / base[0])
            if base is not None and variance > 0.0 and method != "judge-only":
                row["effective_n"] = n * (base[4] / base[0]) / (variance / used)
            out.append(row)
        return out


def simulate(
    qualities: tuple[float, ...] = (0.6, 0.75, 0.9),
    fractions: tuple[float, ...] = (0.05, 0.1, 0.25),
    rows: int = 2000,
    prevalence: float = 0.5,
    bias: float = 0.1,
    reps: int = 500,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Coverage and width of every estimator across judge quality and labelled fraction.

    Each repetition draws ``rows`` 0/1 human labels with probability
    ``prevalence`` of a 1, and a 0/1 judge with sensitivity ``quality + bias``
    and specificity ``quality - bias`` (both kept inside [0, 1]), so a
    positive ``bias`` is a judge that says 1 too often; at the default
    prevalence of 0.5 its mean score is exactly ``bias`` above the truth. A
    fraction of the rows, chosen uniformly, keeps its human label. Coverage
    is measured against ``prevalence``.
    """
    _check_level(level)
    if rows < 20 or reps < 1 or not 0.0 < prevalence < 1.0:
        raise ValueError("need rows >= 20, reps >= 1 and a prevalence strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    cells = []
    for quality in qualities:
        sens = min(1.0, max(0.0, quality + bias))
        spec = min(1.0, max(0.0, quality - bias))
        for fraction in fractions:
            n = max(2, round(fraction * rows))
            if n > rows - 2:
                raise ValueError("each fraction must leave at least 2 rows without a label")
            tally = _Tally(prevalence)
            for _ in range(reps):
                human = (rng.random(rows) < prevalence).astype(float)
                draw = rng.random(rows)
                judge = np.where(human == 1.0, draw < sens, draw >= spec).astype(float)
                tally.add(estimate_all(human[:n], judge[:n], judge[n:], level=level))
            cells.append(
                {
                    "quality": quality,
                    "sensitivity": sens,
                    "specificity": spec,
                    "fraction": fraction,
                    "n_labelled": n,
                    "methods": tally.summary(n),
                }
            )
    return {
        "command": "simulate",
        "rows": rows,
        "prevalence": prevalence,
        "bias": bias,
        "reps": reps,
        "level": level,
        "seed": seed,
        "cells": cells,
    }


def replay(
    judge: ArrayLike,
    human: ArrayLike,
    fractions: tuple[float, ...] = (0.05, 0.1, 0.25),
    reps: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Hide human labels at random on a fully labelled dataset and score every estimator.

    For each fraction, ``reps`` times: keep the human label on a uniformly
    chosen ``fraction`` of the rows, run every estimator, and record whether
    its interval contains the mean of all human labels, and its width. The
    judge-only interval uses no human label, so it is the same in every
    repetition and its coverage is 0 or 1.

    ``effective_n`` is ``n`` times the mean human-only variance over the mean
    variance of the method. Because the target is the mean over the rows at
    hand and not over a wider population, intervals built for the latter
    over-cover as the kept fraction grows.
    """
    _check_level(level)
    f_all = _as_1d(judge, "judge")
    y_all = _as_1d(human, "human")
    if f_all.shape != y_all.shape:
        raise ValueError("judge and human must have the same length")
    total = y_all.shape[0]
    if total < 20 or reps < 1:
        raise ValueError("a replay needs at least 20 fully labelled rows and reps >= 1")
    truth = float(np.mean(y_all))
    rng = np.random.default_rng(seed)
    cells = []
    for fraction in fractions:
        n = max(2, round(fraction * total))
        if n > total - 2:
            raise ValueError("each fraction must leave at least 2 rows without a label")
        tally = _Tally(truth)
        for _ in range(reps):
            order = rng.permutation(total)
            kept, hidden = order[:n], order[n:]
            tally.add(estimate_all(y_all[kept], f_all[kept], f_all[hidden], level=level))
        cells.append({"fraction": fraction, "n_labelled": n, "methods": tally.summary(n)})
    return {
        "command": "replay",
        "rows": total,
        "full_data_mean": truth,
        "judge_mean": float(np.mean(f_all)),
        "reps": reps,
        "level": level,
        "seed": seed,
        "cells": cells,
    }
