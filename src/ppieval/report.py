"""Markdown and JSON rendering of estimates."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .data import Dataset
from .estimators import Estimate

__all__ = ["estimate_json", "estimate_markdown", "estimates_table"]

JUDGE_ONLY_NOTE = (
    "judge-only takes the judge's scores as if they were human labels. It is shown for contrast: "
    "its interval is for the judge's mean score, not for the human-judged mean."
)


def _num(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _percent(level: float) -> str:
    return f"{100.0 * level:g} %"


def estimates_table(estimates: Sequence[Estimate], digits: int = 4) -> str:
    """Markdown table with one row per estimator.

    The ``vs human-only`` column is the interval width divided by the width of
    the human-only interval, when that row is present.
    """
    if not estimates:
        raise ValueError("no estimates to show")
    reference = next((e.width for e in estimates if e.method == "human-only"), None)
    header = [
        "method",
        "estimate",
        "std. error",
        f"{_percent(estimates[0].level)} interval",
        "width",
        "vs human-only",
        "lambda",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for est in estimates:
        ratio = "" if reference is None or reference <= 0.0 else f"{est.width / reference:.2f}"
        cells = [
            est.method,
            _num(est.estimate, digits),
            _num(est.se, digits),
            f"[{_num(est.lo, digits)}, {_num(est.hi, digits)}]",
            _num(est.width, digits),
            ratio,
            "" if est.lam is None else _num(est.lam, digits),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def estimate_markdown(dataset: Dataset, estimates: Sequence[Estimate], digits: int = 4) -> str:
    """Report of ``ppieval estimate`` as Markdown."""
    lines = [
        "# ppieval estimate",
        "",
        f"- rows: {dataset.n_rows}",
        f"- with a human label: {dataset.n_labelled}",
        f"- without a human label: {dataset.n_unlabelled}",
        f"- interval level: {_percent(estimates[0].level)}",
        "",
        estimates_table(estimates, digits=digits),
    ]
    if any(e.method == "judge-only" for e in estimates):
        lines += ["", JUDGE_ONLY_NOTE]
    if dataset.n_unlabelled < 2:
        lines += [
            "",
            "Fewer than 2 rows lack a human label, so the judge adds nothing here: "
            "ppi is left out and ppi++ equals human-only.",
        ]
    return "\n".join(lines) + "\n"


def estimate_payload(dataset: Dataset, estimates: Sequence[Estimate]) -> dict[str, Any]:
    """Report of ``ppieval estimate`` as a plain dictionary."""
    return {
        "command": "estimate",
        "rows": dataset.n_rows,
        "labelled": dataset.n_labelled,
        "unlabelled": dataset.n_unlabelled,
        "level": estimates[0].level,
        "estimates": [est.to_dict() for est in estimates],
    }


def estimate_json(dataset: Dataset, estimates: Sequence[Estimate]) -> str:
    """Report of ``ppieval estimate`` as JSON text."""
    return json.dumps(estimate_payload(dataset, estimates), indent=2) + "\n"
