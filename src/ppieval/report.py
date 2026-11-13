"""Markdown and JSON rendering of the command reports."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from .data import Dataset
from .diagnostics import Diagnostics
from .estimators import Estimate

__all__ = [
    "estimate_json",
    "estimate_markdown",
    "estimate_payload",
    "estimates_table",
    "grid_markdown",
    "plan_markdown",
    "to_json",
]

JUDGE_ONLY_NOTE = (
    "judge-only takes the judge's scores as if they were human labels. It is shown for contrast: "
    "its interval is for the judge's mean score, not for the human-judged mean."
)


def _num(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _percent(level: float) -> str:
    return f"{100.0 * level:g} %"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], left: Sequence[int] = (0,)) -> str:
    rule = ["---" if index in left else "---:" for index in range(len(header))]
    lines = [header, rule, *rows]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines)


def _clean(value: Any) -> Any:
    """Replace non-finite floats by None so that the JSON output is valid."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def to_json(payload: dict[str, Any]) -> str:
    """JSON text of a report, with non-finite numbers written as null."""
    return json.dumps(_clean(payload), indent=2) + "\n"


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
    rows = []
    for est in estimates:
        ratio = "" if reference is None or reference <= 0.0 else f"{est.width / reference:.2f}"
        rows.append(
            [
                est.method,
                _num(est.estimate, digits),
                _num(est.se, digits),
                f"[{_num(est.lo, digits)}, {_num(est.hi, digits)}]",
                _num(est.width, digits),
                ratio,
                _num(est.lam, digits),
            ]
        )
    return _table(header, rows, left=(0, 3))


def _diagnostics_lines(diag: Diagnostics, digits: int) -> list[str]:
    rect = diag.rectifier
    lines = ["## Diagnostics", ""]
    if diag.correlation is not None:
        lines.append(
            f"- judge-human correlation on the labelled rows: {diag.correlation:.{digits}f}"
        )
    if diag.agreement is not None:
        lines.append(f"- judge-human agreement on the labelled rows: {diag.agreement:.{digits}f}")
    if diag.sensitivity is not None and diag.specificity is not None:
        lines.append(
            f"- judge sensitivity {diag.sensitivity:.{digits}f}, "
            f"specificity {diag.specificity:.{digits}f}"
        )
    lines.append(
        f"- rectifier (mean of judge minus human): {rect.estimate:.{digits}f}, "
        f"{_percent(rect.level)} interval [{rect.lo:.{digits}f}, {rect.hi:.{digits}f}]"
    )
    if diag.ks_statistic is not None and diag.ks_pvalue is not None:
        lines.append(
            f"- judge scores, labelled against unlabelled rows: KS statistic "
            f"{diag.ks_statistic:.{digits}f}, p = {diag.ks_pvalue:.3g}"
        )
    lines += [f"- warning: {text}" for text in diag.warnings]
    return lines


def estimate_markdown(
    dataset: Dataset,
    estimates: Sequence[Estimate],
    digits: int = 4,
    diagnostics: Diagnostics | None = None,
    title: str = "ppieval estimate",
    notes: Sequence[str] = (),
) -> str:
    """Report of ``ppieval estimate`` (and ``compare``) as Markdown."""
    lines = [
        f"# {title}",
        "",
        *notes,
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
    if diagnostics is not None:
        lines += ["", *_diagnostics_lines(diagnostics, digits)]
    return "\n".join(lines) + "\n"


def estimate_payload(
    dataset: Dataset,
    estimates: Sequence[Estimate],
    diagnostics: Diagnostics | None = None,
    command: str = "estimate",
) -> dict[str, Any]:
    """Report of ``ppieval estimate`` (and ``compare``) as a plain dictionary."""
    payload: dict[str, Any] = {
        "command": command,
        "rows": dataset.n_rows,
        "labelled": dataset.n_labelled,
        "unlabelled": dataset.n_unlabelled,
        "level": estimates[0].level,
        "estimates": [est.to_dict() for est in estimates],
    }
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics.to_dict()
    return payload


def estimate_json(
    dataset: Dataset, estimates: Sequence[Estimate], diagnostics: Diagnostics | None = None
) -> str:
    """Report of ``ppieval estimate`` as JSON text."""
    return to_json(estimate_payload(dataset, estimates, diagnostics))


def plan_markdown(payload: dict[str, Any], digits: int = 4) -> str:
    """Report of ``ppieval plan`` as Markdown."""
    lines = [
        "# ppieval plan",
        "",
        f"- rows: {payload['rows']}",
        f"- with a human label: {payload['labelled']}",
        f"- without a human label: {payload['unlabelled']}",
    ]
    plan = payload.get("plan")
    if plan is not None:
        level = _percent(plan["level"])
        extra = plan["effective_n"] - plan["n_labelled"]
        lines += [
            f"- judge-human correlation on the labelled rows: {plan['correlation']:.{digits}f}",
            f"- variance of the human label: {plan['var_y']:.{digits}f}",
            f"- {level} interval width now: human-only {plan['width_human_only']:.{digits}f}, "
            f"ppi++ {plan['width_ppi_pp']:.{digits}f}",
            f"- effective sample size: {plan['effective_n']:.1f} human labels "
            f"(the judge is worth {extra:.1f} extra labels)",
        ]
        if plan["target_width"] is not None:
            with_judge = plan["labels_with_judge"]
            reach = (
                "not reachable even with every row labelled" if with_judge is None else with_judge
            )
            lines += [
                "",
                f"## Target width {plan['target_width']:g}",
                "",
                f"- human labels needed with the judge: {reach}",
                f"- human labels needed without the judge: {plan['labels_human_only']}",
            ]
    sample = payload.get("sample")
    if sample is not None:
        lines += [
            "",
            "## Sample to label",
            "",
            f"- method: {sample['method']}"
            + (f", {sample['allocation']} allocation" if sample["method"] == "stratified" else ""),
            f"- rows chosen: {sample['size']} of {sample['eligible']} without a human label",
            f"- seed: {sample['seed']}",
        ]
        if sample["strata"]:
            rows = [
                [str(s["stratum"]), _num(s["judge_min"], digits), _num(s["judge_max"], digits),
                 str(s["rows"]), str(s["chosen"]), _num(s["probability"], digits)]
                for s in sample["strata"]
            ]
            header = ["stratum", "judge min", "judge max", "rows", "chosen", "probability"]
            lines += ["", _table(header, rows, left=())]
        if sample["out"]:
            lines += [
                "",
                f"Wrote {sample['out']} with the columns ppieval_sampled, ppieval_probability "
                "and ppieval_weight.",
            ]
    return "\n".join(lines) + "\n"


def grid_markdown(payload: dict[str, Any], digits: int = 4) -> str:
    """Report of ``ppieval simulate`` or ``ppieval replay`` as Markdown."""
    level = _percent(payload["level"])
    lines = [f"# ppieval {payload['command']}", ""]
    if payload["command"] == "replay":
        lines += [
            f"- rows, all with a human label: {payload['rows']}",
            f"- mean of all human labels (the target): {payload['full_data_mean']:.{digits}f}",
            f"- mean judge score: {payload['judge_mean']:.{digits}f}",
        ]
    else:
        lines += [
            f"- rows per repetition: {payload['rows']}",
            f"- probability of a human 1 (the target): {payload['prevalence']:g}",
            f"- judge bias (added to sensitivity, taken from specificity): {payload['bias']:g}",
        ]
    lines += [
        f"- repetitions per cell: {payload['reps']}",
        f"- interval level: {level}",
        f"- seed: {payload['seed']}",
    ]
    header = ["method", "coverage", "mean width", "vs human-only", "mean error", "effective n"]
    for cell in payload["cells"]:
        share = f"{100.0 * cell['fraction']:g} % of human labels kept (n = {cell['n_labelled']})"
        if payload["command"] == "simulate":
            share = (
                f"judge quality {cell['quality']:g} (sensitivity {cell['sensitivity']:g}, "
                f"specificity {cell['specificity']:g}), {share}"
            )
        rows = [
            [
                row["method"],
                f"{row['coverage']:.3f}",
                _num(row["mean_width"], digits),
                "" if row["relative_width"] is None else f"{row['relative_width']:.2f}",
                f"{row['mean_error']:+.{digits}f}",
                "" if row["effective_n"] is None else f"{row['effective_n']:.0f}",
            ]
            for row in cell["methods"]
        ]
        lines += ["", f"## {share}", "", _table(header, rows)]
    return "\n".join(lines) + "\n"
