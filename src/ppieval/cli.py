"""Command line interface: ``ppieval estimate|compare|plan|simulate|replay``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .data import DataError, Dataset, dataset_from_rows, paired_dataset_from_rows, read_rows
from .diagnostics import diagnose
from .estimators import estimate_all
from .planning import make_plan, sample_to_label
from .report import estimate_markdown, estimate_payload, grid_markdown, plan_markdown, to_json
from .simulate import replay, simulate

__all__ = ["build_parser", "main"]


def _add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="input file (.jsonl, .ndjson, .csv or .tsv)")
    parser.add_argument(
        "--judge-col", default="judge", help="column with the judge's score (default: judge)"
    )
    parser.add_argument(
        "--human-col",
        default="human",
        help="column with the human label, empty where there is none (default: human)",
    )
    parser.add_argument(
        "--input-format",
        choices=["jsonl", "csv", "tsv"],
        default=None,
        help="input format (default: from the file extension)",
    )


def _add_output(parser: argparse.ArgumentParser, level: bool = True) -> None:
    if level:
        parser.add_argument(
            "--level", type=float, default=0.95, help="confidence level (default: 0.95)"
        )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="output format (default: markdown)",
    )
    parser.add_argument(
        "--digits", type=int, default=4, help="decimal places in the Markdown output (default: 4)"
    )


def _add_design(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cluster-col",
        default=None,
        help="column naming each row's cluster; rows of one cluster are not independent",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="B",
        help="also report percentile bootstrap intervals from B resamples (default: 0, off)",
    )
    parser.add_argument("--strata", type=int, default=5, help="judge-score strata (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppieval",
        description=(
            "Estimate a human-judged mean or proportion from a few human labels "
            "and an automatic judge's scores on every row."
        ),
    )
    parser.add_argument("--version", action="version", version=f"ppieval {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    estimate = commands.add_parser(
        "estimate",
        help="estimate the mean of the human label with every estimator",
        description=(
            "Read a file with a judge score on every row and a human label on some rows, and "
            "report the human-only, judge-only, PPI, PPI++, stratified PPI and Rogan-Gladen "
            "estimates of the mean human label with their confidence intervals."
        ),
    )
    _add_input(estimate)
    _add_design(estimate)
    estimate.add_argument(
        "--weight-col",
        default=None,
        help="column of inverse-probability weights, as written by 'ppieval plan --sample'",
    )
    _add_output(estimate)
    estimate.set_defaults(run=_run_estimate)

    compare = commands.add_parser(
        "compare",
        help="estimate the difference between two systems on shared items",
        description=(
            "Read rows for several systems scored on the same items and estimate the mean "
            "difference between two of them, first minus second, from paired differences."
        ),
    )
    _add_input(compare)
    compare.add_argument("--systems", nargs=2, required=True, metavar=("A", "B"))
    compare.add_argument("--system-col", default="system", help="system column (default: system)")
    compare.add_argument("--item-col", default="item", help="item column (default: item)")
    _add_design(compare)
    _add_output(compare)
    compare.set_defaults(run=_run_compare)

    plan = commands.add_parser(
        "plan",
        help="effective sample size, labels needed for a target width, items to label",
        description=(
            "Report how many human labels the judge is worth on this file, how many labels a "
            "target interval width needs, and optionally choose rows to label next."
        ),
    )
    _add_input(plan)
    plan.add_argument(
        "--target-width", type=float, default=None, help="full interval width wanted"
    )
    plan.add_argument(
        "--sample", type=int, default=None, metavar="K", help="choose K rows to label"
    )
    plan.add_argument("--method", choices=["uniform", "stratified"], default="uniform")
    plan.add_argument("--allocation", choices=["proportional", "equal"], default="proportional")
    plan.add_argument("--strata", type=int, default=5, help="judge-score strata (default: 5)")
    plan.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    plan.add_argument(
        "--out", default=None, help="write the rows with the sample marked (.csv, .tsv, .jsonl)"
    )
    _add_output(plan)
    plan.set_defaults(run=_run_plan)

    sim = commands.add_parser(
        "simulate",
        help="coverage and width of every estimator on simulated 0/1 data",
        description=(
            "Simulate 0/1 human labels and a 0/1 judge of a given quality and bias, label a "
            "fraction of the rows, and report coverage and width of every estimator."
        ),
    )
    sim.add_argument("--qualities", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    sim.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.25])
    sim.add_argument("--rows", type=int, default=2000, help="rows per repetition (default: 2000)")
    sim.add_argument("--prevalence", type=float, default=0.5, help="P(human = 1) (default: 0.5)")
    sim.add_argument("--bias", type=float, default=0.1, help="judge bias (default: 0.1)")
    sim.add_argument("--reps", type=int, default=500, help="repetitions per cell (default: 500)")
    sim.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    _add_output(sim)
    sim.set_defaults(run=_run_simulate)

    rep = commands.add_parser(
        "replay",
        help="hide human labels at random on a fully labelled file and score every estimator",
        description=(
            "On a file where every row has a human label and a judge score, repeatedly keep a "
            "random fraction of the human labels, run every estimator, and report how often "
            "each interval contains the mean of all human labels, and how wide it is."
        ),
    )
    _add_input(rep)
    rep.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.25])
    rep.add_argument("--reps", type=int, default=1000, help="repetitions (default: 1000)")
    rep.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    _add_output(rep)
    rep.set_defaults(run=_run_replay)
    return parser


def _check_common(args: argparse.Namespace) -> None:
    if args.digits < 0:
        raise ValueError("--digits must not be negative")


def _rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_rows(args.path, args.input_format)
    if rows and all(args.judge_col not in row for row in rows):
        found = ", ".join(sorted({key for row in rows[:50] for key in row}))
        raise DataError(f"no row has a column named {args.judge_col!r}; columns found: {found}")
    return rows


def _report(args: argparse.Namespace, dataset: Dataset, title: str, notes: Sequence[str]) -> str:
    estimates = estimate_all(
        dataset.y,
        dataset.f,
        dataset.f_unlabelled,
        level=args.level,
        n_strata=args.strata,
        n_boot=args.bootstrap,
        seed=args.seed,
        **dataset.design(),
    )
    diagnostics = diagnose(dataset.y, dataset.f, dataset.f_unlabelled, level=args.level)
    if args.format == "json":
        return to_json(estimate_payload(dataset, estimates, diagnostics, command=args.command))
    return estimate_markdown(dataset, estimates, args.digits, diagnostics, title, notes)


def _run_estimate(args: argparse.Namespace) -> str:
    dataset = dataset_from_rows(
        _rows(args), args.judge_col, args.human_col, args.cluster_col, args.weight_col
    )
    return _report(args, dataset, "ppieval estimate", ())


def _run_compare(args: argparse.Namespace) -> str:
    first, second = args.systems
    dataset = paired_dataset_from_rows(
        _rows(args),
        (first, second),
        item_col=args.item_col,
        system_col=args.system_col,
        judge_col=args.judge_col,
        human_col=args.human_col,
        cluster_col=args.cluster_col,
    )
    notes = [f"- difference: {first} minus {second}, on the items scored for both"]
    return _report(args, dataset, "ppieval compare", notes)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        text = "".join(json.dumps(row) + "\n" for row in rows)
        path.write_text(text, encoding="utf-8")
        return
    if suffix not in {".csv", ".tsv"}:
        raise DataError("--out must end in .csv, .tsv or .jsonl")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t" if suffix == ".tsv" else ",",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _run_plan(args: argparse.Namespace) -> str:
    rows = _rows(args)
    dataset = dataset_from_rows(rows, args.judge_col, args.human_col)
    payload: dict[str, Any] = {
        "command": "plan",
        "rows": dataset.n_rows,
        "labelled": dataset.n_labelled,
        "unlabelled": dataset.n_unlabelled,
        "plan": None,
        "sample": None,
    }
    if dataset.n_labelled >= 2:
        plan = make_plan(dataset.y, dataset.f, dataset.f_unlabelled, args.level, args.target_width)
        payload["plan"] = plan.to_dict()
    elif args.sample is None:
        raise ValueError("at least 2 labelled rows are needed to plan; use --sample to choose rows")
    if args.sample is not None:
        if args.method == "stratified" and dataset.n_labelled > 0:
            raise ValueError(
                "stratified sampling needs a file without human labels: with labels already "
                "present the combined inclusion probabilities are not handled"
            )
        sample = sample_to_label(
            dataset.judge, args.sample, args.method, args.strata, args.allocation, args.seed,
            eligible=~dataset.labelled,
        )
        strata = []
        for index in np.unique(sample.stratum[sample.stratum >= 0]):
            inside = sample.stratum == index
            strata.append(
                {
                    "stratum": int(index) + 1,
                    "judge_min": float(dataset.judge[inside].min()),
                    "judge_max": float(dataset.judge[inside].max()),
                    "rows": int(inside.sum()),
                    "chosen": int((inside & sample.selected).sum()),
                    "probability": float(sample.probability[inside][0]),
                }
            )
        payload["sample"] = {
            "method": args.method,
            "allocation": args.allocation,
            "size": args.sample,
            "eligible": dataset.n_unlabelled,
            "seed": args.seed,
            "strata": strata if args.method == "stratified" else [],
            "rows_chosen": [int(i) + 1 for i in np.flatnonzero(sample.selected)],
            "out": args.out,
        }
        if args.out is not None:
            marked = []
            for index, row in enumerate(rows):
                eligible = sample.stratum[index] >= 0
                marked.append(
                    {
                        **row,
                        "ppieval_sampled": int(sample.selected[index]),
                        "ppieval_probability": float(sample.probability[index]) if eligible else "",
                        "ppieval_weight": float(sample.weight[index]) if eligible else "",
                    }
                )
            _write_rows(Path(args.out), marked)
    return to_json(payload) if args.format == "json" else plan_markdown(payload, args.digits)


def _run_simulate(args: argparse.Namespace) -> str:
    payload = simulate(
        tuple(args.qualities), tuple(args.fractions), args.rows, args.prevalence, args.bias,
        args.reps, args.level, args.seed,
    )
    return to_json(payload) if args.format == "json" else grid_markdown(payload, args.digits)


def _run_replay(args: argparse.Namespace) -> str:
    dataset = dataset_from_rows(_rows(args), args.judge_col, args.human_col)
    if dataset.n_unlabelled > 0:
        raise ValueError(
            f"a replay needs a human label on every row; {dataset.n_unlabelled} rows have none"
        )
    payload = replay(
        dataset.judge, dataset.human, tuple(args.fractions), args.reps, args.level, args.seed
    )
    return to_json(payload) if args.format == "json" else grid_markdown(payload, args.digits)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit status."""
    args = build_parser().parse_args(argv)
    try:
        _check_common(args)
        output = args.run(args)
    except ValueError as exc:
        print(f"ppieval: error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
