"""Command line interface: ``ppieval estimate FILE``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .data import load_dataset
from .estimators import estimate_all
from .report import estimate_json, estimate_markdown

__all__ = ["build_parser", "main"]


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
            "Read a JSON Lines, CSV or TSV file with a judge score on every row and a human "
            "label on some rows, and report the human-only, judge-only, PPI and PPI++ "
            "estimates of the mean human label with their confidence intervals."
        ),
    )
    estimate.add_argument("path", help="input file (.jsonl, .ndjson, .csv or .tsv)")
    estimate.add_argument(
        "--judge-col", default="judge", help="column with the judge's score (default: judge)"
    )
    estimate.add_argument(
        "--human-col",
        default="human",
        help="column with the human label, empty where there is none (default: human)",
    )
    estimate.add_argument(
        "--level", type=float, default=0.95, help="confidence level (default: 0.95)"
    )
    estimate.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="output format (default: markdown)",
    )
    estimate.add_argument(
        "--input-format",
        choices=["jsonl", "csv", "tsv"],
        default=None,
        help="input format (default: from the file extension)",
    )
    estimate.add_argument(
        "--digits", type=int, default=4, help="decimal places in the Markdown table (default: 4)"
    )
    estimate.set_defaults(run=_run_estimate)
    return parser


def _run_estimate(args: argparse.Namespace) -> str:
    if args.digits < 0:
        raise ValueError("--digits must not be negative")
    dataset = load_dataset(
        args.path, judge_col=args.judge_col, human_col=args.human_col, fmt=args.input_format
    )
    estimates = estimate_all(dataset.y, dataset.f, dataset.f_unlabelled, level=args.level)
    if args.format == "json":
        return estimate_json(dataset, estimates)
    return estimate_markdown(dataset, estimates, digits=args.digits)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit status."""
    args = build_parser().parse_args(argv)
    try:
        output = args.run(args)
    except ValueError as exc:
        print(f"ppieval: error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
