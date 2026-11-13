"""Estimate a pass rate from 100 human labels and 1,000 judge verdicts, from Python.

Run from the repository root:

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from ppieval import diagnose, effective_sample_size, estimate_all, load_dataset


def main() -> None:
    data = load_dataset(Path(__file__).with_name("small_example.csv"))
    y, f, f_unlabelled = data.y, data.f, data.f_unlabelled

    for est in estimate_all(y, f, f_unlabelled):
        print(f"{est.method:15s} {est.estimate:.4f}  [{est.lo:.4f}, {est.hi:.4f}]")

    checks = diagnose(y, f, f_unlabelled)
    print(f"judge minus human on the labelled rows: {checks.rectifier.estimate:+.4f}")
    print(f"labelled rows look like a random subset: {not checks.warnings}")
    print(f"the judge makes {y.size} labels worth {effective_sample_size(y, f, f_unlabelled):.0f}")


if __name__ == "__main__":
    main()
