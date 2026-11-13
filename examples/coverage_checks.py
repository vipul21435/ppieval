"""Coverage of the 95 % intervals where the normal approximation is under strain.

Five seeded checks, each with ``M = 2000`` rows per repetition:

1. ten human labels, a good 0/1 judge (sensitivity 0.95, specificity 0.90), P(label = 1) = 0.5;
2. a rare class: P(label = 1) = 0.03 with 100 human labels and the same judge;
3. skewed continuous labels (log-normal) with 30 human labels;
4. every row labelled independently with probability 0.25 or 0.04 according to the judge's
   verdict, estimated with the correct inverse-probability weights;
5. a fixed-size sample of 250 rows, half from each verdict, with its weights.

    python examples/coverage_checks.py
"""

from __future__ import annotations

import math

import numpy as np

from ppieval import (
    classical_mean,
    estimate_all,
    ppi_mean,
    ppi_pp_mean,
    sample_to_label,
    simulate,
)

ROWS = 2000


def binary_case(n: int, prevalence: float, reps: int = 1000, seed: int = 0) -> dict[str, float]:
    """Coverage by method for a 0/1 judge with sensitivity 0.95 and specificity 0.90."""
    result = simulate(
        qualities=(0.925,), bias=0.025, fractions=(n / ROWS,), rows=ROWS,
        prevalence=prevalence, reps=reps, seed=seed,
    )
    return {row["method"]: row["coverage"] for row in result["cells"][0]["methods"]}


def skewed_case(n: int = 30, reps: int = 1000, seed: int = 0) -> dict[str, float]:
    """Coverage by method for log-normal labels and a judge that tracks them with noise."""
    rng = np.random.default_rng(seed)
    truth = math.exp(0.5)
    hits: dict[str, int] = {}
    for _ in range(reps):
        y = rng.lognormal(0.0, 1.0, ROWS)
        f = 0.8 * y + rng.normal(0.0, 0.5, ROWS)
        for est in estimate_all(y[:n], f[:n], f[n:]):
            hits[est.method] = hits.get(est.method, 0) + (est.lo <= truth <= est.hi)
    return {name: count / reps for name, count in hits.items() if name != "judge-only"}


def weighted_case(fixed_size: bool, reps: int = 3000, seed: int = 0) -> dict[str, float]:
    """Coverage of the weighted estimators when the judge's verdict decides who gets labelled.

    ``fixed_size=False`` labels each row independently, with probability 0.25 after a
    verdict of 0 and 0.04 after a verdict of 1. ``fixed_size=True`` draws exactly 125 rows
    from each verdict with :func:`ppieval.sample_to_label`.
    """
    rng = np.random.default_rng(seed)
    truth = 0.5
    hits = {"human-only": 0, "ppi": 0, "ppi++": 0}
    for _ in range(reps):
        y = (rng.random(ROWS) < truth).astype(float)
        draw = rng.random(ROWS)
        f = np.where(y == 1.0, draw < 0.9, draw >= 0.7).astype(float)
        if fixed_size:
            sample = sample_to_label(
                f, 250, method="stratified", allocation="equal", seed=int(rng.integers(2**31))
            )
            lab, weight = sample.selected, sample.weight
        else:
            chance = np.where(f == 0.0, 0.25, 0.04)
            lab = rng.random(ROWS) < chance
            weight = np.where(lab, 1.0 / chance, 1.0 / (1.0 - chance))
        design = {"weights": weight[lab], "weights_unlabelled": weight[~lab]}
        estimates = [
            classical_mean(y[lab], weights=design["weights"]),
            ppi_mean(y[lab], f[lab], f[~lab], **design),
            ppi_pp_mean(y[lab], f[lab], f[~lab], **design),
        ]
        for est in estimates:
            hits[est.method] += est.lo <= truth <= est.hi
    return {name: count / reps for name, count in hits.items()}


def main() -> None:
    cases = {
        "10 labels, P(1) = 0.5": binary_case(10, 0.5),
        "100 labels, P(1) = 0.03": binary_case(100, 0.03),
        "30 labels, log-normal": skewed_case(),
        "weighted, each row labelled independently": weighted_case(fixed_size=False),
        "weighted, fixed-size sample by verdict": weighted_case(fixed_size=True),
    }
    names = ["human-only", "ppi", "ppi++", "ppi-stratified", "rogan-gladen"]
    print("| case | " + " | ".join(names) + " |")
    print("| --- | " + " | ".join("---:" for _ in names) + " |")
    for label, coverage in cases.items():
        cells = [f"{coverage[name]:.3f}" if name in coverage else "" for name in names]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
