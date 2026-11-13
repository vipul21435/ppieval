"""Write ``small_example.csv``: simulated pass/fail labels and a judge that passes too often.

Every row gets a judge verdict; only some rows keep their human label, chosen
uniformly at random. Nothing here is real data. Run it from anywhere:

    python examples/make_small_example.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROWS = 1000
LABELLED = 100
ACCURACY = 0.70  # probability that the human label is 1
FALSE_POSITIVE = 0.40  # judge says 1 where the human says 0
FALSE_NEGATIVE = 0.05  # judge says 0 where the human says 1
SEED = 0


def main(out: Path | None = None) -> Path:
    rng = np.random.default_rng(SEED)
    human = rng.random(ROWS) < ACCURACY
    draw = rng.random(ROWS)
    judge = np.where(human, draw >= FALSE_NEGATIVE, draw < FALSE_POSITIVE)
    keep = np.zeros(ROWS, dtype=bool)
    keep[rng.choice(ROWS, size=LABELLED, replace=False)] = True

    out = Path(__file__).with_name("small_example.csv") if out is None else out
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item", "judge", "human"])
        for index in range(ROWS):
            label = int(human[index]) if keep[index] else ""
            writer.writerow([f"item-{index + 1:04d}", int(judge[index]), label])

    print(f"wrote {out.name}: {ROWS} rows, {LABELLED} with a human label")
    print(f"simulated probability of a human 1: {ACCURACY}")
    print(f"mean of all {ROWS} simulated human labels, hidden ones included: {human.mean():.4f}")
    print(f"mean judge score over all rows: {judge.mean():.4f}")
    return out


if __name__ == "__main__":
    main()
