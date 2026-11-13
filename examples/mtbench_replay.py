"""Replay on the MT-Bench human judgments, one win rate per model.

For each of the six models: keep 10 % of the human votes at random, 1,000
times, and compare the intervals with the win rate over all human votes.
Reads the tables in ``data/`` (see ``data/README.md`` for their source).

    python examples/mtbench_replay.py
"""

from __future__ import annotations

from pathlib import Path

from ppieval import load_dataset, replay

DATA = Path(__file__).resolve().parents[1] / "data"
FRACTION, REPS, SEED = 0.10, 1000, 0


def main() -> None:
    print("| model | votes | human win rate | GPT-4 judge | judge-only covers | "
          "ppi++ coverage | ppi++ width vs human-only | effective n (of n) |")
    print("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for path in sorted(DATA.glob("mtbench_winrate_*.csv")):
        data = load_dataset(path)
        result = replay(data.judge, data.human, fractions=(FRACTION,), reps=REPS, seed=SEED)
        cell = result["cells"][0]
        rows = {row["method"]: row for row in cell["methods"]}
        tuned = rows["ppi++"]
        print(
            f"| {path.stem.removeprefix('mtbench_winrate_')} | {result['rows']} "
            f"| {result['full_data_mean']:.4f} | {result['judge_mean']:.4f} "
            f"| {'yes' if rows['judge-only']['coverage'] == 1.0 else 'no'} "
            f"| {tuned['coverage']:.3f} | {tuned['relative_width']:.2f} "
            f"| {tuned['effective_n']:.0f} ({cell['n_labelled']}) |"
        )


if __name__ == "__main__":
    main()
