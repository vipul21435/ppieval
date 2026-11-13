"""Download the MT-Bench human judgments and derive the tables the replay reads.

Source: the ``human`` and ``gpt4_pair`` splits of
https://huggingface.co/datasets/lmsys/mt_bench_human_judgments (licence CC-BY-4.0
on its dataset card; Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and
Chatbot Arena", arXiv:2306.05685). Only the verdict columns are kept; the
conversation texts are not stored.

Run from the repository root. With pyarrow available the two Parquet files of
the dataset are downloaded (two requests):

    uv run --with pyarrow python scripts/fetch_mtbench.py --cache ../mtbench_cache --out data

Without pyarrow the script pages through the dataset viewer's rows API with the
standard library only, which is slower and may be throttled.

Outputs:

``mtbench_votes.csv``
    one row per human vote that has a GPT-4 pairwise verdict on the same
    question, turn and pair of models. The two models are in alphabetical
    order; ``human_winner`` and ``gpt4_winner`` hold a model name or ``tie``.
``mtbench_winrate_<model>.csv``
    the votes that involve ``<model>``, scored from its side: 1 for a win,
    0.5 for a tie, 0 for a loss, in ``human`` (the person) and ``judge``
    (GPT-4). ``cluster`` is the question.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
PARQUET = (
    "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/"
    "refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
)
DATASET = "lmsys/mt_bench_human_judgments"
KEEP = ("question_id", "model_a", "model_b", "winner", "judge", "turn")
PAGE = 100


def get_json(url: str, attempts: int = 12) -> dict:
    """GET a JSON document, waiting and retrying when the server throttles or drops the call."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            wait = float(error.headers.get("Retry-After") or 20.0)
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            wait = 5.0
        time.sleep(min(120.0, wait))
    raise RuntimeError("unreachable")


def rows_from_parquet(split: str, cache: Path) -> list[dict] | None:
    """Rows of one split from its Parquet file, or None when pyarrow is not installed."""
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return None
    path = cache / f"mt_bench_{split}.parquet"
    if not path.exists():
        with urllib.request.urlopen(PARQUET.format(split=split), timeout=120) as response:
            path.write_bytes(response.read())
    return parquet.read_table(path, columns=list(KEEP)).to_pylist()


def rows_from_api(split: str) -> list[dict]:
    """Rows of one split from the dataset viewer's rows API, 100 at a time."""
    rows: list[dict] = []
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": split,
                "offset": len(rows),
                "length": PAGE,
            }
        )
        payload = get_json(f"{API}?{query}")
        batch = [{key: item["row"][key] for key in KEEP} for item in payload["rows"]]
        rows.extend(batch)
        if not batch or len(rows) >= payload["num_rows_total"]:
            return rows
        time.sleep(0.5)


def fetch_split(split: str, cache: Path) -> list[dict]:
    """All rows of one split, verdict columns only, cached as JSON Lines."""
    path = cache / f"mt_bench_{split}.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    cache.mkdir(parents=True, exist_ok=True)
    rows = rows_from_parquet(split, cache)
    if rows is None:
        rows = rows_from_api(split)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return rows


def verdict(row: dict) -> tuple[tuple[int, int, str, str], str]:
    """Key (question, turn, the two models in alphabetical order) and the winner's name."""
    first, second = sorted([row["model_a"], row["model_b"]])
    winner = {"model_a": row["model_a"], "model_b": row["model_b"]}.get(row["winner"], "tie")
    return (int(row["question_id"]), int(row["turn"]), first, second), winner


def score(winner: str, model: str) -> float:
    return 0.5 if winner == "tie" else float(winner == model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True, help="directory for the download")
    parser.add_argument("--out", type=Path, default=Path("data"), help="directory for the tables")
    args = parser.parse_args()

    human = fetch_split("human", args.cache)
    gpt4 = fetch_split("gpt4_pair", args.cache)
    judged: dict[tuple[int, int, str, str], str] = {}
    for row in gpt4:
        key, winner = verdict(row)
        if judged.setdefault(key, winner) != winner:
            raise SystemExit(f"conflicting GPT-4 verdicts for {key}")

    votes = []
    for row in human:
        key, winner = verdict(row)
        if key in judged:
            votes.append((*key, row["judge"], winner, judged[key]))
    votes.sort()

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "mtbench_votes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["question_id", "turn", "model_a", "model_b", "voter", "human_winner", "gpt4_winner"]
        )
        writer.writerows(votes)

    models = sorted({name for vote in votes for name in vote[2:4]})
    print(f"human votes: {len(human)}; GPT-4 pair verdicts: {len(gpt4)}; matched: {len(votes)}")
    for model in models:
        mine = [vote for vote in votes if model in vote[2:4]]
        path = args.out / f"mtbench_winrate_{model}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["question_id", "turn", "opponent", "cluster", "judge", "human"])
            for question, turn, first, second, _voter, human_winner, gpt4_winner in mine:
                opponent = second if first == model else first
                writer.writerow(
                    [question, turn, opponent, f"q{question}", score(gpt4_winner, model),
                     score(human_winner, model)]
                )
        human_rate = sum(score(vote[5], model) for vote in mine) / len(mine)
        judge_rate = sum(score(vote[6], model) for vote in mine) / len(mine)
        print(f"{model}: {len(mine)} votes, human {human_rate:.4f}, GPT-4 {judge_rate:.4f}")


if __name__ == "__main__":
    main()
