"""simulate, replay, compare and plan, through the library and the command line."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ppieval.cli import main
from ppieval.data import DataError, dataset_from_rows, load_dataset, paired_dataset_from_rows
from ppieval.estimators import estimate_all
from ppieval.simulate import replay, simulate


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def judged_rows(total: int = 400, labelled: int = 400, seed: int = 0) -> list[list[object]]:
    rng = np.random.default_rng(seed)
    human = rng.random(total) < 0.6
    judge = np.where(rng.random(total) < 0.85, human, ~human)
    return [
        [f"item-{i}", f"group-{i // 4}", int(judge[i]), int(human[i]) if i < labelled else ""]
        for i in range(total)
    ]


HEADER = ["item", "group", "judge", "human"]


def test_simulate_reports_coverage_for_every_method() -> None:
    got = simulate(qualities=(0.8,), fractions=(0.1,), rows=600, reps=150, seed=1)
    assert got["command"] == "simulate" and len(got["cells"]) == 1
    cell = got["cells"][0]
    assert (cell["sensitivity"], cell["specificity"]) == pytest.approx((0.9, 0.7))
    assert cell["n_labelled"] == 60
    rows = {row["method"]: row for row in cell["methods"]}
    assert set(rows) >= {"human-only", "judge-only", "ppi", "ppi++", "ppi-stratified"}
    # 150 draws: the standard error of a coverage estimate near 0.95 is about 0.018.
    for name in ("human-only", "ppi", "ppi++", "ppi-stratified"):
        assert 0.89 <= rows[name]["coverage"] <= 1.0, name
    assert rows["judge-only"]["coverage"] < 0.2
    assert rows["human-only"]["relative_width"] == pytest.approx(1.0)
    assert rows["human-only"]["effective_n"] == pytest.approx(60)
    assert rows["ppi++"]["effective_n"] > 60
    assert simulate(qualities=(0.8,), fractions=(0.1,), rows=600, reps=150, seed=1) == got
    with pytest.raises(ValueError, match="rows >= 20"):
        simulate(rows=10)
    with pytest.raises(ValueError, match="at least 2 rows without a label"):
        simulate(fractions=(1.0,), rows=50, reps=1)


def test_effective_n_follows_the_reported_interval() -> None:
    """A Rogan-Gladen interval truncated to [0, 1] must not report the untruncated variance."""
    from ppieval.estimators import Estimate
    from ppieval.simulate import _Tally

    tally = _Tally(truth=0.5)
    tally.add(
        [
            Estimate("human-only", 0.5, 0.05, 0.4, 0.6, 0.95, 200, 0),
            Estimate("ppi++", 0.5, 0.025, 0.45, 0.55, 0.95, 200, 2000, 0.5),
            Estimate("rogan-gladen", 0.5, 5.0, 0.0, 1.0, 0.95, 200, 2000),
        ]
    )
    rows = {row["method"]: row for row in tally.summary(200)}
    assert rows["human-only"]["effective_n"] == pytest.approx(200)
    assert rows["ppi++"]["effective_n"] == pytest.approx(800)
    assert rows["rogan-gladen"]["relative_width"] == pytest.approx(5.0)
    assert rows["rogan-gladen"]["effective_n"] == pytest.approx(200 / 25)

    weak = simulate(qualities=(0.6,), fractions=(0.1,), rows=1000, reps=100, seed=3)
    by_name = {row["method"]: row for row in weak["cells"][0]["methods"]}
    truncated = by_name["rogan-gladen"]
    assert truncated["relative_width"] > 3.0
    assert truncated["effective_n"] == pytest.approx(
        100 / truncated["relative_width"] ** 2, rel=0.35
    )


def test_replay_targets_the_mean_of_all_human_labels() -> None:
    rng = np.random.default_rng(2)
    human = (rng.random(800) < 0.55).astype(float)
    judge = np.where(rng.random(800) < 0.8, human, 1.0).astype(float)
    got = replay(judge, human, fractions=(0.1, 0.25), reps=150, seed=4)
    assert got["full_data_mean"] == pytest.approx(human.mean())
    assert [cell["n_labelled"] for cell in got["cells"]] == [80, 200]
    for cell in got["cells"]:
        rows = {row["method"]: row for row in cell["methods"]}
        assert rows["judge-only"]["coverage"] in (0.0, 1.0)
        assert rows["judge-only"]["effective_n"] is None
        assert rows["ppi++"]["coverage"] >= 0.90
        assert rows["ppi++"]["relative_width"] < 1.0
        assert abs(rows["ppi++"]["mean_error"]) < 0.012
    with pytest.raises(ValueError, match="same length"):
        replay(judge, human[:-1])
    with pytest.raises(ValueError, match="at least 20"):
        replay(judge[:10], human[:10])
    with pytest.raises(ValueError, match="at least 2 rows without a label"):
        replay(judge, human, fractions=(1.0,), reps=1)


def test_a_cell_does_not_depend_on_the_other_cells_requested() -> None:
    rng = np.random.default_rng(6)
    human = (rng.random(500) < 0.4).astype(float)
    judge = np.where(rng.random(500) < 0.8, human, 1.0 - human)
    alone = replay(judge, human, fractions=(0.1,), reps=40, seed=7)["cells"][0]
    after = replay(judge, human, fractions=(0.05, 0.1), reps=40, seed=7)["cells"][1]
    assert alone == after
    assert alone != replay(judge, human, fractions=(0.1,), reps=40, seed=8)["cells"][0]

    one = simulate(qualities=(0.8,), fractions=(0.1,), rows=300, reps=30, seed=7)["cells"][0]
    grid = simulate(qualities=(0.6, 0.8), fractions=(0.05, 0.1), rows=300, reps=30, seed=7)
    assert one == grid["cells"][3]


def test_pairing_two_systems_by_item() -> None:
    rows = [
        {"item": "q1", "system": "a", "judge": 1, "human": 1},
        {"item": "q1", "system": "b", "judge": 0, "human": 1},
        {"item": "q2", "system": "a", "judge": 1, "human": ""},
        {"item": "q2", "system": "b", "judge": 1, "human": 0},
        {"item": "q3", "system": "a", "judge": 0, "human": 0},
        {"item": "q3", "system": "c", "judge": 1, "human": 1},
        {"item": "q4", "system": "b", "judge": 0.5, "human": 1},
        {"item": "q4", "system": "a", "judge": 1, "human": 0},
    ]
    data = paired_dataset_from_rows(rows, ("a", "b"))
    np.testing.assert_array_equal(data.judge, [1.0, 0.0, 0.5])
    np.testing.assert_array_equal(data.labelled, [True, False, True])
    np.testing.assert_array_equal(data.y, [0.0, -1.0])
    assert data.clusters is None
    assert paired_dataset_from_rows(rows, ("a", "b"), cluster_col="item").clusters == (
        "q1", "q2", "q4",
    )
    with pytest.raises(DataError, match="must differ"):
        paired_dataset_from_rows(rows, ("a", "a"))
    with pytest.raises(DataError, match="no item has rows for both"):
        paired_dataset_from_rows(rows, ("b", "c"))
    with pytest.raises(DataError, match="appears twice"):
        paired_dataset_from_rows([*rows, rows[0]], ("a", "b"))
    with pytest.raises(DataError, match="no value in column 'system'"):
        paired_dataset_from_rows([{"item": "q1", "judge": 1}], ("a", "b"))
    with pytest.raises(DataError, match="no judge score"):
        paired_dataset_from_rows([{"item": "q1", "system": "a", "human": 1}], ("a", "b"))


def test_two_system_difference_covers_the_true_difference() -> None:
    """Paired differences through the same code path as ``ppieval compare``; 150 draws."""
    rng = np.random.default_rng(12)
    truth, reps, items, labelled = 0.15, 150, 400, 80
    hits = {"human-only": 0, "ppi++": 0, "judge-only": 0}
    for _ in range(reps):
        rows = []
        for system, skill in (("new", 0.70), ("old", 0.55)):
            human = rng.random(items) < skill
            # The judge passes too often, and more so for the new system.
            judge = human | (rng.random(items) < (0.45 if system == "new" else 0.15))
            for i in range(items):
                label = int(human[i]) if i < labelled else ""
                rows.append({"item": i, "system": system, "judge": int(judge[i]), "human": label})
        data = paired_dataset_from_rows(rows, ("new", "old"))
        for est in estimate_all(data.y, data.f, data.f_unlabelled):
            if est.method in hits:
                hits[est.method] += est.lo <= truth <= est.hi
    # 150 draws: the standard error of a coverage estimate near 0.95 is about 0.018.
    assert 0.89 <= hits["human-only"] / reps <= 1.0
    assert 0.89 <= hits["ppi++"] / reps <= 1.0
    assert hits["judge-only"] / reps < 0.5


def test_cluster_and_weight_columns() -> None:
    rows = [
        {"judge": 1, "human": 1, "group": "x", "w": 2},
        {"judge": 0, "human": "", "group": "x", "w": 1.5},
        {"judge": 1, "human": 0, "group": "y", "w": 4},
    ]
    data = dataset_from_rows(rows, cluster_col="group", weight_col="w")
    design = data.design()
    assert design["clusters"] == ["x", "y"] and design["clusters_unlabelled"] == ["x"]
    assert design["weights"] == [2.0, 4.0] and design["weights_unlabelled"] == [1.5]
    assert dataset_from_rows(rows).design() == dict.fromkeys(design)
    with pytest.raises(DataError, match="positive weight"):
        dataset_from_rows([{"judge": 1, "human": 1, "w": 0}], weight_col="w")
    with pytest.raises(DataError, match="no value in column 'group'"):
        dataset_from_rows([{"judge": 1, "human": 1}], cluster_col="group")


def test_compare_command(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rng = np.random.default_rng(3)
    rows = []
    for i in range(300):
        for system, skill in (("new", 0.7), ("old", 0.55)):
            human = int(rng.random() < skill)
            judge = human if rng.random() < 0.85 else 1 - human
            rows.append([f"q{i}", system, judge, human if i < 80 else ""])
    path = write_csv(tmp_path / "two.csv", ["item", "system", "judge", "human"], rows)
    assert main(["compare", str(path), "--systems", "new", "old", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "compare"
    assert (payload["rows"], payload["labelled"]) == (300, 80)
    methods = [row["method"] for row in payload["estimates"]]
    assert methods[:4] == ["human-only", "judge-only", "ppi", "ppi++"]
    assert "rogan-gladen" not in methods
    assert main(["compare", str(path), "--systems", "new", "old"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# ppieval compare\n")
    assert "- difference: new minus old, on the items scored for both" in out
    assert main(["compare", str(path), "--systems", "new", "none"]) == 2
    assert "no item has rows for both" in capsys.readouterr().err


def test_estimate_with_clusters_bootstrap_and_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = write_csv(tmp_path / "rows.csv", HEADER, judged_rows(labelled=80))
    args = ["estimate", str(path), "--cluster-col", "group", "--bootstrap", "40", "--seed", "2"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "| ppi++ (bootstrap) |" in out and "| ppi-stratified |" not in out
    assert "## Diagnostics" in out and "- rectifier (mean of judge minus human):" in out
    assert "KS statistic" in out
    assert main([*args, "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    data = load_dataset(path, cluster_col="group")
    expected = estimate_all(data.y, data.f, data.f_unlabelled, n_boot=40, seed=2, **data.design())
    assert [row["se"] for row in payload["estimates"]] == [est.se for est in expected]
    assert payload["diagnostics"]["agreement"] == pytest.approx(np.mean(data.y == data.f))


def test_plan_command_and_weighted_estimate(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = write_csv(tmp_path / "rows.csv", HEADER, judged_rows(labelled=80))
    assert main(["plan", str(path), "--target-width", "0.1"]) == 0
    out = capsys.readouterr().out
    assert "- effective sample size:" in out and "## Target width 0.1" in out
    assert "- human labels needed with the judge:" in out
    assert main(["plan", str(path), "--target-width", "0.001"]) == 0
    assert "not reachable even with every row labelled" in capsys.readouterr().out

    more = tmp_path / "more.jsonl"
    assert main(["plan", str(path), "--sample", "50", "--out", str(more), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample"]["size"] == 50 and len(payload["sample"]["rows_chosen"]) == 50
    marked = [json.loads(line) for line in more.read_text().splitlines()]
    assert sum(row["ppieval_sampled"] for row in marked) == 50
    assert all(row["ppieval_weight"] == "" for row in marked[:80])
    assert main(["plan", str(path), "--sample", "50", "--method", "stratified"]) == 2
    assert "needs a file without human labels" in capsys.readouterr().err
    assert main(["plan", str(path), "--sample", "5", "--out", str(tmp_path / "x.txt")]) == 2
    assert "--out must end in" in capsys.readouterr().err

    # No labels yet: choose rows by judge score, label them, estimate with the weights.
    rows = judged_rows(total=600, labelled=600, seed=5)
    blank = write_csv(tmp_path / "blank.csv", HEADER, [[*row[:3], ""] for row in rows])
    assert main(["plan", str(blank)]) == 2
    assert "use --sample" in capsys.readouterr().err
    chosen = tmp_path / "chosen.csv"
    args = ["plan", str(blank), "--sample", "120", "--method", "stratified"]
    assert main([*args, "--allocation", "equal", "--out", str(chosen)]) == 0
    out = capsys.readouterr().out
    assert "## Sample to label" in out and "equal allocation" in out and "| stratum |" in out
    with chosen.open(newline="", encoding="utf-8") as handle:
        marked_rows = list(csv.DictReader(handle))
    for row, original in zip(marked_rows, rows, strict=True):
        if row["ppieval_sampled"] == "1":
            row["human"] = str(original[3])
    labelled = tmp_path / "labelled.csv"
    with labelled.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(marked_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(marked_rows)
    args = ["estimate", str(labelled), "--weight-col", "ppieval_weight", "--format", "json"]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    tuned = next(row for row in payload["estimates"] if row["method"] == "ppi++")
    truth = np.mean([row[3] for row in rows])
    assert payload["labelled"] == 120 and abs(tuned["estimate"] - truth) < 3 * tuned["se"]
    assert main([*args, "--bootstrap", "20"]) == 2
    assert "not available with weights" in capsys.readouterr().err


def test_simulate_and_replay_commands(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    args = ["simulate", "--qualities", "0.8", "--fractions", "0.1", "--rows", "300", "--reps", "20"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("# ppieval simulate\n")
    assert "## judge quality 0.8 (sensitivity 0.9, specificity 0.7), 10 % of human labels" in out
    assert main([*args, "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["reps"] == 20

    full = write_csv(tmp_path / "full.csv", HEADER, judged_rows())
    args = ["replay", str(full), "--fractions", "0.1", "0.25", "--reps", "30"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "- rows, all with a human label: 400" in out
    assert "## 10 % of human labels kept (n = 40)" in out and "| ppi++ |" in out
    assert main([*args, "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "replay" and len(payload["cells"]) == 2
    part = write_csv(tmp_path / "part.csv", HEADER, judged_rows(labelled=300))
    assert main(["replay", str(part)]) == 2
    assert "needs a human label on every row; 100 rows have none" in capsys.readouterr().err
