from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ppieval import __version__, estimate_all
from ppieval.cli import main
from ppieval.data import load_dataset

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "small_example.csv"
METHODS = ["human-only", "judge-only", "ppi", "ppi++", "ppi-stratified", "rogan-gladen"]


def write_rows(
    path: Path,
    rows: int = 300,
    labelled: int = 60,
    judge_col: str = "judge",
    human_col: str = "human",
) -> Path:
    """A CSV with a judge that agrees with the human label 85 % of the time."""
    rng = np.random.default_rng(0)
    human = rng.random(rows) < 0.6
    judge = np.where(rng.random(rows) < 0.85, human, ~human)
    lines = [f"{judge_col},{human_col}"]
    for index in range(rows):
        label = str(int(human[index])) if index < labelled else ""
        lines.append(f"{int(judge[index])},{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_estimate_prints_a_markdown_table(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = write_rows(tmp_path / "rows.csv")
    assert main(["estimate", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# ppieval estimate\n")
    assert "- rows: 300\n- with a human label: 60\n- without a human label: 240\n" in out
    assert "- interval level: 95 %" in out
    table = [line for line in out.splitlines() if line.startswith("|")]
    assert table[0].startswith("| method | estimate | std. error | 95 % interval |")
    assert [line.split("|")[1].strip() for line in table[2:]] == METHODS
    assert "judge-only takes the judge's scores as if they were human labels" in out


def test_estimate_json_matches_the_library(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = write_rows(tmp_path / "rows.csv")
    assert main(["estimate", str(path), "--format", "json", "--level", "0.9"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "estimate"
    assert (payload["rows"], payload["labelled"], payload["unlabelled"]) == (300, 60, 240)
    assert payload["level"] == 0.9
    data = load_dataset(path)
    expected = estimate_all(data.y, data.f, data.f_unlabelled, level=0.9)
    assert [row["method"] for row in payload["estimates"]] == METHODS
    for row, est in zip(payload["estimates"], expected, strict=True):
        assert row == est.to_dict()


def test_column_names_input_format_and_digits(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = write_rows(tmp_path / "rows.txt", judge_col="auto", human_col="gold")
    args = ["estimate", str(path), "--judge-col", "auto", "--human-col", "gold"]
    assert main([*args, "--input-format", "csv", "--digits", "2"]) == 0
    out = capsys.readouterr().out
    human_row = next(line for line in out.splitlines() if line.startswith("| human-only"))
    cells = [cell.strip() for cell in human_row.strip("|").split("|")]
    assert len(cells[1].split(".")[1]) == 2
    assert cells[5] == "1.00"


def test_all_rows_labelled_leaves_ppi_out(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = write_rows(tmp_path / "rows.csv", rows=40, labelled=40)
    assert main(["estimate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "| ppi |" not in out
    assert "| ppi++ |" in out
    assert "ppi is left out and ppi++ equals human-only" in out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--level", "1.5"], "level must be strictly between 0 and 1"),
        (["--digits", "-1"], "--digits must not be negative"),
        (["--judge-col", "score"], "no row has a column named 'score'"),
        (["--input-format", "jsonl"], "line 1 is not valid JSON"),
    ],
)
def test_errors_go_to_stderr_with_status_2(
    tmp_path: Path, capsys: pytest.CaptureFixture, extra: list[str], message: str
) -> None:
    path = write_rows(tmp_path / "rows.csv")
    assert main(["estimate", str(path), *extra]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("ppieval: error: ")
    assert message in captured.err


def test_missing_file_and_too_few_labels(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["estimate", str(tmp_path / "absent.csv")]) == 2
    assert "no such file" in capsys.readouterr().err
    path = write_rows(tmp_path / "rows.csv", rows=30, labelled=1)
    assert main(["estimate", str(path)]) == 2
    assert "at least 2 labelled rows" in capsys.readouterr().err


def test_version_and_missing_command(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.strip() == f"ppieval {__version__}"
    with pytest.raises(SystemExit) as usage_exit:
        main([])
    assert usage_exit.value.code == 2


def test_module_entry_point_runs_the_example() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "ppieval", "estimate", str(EXAMPLE), "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    assert (payload["rows"], payload["labelled"]) == (1000, 100)
    # The numbers quoted in the README quickstart.
    shown = {row["method"]: round(row["estimate"], 4) for row in payload["estimates"]}
    assert shown == {
        "human-only": 0.67,
        "judge-only": 0.777,
        "ppi": 0.7,
        "ppi++": 0.6869,
        "ppi-stratified": 0.6855,
        "rogan-gladen": 0.7317,
    }


def test_example_generator_writes_the_documented_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    module = runpy.run_path(str(REPO / "examples" / "make_small_example.py"))
    out = module["main"](tmp_path / "generated.csv")
    assert "1000 rows, 100 with a human label" in capsys.readouterr().out
    data = load_dataset(out)
    assert (data.n_rows, data.n_labelled) == (1000, 100)
    assert set(np.unique(data.judge)) == {0.0, 1.0}
    assert set(np.unique(data.y)) == {0.0, 1.0}
