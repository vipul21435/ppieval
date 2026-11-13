from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppieval.data import DataError, Dataset, dataset_from_rows, load_dataset, read_rows


def test_csv_with_empty_human_fields(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("item,judge,human\na,1,1\nb,0,\nc,0.5, \nd,1,0\n", encoding="utf-8")
    data = load_dataset(path)
    assert data.n_rows == 4
    assert data.n_labelled == 2
    assert data.n_unlabelled == 2
    np.testing.assert_array_equal(data.judge, [1.0, 0.0, 0.5, 1.0])
    np.testing.assert_array_equal(data.labelled, [True, False, False, True])
    np.testing.assert_array_equal(data.y, [1.0, 0.0])
    np.testing.assert_array_equal(data.f, [1.0, 1.0])
    np.testing.assert_array_equal(data.f_unlabelled, [0.0, 0.5])


def test_jsonl_with_null_missing_and_boolean_values(tmp_path: Path) -> None:
    rows = [
        {"judge": True, "human": False},
        {"judge": 0.25, "human": None},
        {"judge": "0.75"},
        {"judge": 1, "human": "true"},
        {"judge": 0, "human": "NA"},
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")
    data = load_dataset(path)
    np.testing.assert_array_equal(data.judge, [1.0, 0.25, 0.75, 1.0, 0.0])
    np.testing.assert_array_equal(data.labelled, [True, False, False, True, False])
    np.testing.assert_array_equal(data.y, [0.0, 1.0])


def test_tsv_and_custom_column_names(tmp_path: Path) -> None:
    path = tmp_path / "rows.tsv"
    path.write_text("score\tgold\n3.5\t4\n2.0\t\n", encoding="utf-8")
    data = load_dataset(path, judge_col="score", human_col="gold")
    np.testing.assert_array_equal(data.judge, [3.5, 2.0])
    np.testing.assert_array_equal(data.y, [4.0])


def test_format_can_be_named_for_an_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "rows.txt"
    path.write_text("judge,human\n1,1\n0,\n", encoding="utf-8")
    assert load_dataset(path, fmt="csv").n_rows == 2
    with pytest.raises(DataError, match="cannot tell the format"):
        load_dataset(path)
    with pytest.raises(DataError, match="unknown format"):
        load_dataset(path, fmt="parquet")


def test_byte_order_mark_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes("﻿judge,human\n1,1\n0,0\n".encode())
    assert load_dataset(path).n_labelled == 2


def test_nan_human_label_counts_as_missing() -> None:
    data = dataset_from_rows([{"judge": 1, "human": float("nan")}, {"judge": 0, "human": 1}])
    np.testing.assert_array_equal(data.labelled, [False, True])


@pytest.mark.parametrize(
    ("text", "name", "message"),
    [
        ("judge,human\n1,1\n,0\n", "a.csv", "row 2: no judge score"),
        ("judge,human\n1,1\nhigh,0\n", "a.csv", "row 2: column 'judge' is not a number"),
        ("judge,human\n1,maybe\n", "a.csv", "row 1: column 'human' is not a number"),
        ("judge,human\n1,inf\n", "a.csv", "row 1: column 'human' is infinite"),
        ("judge,human\n1,1,9\n", "a.csv", "row 1 has more fields than the header"),
        ("judge,human\n", "a.csv", "no rows"),
        ("", "a.csv", "empty"),
        ("score,human\n1,1\n", "a.csv", "no row has a column named 'judge'; columns found"),
        ("judge,gold\n1,1\n", "a.csv", "no row has a column named 'human'"),
        ('{"judge": 1, "human": 1}\n{"judge": 1,\n', "a.jsonl", "line 2 is not valid JSON"),
        ("[1, 2]\n", "a.jsonl", "line 1 is not a JSON object"),
        ('{"judge": [1], "human": 1}\n', "a.jsonl", "row 1: column 'judge' is not a number"),
    ],
)
def test_bad_files_are_rejected_with_the_row(
    tmp_path: Path, text: str, name: str, message: str
) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DataError, match=message):
        load_dataset(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no such file"):
        read_rows(tmp_path / "absent.csv")


def test_dataset_checks_its_arrays() -> None:
    with pytest.raises(DataError, match="same length"):
        Dataset(np.array([1.0, 0.0]), np.array([1.0]))
    with pytest.raises(DataError, match="finite judge score"):
        Dataset(np.array([1.0, np.nan]), np.array([1.0, 0.0]))
    with pytest.raises(DataError, match="finite or missing"):
        Dataset(np.array([1.0, 0.0]), np.array([np.inf, 0.0]))
