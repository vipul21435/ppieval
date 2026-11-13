"""Read rows that carry a judge score everywhere and a human label on a subset.

Two file formats are read: JSON Lines (one object per line) and delimited text
with a header row (CSV, or TSV for a ``.tsv`` file). A human label is missing
when the field is absent, empty, ``null`` or NaN.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DataError",
    "Dataset",
    "dataset_from_rows",
    "load_dataset",
    "paired_dataset_from_rows",
    "read_rows",
]

_FORMATS = {
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
}
_MISSING_WORDS = {"", "na", "nan", "null", "none"}
_BOOL_WORDS = {"true": 1.0, "false": 0.0}


class DataError(ValueError):
    """An input file or row that cannot be used."""


@dataclass(frozen=True)
class Dataset:
    """Judge scores on every row and human labels on some of them.

    ``human`` has the same length as ``judge`` and holds NaN where a row has no
    human label.
    """

    judge: NDArray[np.float64]
    human: NDArray[np.float64]
    clusters: tuple[str, ...] | None = None
    weights: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.judge.ndim != 1 or self.judge.shape != self.human.shape:
            raise DataError("judge and human must be one-dimensional and of the same length")
        if not np.all(np.isfinite(self.judge)):
            raise DataError("every row needs a finite judge score")
        if np.any(np.isinf(self.human)):
            raise DataError("human labels must be finite or missing")
        if self.clusters is not None and len(self.clusters) != self.judge.shape[0]:
            raise DataError("clusters must have one entry per row")
        if self.weights is not None and (
            self.weights.shape != self.judge.shape or not np.all(self.weights > 0.0)
        ):
            raise DataError("weights must be positive, one per row")

    def _split(self, values: Any, labelled: bool) -> Any:
        if values is None:
            return None
        mask = self.labelled if labelled else ~self.labelled
        return [values[i] for i in np.flatnonzero(mask)]

    def design(self) -> dict[str, Any]:
        """Weights and clusters split into labelled and unlabelled rows, as estimators take them."""
        return {
            "weights": self._split(self.weights, True),
            "weights_unlabelled": self._split(self.weights, False),
            "clusters": self._split(self.clusters, True),
            "clusters_unlabelled": self._split(self.clusters, False),
        }

    @property
    def labelled(self) -> NDArray[np.bool_]:
        """Mask of the rows that have a human label."""
        return ~np.isnan(self.human)

    @property
    def n_rows(self) -> int:
        return int(self.judge.shape[0])

    @property
    def n_labelled(self) -> int:
        return int(np.count_nonzero(self.labelled))

    @property
    def n_unlabelled(self) -> int:
        return self.n_rows - self.n_labelled

    @property
    def y(self) -> NDArray[np.float64]:
        """Human labels on the labelled rows."""
        return self.human[self.labelled]

    @property
    def f(self) -> NDArray[np.float64]:
        """Judge scores on the labelled rows."""
        return self.judge[self.labelled]

    @property
    def f_unlabelled(self) -> NDArray[np.float64]:
        """Judge scores on the rows without a human label."""
        return self.judge[~self.labelled]


def _parse_number(value: Any, row: int, column: str) -> float:
    """Turn one field into a float; NaN stands for a missing value."""
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in _MISSING_WORDS:
            return math.nan
        if text in _BOOL_WORDS:
            return _BOOL_WORDS[text]
        try:
            number = float(text)
        except ValueError:
            raise DataError(f"row {row}: column {column!r} is not a number: {value!r}") from None
    else:
        raise DataError(f"row {row}: column {column!r} is not a number: {value!r}")
    if math.isinf(number):
        raise DataError(f"row {row}: column {column!r} is infinite")
    return number


def _label(row: Mapping[str, Any], column: str, number: int) -> str:
    value = row.get(column)
    if value is None or str(value).strip() == "":
        raise DataError(f"row {number}: no value in column {column!r}")
    return str(value).strip()


def dataset_from_rows(
    rows: Iterable[Mapping[str, Any]],
    judge_col: str = "judge",
    human_col: str = "human",
    cluster_col: str | None = None,
    weight_col: str | None = None,
) -> Dataset:
    """Build a :class:`Dataset` from dictionaries. Row numbers in errors start at 1.

    ``cluster_col`` names a column of cluster labels (rows sharing a label are
    not independent); ``weight_col`` a column of inverse-probability weights.
    """
    judge: list[float] = []
    human: list[float] = []
    clusters: list[str] = []
    weights: list[float] = []
    human_col_seen = False
    for number, row in enumerate(rows, start=1):
        score = _parse_number(row.get(judge_col), number, judge_col)
        if math.isnan(score):
            raise DataError(f"row {number}: no judge score in column {judge_col!r}")
        judge.append(score)
        human_col_seen = human_col_seen or human_col in row
        human.append(_parse_number(row.get(human_col), number, human_col))
        if cluster_col is not None:
            clusters.append(_label(row, cluster_col, number))
        if weight_col is not None:
            weight = _parse_number(row.get(weight_col), number, weight_col)
            if not weight > 0.0:
                raise DataError(f"row {number}: column {weight_col!r} needs a positive weight")
            weights.append(weight)
    if not judge:
        raise DataError("the input has no rows")
    if not human_col_seen:
        raise DataError(f"no row has a column named {human_col!r}")
    return Dataset(
        np.asarray(judge, dtype=np.float64),
        np.asarray(human, dtype=np.float64),
        tuple(clusters) if cluster_col is not None else None,
        np.asarray(weights, dtype=np.float64) if weight_col is not None else None,
    )


def paired_dataset_from_rows(
    rows: Iterable[Mapping[str, Any]],
    systems: tuple[str, str],
    item_col: str = "item",
    system_col: str = "system",
    judge_col: str = "judge",
    human_col: str = "human",
    cluster_col: str | None = None,
) -> Dataset:
    """Differences between two systems on the items both were scored on.

    The judge value of a pair is the first system's judge score minus the
    second's; the human value is the same difference where both rows have a
    human label, and missing otherwise. Items scored for one system only are
    dropped. The mean of the result is the mean paired difference.
    """
    first, second = systems
    if first == second:
        raise DataError("the two systems must differ")
    by_item: dict[str, dict[str, tuple[float, float, str]]] = {}
    for number, row in enumerate(rows, start=1):
        system = _label(row, system_col, number)
        if system not in systems:
            continue
        item = _label(row, item_col, number)
        score = _parse_number(row.get(judge_col), number, judge_col)
        if math.isnan(score):
            raise DataError(f"row {number}: no judge score in column {judge_col!r}")
        label = _parse_number(row.get(human_col), number, human_col)
        cluster = _label(row, cluster_col, number) if cluster_col is not None else item
        if system in by_item.setdefault(item, {}):
            raise DataError(f"row {number}: item {item!r} appears twice for system {system!r}")
        by_item[item][system] = (score, label, cluster)
    shared = [pair for pair in by_item.values() if len(pair) == 2]
    if not shared:
        raise DataError(f"no item has rows for both {first!r} and {second!r}")
    judge = np.asarray([pair[first][0] - pair[second][0] for pair in shared], dtype=np.float64)
    human = np.asarray([pair[first][1] - pair[second][1] for pair in shared], dtype=np.float64)
    clusters = tuple(pair[first][2] for pair in shared) if cluster_col is not None else None
    return Dataset(judge, human, clusters)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                message = f"{path.name}: line {number} is not valid JSON ({exc.msg})"
                raise DataError(message) from None
            if not isinstance(row, dict):
                raise DataError(f"{path.name}: line {number} is not a JSON object")
            rows.append(row)
    return rows


def _read_delimited(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise DataError(f"{path.name}: the file is empty")
        rows: list[dict[str, Any]] = []
        for number, row in enumerate(reader, start=1):
            if None in row:
                raise DataError(f"{path.name}: row {number} has more fields than the header")
            rows.append(dict(row))
    return rows


def read_rows(path: str | Path, fmt: str | None = None) -> list[dict[str, Any]]:
    """Read a JSON Lines, CSV or TSV file into a list of dictionaries.

    ``fmt`` is ``"jsonl"``, ``"csv"`` or ``"tsv"``; when it is ``None`` the
    file extension decides.
    """
    path = Path(path)
    if fmt is None:
        fmt = _FORMATS.get(path.suffix.lower())
        if fmt is None:
            raise DataError(
                f"cannot tell the format of {path.name!r} from its extension; "
                "use .jsonl, .csv or .tsv, or name the format"
            )
    if fmt not in {"jsonl", "csv", "tsv"}:
        raise DataError(f"unknown format {fmt!r}; expected jsonl, csv or tsv")
    if not path.is_file():
        raise DataError(f"no such file: {path}")
    if fmt == "jsonl":
        return _read_jsonl(path)
    return _read_delimited(path, "\t" if fmt == "tsv" else ",")


def load_dataset(
    path: str | Path,
    judge_col: str = "judge",
    human_col: str = "human",
    fmt: str | None = None,
    cluster_col: str | None = None,
    weight_col: str | None = None,
) -> Dataset:
    """Read a file and pick the judge and human columns."""
    rows = read_rows(path, fmt)
    if rows and all(judge_col not in row for row in rows):
        found = ", ".join(sorted({key for row in rows[:50] for key in row}))
        raise DataError(f"no row has a column named {judge_col!r}; columns found: {found}")
    return dataset_from_rows(rows, judge_col, human_col, cluster_col, weight_col)
