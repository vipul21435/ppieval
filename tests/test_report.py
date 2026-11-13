from __future__ import annotations

import json

import numpy as np
import pytest

from ppieval import classical_mean, estimate_all, ppi_mean
from ppieval.data import Dataset
from ppieval.report import estimate_json, estimate_markdown, estimates_table


def small_dataset() -> Dataset:
    judge = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    human = np.array([1.0, 0.0, 0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    return Dataset(judge, human)


def test_table_has_one_row_per_estimate_and_the_width_ratio() -> None:
    data = small_dataset()
    estimates = estimate_all(data.y, data.f, data.f_unlabelled)
    lines = estimates_table(estimates, digits=3).splitlines()
    assert len(lines) == 2 + len(estimates)
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]
    assert [row[0] for row in rows] == ["human-only", "judge-only", "ppi", "ppi++"]
    human_only = estimates[0]
    assert rows[0][1] == f"{human_only.estimate:.3f}"
    assert rows[0][3] == f"[{human_only.lo:.3f}, {human_only.hi:.3f}]"
    assert rows[0][5] == "1.00"
    assert rows[2][5] == f"{estimates[2].width / human_only.width:.2f}"
    assert rows[0][6] == "" and rows[2][6] == "1.000"


def test_table_without_a_human_only_row_leaves_the_ratio_empty() -> None:
    data = small_dataset()
    line = estimates_table([ppi_mean(data.y, data.f, data.f_unlabelled)]).splitlines()[2]
    assert [cell.strip() for cell in line.strip("|").split("|")][5] == ""


def test_table_with_a_zero_width_reference_leaves_the_ratio_empty() -> None:
    line = estimates_table([classical_mean([1.0, 1.0, 1.0])]).splitlines()[2]
    assert [cell.strip() for cell in line.strip("|").split("|")][5] == ""


def test_empty_table_is_an_error() -> None:
    with pytest.raises(ValueError, match="no estimates"):
        estimates_table([])


def test_markdown_and_json_agree_on_the_counts() -> None:
    data = small_dataset()
    estimates = estimate_all(data.y, data.f, data.f_unlabelled, level=0.9)
    text = estimate_markdown(data, estimates)
    assert "- rows: 10" in text and "- with a human label: 4" in text
    assert "- interval level: 90 %" in text
    assert text.endswith("\n")
    payload = json.loads(estimate_json(data, estimates))
    assert (payload["rows"], payload["labelled"], payload["unlabelled"]) == (10, 4, 6)
    assert len(payload["estimates"]) == 4
    assert payload["estimates"][3]["lam"] == pytest.approx(estimates[3].lam)
