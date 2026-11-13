"""Checks that keep the documentation in step with the repository."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_derived_data_carries_its_attribution() -> None:
    text = (REPO / "data" / "README.md").read_text(encoding="utf-8")
    assert "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments" in text
    assert "https://creativecommons.org/licenses/by/4.0/" in text
    assert "Changes made to the\noriginal" in text or "Changes made to the original" in text
    assert "https://creativecommons.org/licenses/by/4.0/" in (REPO / "README.md").read_text(
        encoding="utf-8"
    )
