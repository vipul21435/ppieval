"""ppieval: combine a few human labels with many automatic judgments."""

from __future__ import annotations

from .estimators import (
    Estimate,
    classical_mean,
    estimate_all,
    judge_only_mean,
    optimal_lambda,
    optimal_lambda_from_moments,
    ppi_mean,
    ppi_pp_mean,
    variance_from_moments,
)

__version__ = "0.1.0"

__all__ = [
    "Estimate",
    "__version__",
    "classical_mean",
    "estimate_all",
    "judge_only_mean",
    "optimal_lambda",
    "optimal_lambda_from_moments",
    "ppi_mean",
    "ppi_pp_mean",
    "variance_from_moments",
]
