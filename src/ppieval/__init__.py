"""ppieval: combine a few human labels with many automatic judgments."""

from __future__ import annotations

from .data import DataError, Dataset, dataset_from_rows, load_dataset, paired_dataset_from_rows
from .diagnostics import Diagnostics, diagnose, ks_two_sample
from .estimators import (
    Estimate,
    bootstrap_interval,
    classical_mean,
    estimate_all,
    judge_only_mean,
    optimal_lambda,
    optimal_lambda_from_moments,
    ppi_mean,
    ppi_pp_mean,
    rogan_gladen_mean,
    stratified_ppi_mean,
    variance_from_moments,
)
from .planning import (
    Plan,
    Sample,
    effective_sample_size,
    effective_sample_size_from_correlation,
    labels_needed,
    make_plan,
    sample_to_label,
)
from .simulate import replay, simulate

__version__ = "0.1.0"

__all__ = [
    "DataError",
    "Dataset",
    "Diagnostics",
    "Estimate",
    "Plan",
    "Sample",
    "__version__",
    "bootstrap_interval",
    "classical_mean",
    "dataset_from_rows",
    "diagnose",
    "effective_sample_size",
    "effective_sample_size_from_correlation",
    "estimate_all",
    "judge_only_mean",
    "ks_two_sample",
    "labels_needed",
    "load_dataset",
    "make_plan",
    "optimal_lambda",
    "optimal_lambda_from_moments",
    "paired_dataset_from_rows",
    "ppi_mean",
    "ppi_pp_mean",
    "replay",
    "rogan_gladen_mean",
    "sample_to_label",
    "simulate",
    "stratified_ppi_mean",
    "variance_from_moments",
]
