"""Comparison with the ppi-python package (import name ``ppi_py``).

The package is a test-only dependency in the ``reference`` extra; the tests are
skipped when it is not installed. Two conventions differ and are handled here:

* ppi_py uses variances with an ``n`` denominator, ppieval ``n - 1``;
* when it estimates the tuning parameter, ppi_py uses a covariance with an
  ``n`` denominator and clips the result to [0, 1]; ppieval uses ``n - 1`` and
  does not clip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ppieval.estimators import classical_mean, optimal_lambda, ppi_mean, ppi_pp_mean

ppi_py = pytest.importorskip("ppi_py")


def sample(seed: int, n: int = 150, n_unl: int = 1200, slope: float = 0.7):
    rng = np.random.default_rng(seed)
    y = rng.normal(1.0, 1.0, n + n_unl)
    f = 0.3 + slope * y + rng.normal(0.0, 0.7, n + n_unl)
    return y[:n], f[:n], f[n:]


@pytest.mark.parametrize("lam", [1.0, 0.37, 0.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_point_estimate_and_interval_with_a_fixed_lambda(seed: int, lam: float) -> None:
    y, f, fu = sample(seed)
    n, n_unl = y.size, fu.size
    mine = ppi_mean(y, f, fu, lam=lam, level=0.9)
    theirs = float(np.squeeze(ppi_py.ppi_mean_pointestimate(y, f, fu, lam=lam)))
    assert mine.estimate == pytest.approx(theirs, rel=1e-10)
    lo, hi = (float(np.squeeze(v)) for v in ppi_py.ppi_mean_ci(y, f, fu, alpha=0.1, lam=lam))
    their_se = (hi - lo) / (mine.hi - mine.lo) * mine.se
    # Same variance up to the n versus n - 1 denominators of its two terms.
    assert their_se < mine.se <= their_se * math.sqrt(n / (n - 1))
    rebuilt = math.sqrt(
        np.var(y - lam * f) / n + lam**2 * np.var(fu) / n_unl
    )
    assert their_se == pytest.approx(rebuilt, rel=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_estimated_lambda_inside_the_unit_interval(seed: int) -> None:
    y, f, fu = sample(seed)
    n = y.size
    mine = optimal_lambda(y, f, fu)
    assert 0.0 < mine < 1.0
    # ppi_py's covariance has an n denominator, so its lambda is (n - 1) / n times ours.
    theirs = float(np.squeeze(ppi_py.ppi_mean_pointestimate(y, f, fu)))
    assert ppi_mean(y, f, fu, lam=mine * (n - 1) / n).estimate == pytest.approx(theirs, rel=1e-10)
    assert ppi_pp_mean(y, f, fu).estimate == pytest.approx(theirs, abs=1e-3)


def test_negatively_correlated_judge_is_clipped_by_ppi_py_and_not_by_ppieval() -> None:
    y, f, fu = sample(3, slope=-0.7)
    mine = ppi_pp_mean(y, f, fu)
    assert mine.lam is not None and mine.lam < -0.3
    assert mine.se < 0.85 * classical_mean(y).se
    # ppi_py clips lambda at 0 and so returns the classical estimate.
    theirs = float(np.squeeze(ppi_py.ppi_mean_pointestimate(y, f, fu)))
    assert theirs == pytest.approx(classical_mean(y).estimate, rel=1e-10)
    assert ppi_mean(y, f, fu, lam=0.0).estimate == pytest.approx(theirs, rel=1e-10)


def test_lambda_above_one_is_clipped_by_ppi_py() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(0.0, 1.0, 5200)
    f = 0.4 * y + rng.normal(0.0, 0.05, 5200)  # the best lambda is about 2.5 * N / (n + N)
    y, f, fu = y[:200], f[:200], f[200:]
    assert optimal_lambda(y, f, fu) > 1.5
    theirs = float(np.squeeze(ppi_py.ppi_mean_pointestimate(y, f, fu)))
    assert ppi_mean(y, f, fu, lam=1.0).estimate == pytest.approx(theirs, rel=1e-10)
    assert ppi_pp_mean(y, f, fu).se < ppi_mean(y, f, fu, lam=1.0).se
