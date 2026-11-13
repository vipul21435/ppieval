# ppieval

Estimate a human-judged quantity (accuracy, win rate, mean rating) from a small human-labelled sample plus an
automatic judge's scores on every item, with a confidence interval that stays valid when the judge is biased.

**Status: early version.** Four estimators of a mean or proportion and the `ppieval estimate` command work and are
tested. Everything under [Not implemented yet](#not-implemented-yet) is still to be written, and there are no results
on real data yet.

## Why this is hard

An automatic judge is cheap enough to score every item, but it is biased: it passes answers a person would fail, or
the reverse. Treating its verdicts as labels gives an interval that is narrow and centred on the wrong value, and
judging more items only makes it narrower. Using the human labels alone is valid but ignores the judge, and a few
hundred labels leave a wide interval. The estimate worth having uses the judge on every item and uses the human
labels to measure and remove the judge's bias, and its interval has to carry the noise of both steps.

## Install

```
pip install git+https://github.com/vipul21435/ppieval
```

Python 3.10 or later. The only runtime dependency is numpy.

## Quickstart

The input is a JSON Lines, CSV or TSV file with a judge score on every row and a human label on some rows, empty
where there is none:

```
$ head -4 examples/small_example.csv
item,judge,human
item-0001,0,1
item-0002,1,
item-0003,1,
```

```
$ ppieval estimate examples/small_example.csv
# ppieval estimate

- rows: 1000
- with a human label: 100
- without a human label: 900
- interval level: 95 %

| method | estimate | std. error | 95 % interval | width | vs human-only | lambda |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| human-only | 0.6700 | 0.0473 | [0.5774, 0.7626] | 0.1852 | 1.00 |  |
| judge-only | 0.7770 | 0.0132 | [0.7512, 0.8028] | 0.0516 | 0.28 |  |
| ppi | 0.7000 | 0.0463 | [0.6092, 0.7908] | 0.1816 | 0.98 | 1.0000 |
| ppi++ | 0.6869 | 0.0409 | [0.6068, 0.7670] | 0.1603 | 0.87 | 0.5635 |

judge-only takes the judge's scores as if they were human labels. It is shown for contrast: its interval is for the judge's mean score, not for the human-judged mean.
```

The example file is simulated, not real data. `python examples/make_small_example.py` writes it: 1,000 items whose
human label is 1 with probability 0.70, a judge that says 1 on 40 % of the human 0s and 0 on 5 % of the human 1s, and
100 human labels kept uniformly at random. The mean of all 1,000 simulated human labels, the hidden ones included,
is 0.678. The judge-only interval contains neither 0.678 nor 0.70; the other three contain both, and the ppi++
interval is 0.87 times as wide as the human-only one. This is a single draw, so it illustrates the output and proves
nothing about coverage.

`--format json` prints the same numbers as JSON. `--judge-col`, `--human-col`, `--level`, `--input-format` and
`--digits` are described by `ppieval estimate --help`.

From Python:

```python
from ppieval import load_dataset, ppi_pp_mean

data = load_dataset("examples/small_example.csv")
est = ppi_pp_mean(data.y, data.f, data.f_unlabelled)
print(f"{est.estimate:.4f} [{est.lo:.4f}, {est.hi:.4f}] lam={est.lam:.4f}")
# 0.6869 [0.6068, 0.7670] lam=0.5635
```

## How it works

Write `y` for the human labels on the `n` labelled rows, `f` for the judge's scores on the same rows, and `g` for
the judge's scores on the `N` rows without a human label.

| method | estimate | variance |
| --- | --- | --- |
| human-only | `mean(y)` | `var(y) / n` |
| judge-only | mean of all `n + N` judge scores | their variance over `n + N` |
| ppi | `mean(g) - mean(f - y)` | `var(g) / N + var(f - y) / n` |
| ppi++ | `mean(y) + lam * (mean(g) - mean(f))` | `var(y - lam * f) / n + lam^2 * var(g) / N` |

- **ppi** is the prediction-powered estimate of a mean from Angelopoulos, Bates, Fannjiang, Jordan and Zrnic,
  "Prediction-Powered Inference" ([arXiv:2301.09633](https://arxiv.org/abs/2301.09633)): the judge's mean on the
  unlabelled rows minus the rectifier, which is the judge's average error measured on the labelled rows. A constant
  bias in the judge cancels, and no assumption is made about how good the judge is.
- **ppi++** adds the tuning parameter `lam` of Angelopoulos, Duchi and Zrnic, "PPI++: Efficient Prediction-Powered
  Inference" ([arXiv:2311.01453](https://arxiv.org/abs/2311.01453), Example 6.1). `lam = 0` is the human-only
  estimate and `lam = 1` is ppi. ppieval uses the paper's plug-in for the value that minimises the variance,
  `lam = cov(y, f) / ((1 + n / N) * var(f))`, with the covariance taken from the labelled rows and the variance from
  all rows. It is not clipped to [0, 1]: for a mean the variance is a convex quadratic in `lam`, and a judge that is
  negatively correlated with the human label gets a negative `lam`.
- With population moments, the variance at that `lam` is `var(y) / n * (1 - rho^2 * N / (n + N))`, where `rho` is
  the correlation between judge and human label. A judge with `rho = 0` leaves the human-only variance; a perfect
  judge gives `var(y) / (n + N)`, the variance of labelling every row.
- All intervals are normal-approximation intervals, `estimate ± z * sqrt(variance)`, with sample variances
  (`n - 1` denominators).

The tests check these closed forms, compare the human-only interval and the normal quantile with scipy, and run a
seeded simulation (2,000 draws, `n = 100`, `N = 1,000`, a binary judge that passes too often) in which the coverage
of the human-only, ppi and ppi++ intervals must fall within 0.95 ± 0.02 and that of judge-only below 0.05. Run them
with `uv run --extra dev pytest -q`.

## Not implemented yet

- PPI stratified by judge score, and the Rogan-Gladen correction for binary labels.
- The difference between two systems on shared items (`ppieval compare`).
- Bootstrap intervals, and clustered intervals for rows that are not independent.
- Planning (`ppieval plan`): the effective sample size, the number of human labels needed for a target width, and
  sampling of items to label with the weights the estimator then needs.
- Diagnostics: judge-human agreement, the rectifier with its interval, and a check that the labelled rows look like
  a random subset of all rows.
- `ppieval simulate` and `ppieval replay`, and results on a real public dataset.
- A comparison test against the `ppi-python` package.

## Limitations

- The intervals are asymptotic. With few human labels, or a proportion close to 0 or 1, the normal approximation
  is poor for every row of the table.
- The labelled rows must be a uniform random subset of all rows. If people labelled the easy items, the first items,
  or items chosen by looking at the judge's score, the rectifier is biased and so is the estimate. Nothing checks
  this yet.
- Rows are treated as independent.
- Only means and proportions are estimated.

## Related tools

- [ppi_py](https://github.com/aangelopoulos/ppi_py) (`pip install ppi-python`, MIT) implements prediction-powered
  inference for more estimands, including medians and linear and logistic regression coefficients.
- [evalstats](https://github.com/vipul21435/evalstats) covers error bars and paired comparisons when every item is
  labelled.
- [judgebias](https://github.com/vipul21435/judgebias) audits an LLM judge.

## Licence

MIT. Copyright (c) 2026 Vipul Raj Jha.

Built by Vipul Raj Jha · https://vipul21435.github.io
