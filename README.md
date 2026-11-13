# ppieval

Estimate a human-judged quantity (accuracy, win rate, mean rating) from a small human-labelled sample plus an
automatic judge's scores on every item, with a confidence interval that stays valid when the judge is biased.

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
where there is none (`item,judge,human` / `item-0001,0,1` / `item-0002,1,`).

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
| ppi-stratified | 0.6855 | 0.0409 | [0.6053, 0.7657] | 0.1604 | 0.87 |  |
| rogan-gladen | 0.7317 | 0.0763 | [0.5822, 0.8812] | 0.2990 | 1.61 |  |

judge-only takes the judge's scores as if they were human labels. It is shown for contrast: its interval is for the judge's mean score, not for the human-judged mean.

## Diagnostics

- judge-human correlation on the labelled rows: 0.5280
- judge-human agreement on the labelled rows: 0.8000
- judge sensitivity 0.9104, specificity 0.5758
- rectifier (mean of judge minus human): 0.0800, 95 % interval [-0.0067, 0.1667]
- judge scores, labelled against unlabelled rows: KS statistic 0.0300, p = 1
```

The example file is simulated, not real data. `python examples/make_small_example.py` writes it: 1,000 items whose
human label is 1 with probability 0.70, a judge that says 1 on 40 % of the human 0s and 0 on 5 % of the human 1s, and
100 human labels kept uniformly at random. The mean of all 1,000 simulated human labels, the hidden ones included,
is 0.678. The judge-only interval contains neither 0.678 nor 0.70; the other five contain both. This is a single
draw, so it illustrates the output and proves nothing about coverage; the replay and the simulation below do that.

```
$ ppieval plan examples/small_example.csv --target-width 0.1
# ppieval plan

- rows: 1000
- with a human label: 100
- without a human label: 900
- judge-human correlation on the labelled rows: 0.5280
- variance of the human label: 0.2233
- 95 % interval width now: human-only 0.1852, ppi++ 0.1603
- effective sample size: 133.6 human labels (the judge is worth 33.6 extra labels)

## Target width 0.1

- human labels needed with the judge: 274
- human labels needed without the judge: 344
```

The commands, each with `--format json` for machine-readable output and `--help` for its options:

| command | what it does |
| --- | --- |
| `ppieval estimate FILE` | every estimator and the diagnostics; `--cluster-col`, `--weight-col`, `--bootstrap B`, `--level` |
| `ppieval compare FILE --systems A B` | the mean difference A minus B on the items both systems were scored on (`--item-col`, `--system-col`) |
| `ppieval plan FILE` | effective sample size; `--target-width W` for the labels needed; `--sample K --method uniform\|stratified --out FILE` to choose rows to label |
| `ppieval simulate` | coverage and width of every estimator across judge quality and labelled fraction |
| `ppieval replay FILE` | on a fully labelled file, hide labels at random and measure coverage and width |

From Python (`python examples/quickstart.py`):

```
human-only      0.6700  [0.5774, 0.7626]
judge-only      0.7770  [0.7512, 0.8028]
ppi             0.7000  [0.6092, 0.7908]
ppi++           0.6869  [0.6068, 0.7670]
ppi-stratified  0.6855  [0.6053, 0.7657]
rogan-gladen    0.7317  [0.5822, 0.8812]
judge minus human on the labelled rows: +0.0800
labelled rows look like a random subset: True
the judge makes 100 labels worth 134
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
| ppi-stratified | `sum_k W_k * theta_k` | `sum_k W_k^2 * V_k + sum_k W_k * (theta_k - theta)^2 / (n + N)` |
| rogan-gladen | `(p + sp - 1) / (se + sp - 1)` | delta method, below |

- **ppi** is the prediction-powered estimate of a mean from Angelopoulos, Bates, Fannjiang, Jordan and Zrnic,
  "Prediction-powered inference", Science 382 (6671), 669-674, 2023
  ([arXiv:2301.09633](https://arxiv.org/abs/2301.09633)): the judge's mean on the unlabelled rows minus the
  rectifier, the judge's average error measured on the labelled rows. A constant bias in the judge cancels, and no
  assumption is made about how good the judge is.
- **ppi++** adds the tuning parameter `lam` of Angelopoulos, Duchi and Zrnic, "PPI++: Efficient Prediction-Powered
  Inference" ([arXiv:2311.01453](https://arxiv.org/abs/2311.01453), Example 6.1). `lam = 0` is the human-only
  estimate and `lam = 1` is ppi. ppieval uses the paper's plug-in for the value that minimises the variance,
  `lam = cov(y, f) / ((1 + n / N) * var(f))`, with the covariance from the labelled rows and the variance from all
  rows. It is not clipped to [0, 1]: for a mean the variance is a convex quadratic in `lam`, and a judge that is
  negatively correlated with the human label gets a negative `lam`. Written with the mean of all `n + N` judge
  scores, the result is `mean(y) + b * (mean of all judge scores - mean(f))` with slope `b = cov(y, f) / var(f)`,
  the form known in survey sampling as the difference, or regression, estimator.
- With population moments, the variance at that `lam` is `var(y) / n * (1 - rho^2 * N / (n + N))`, where `rho` is
  the judge-human correlation. A judge with `rho = 0` leaves the human-only variance; a perfect judge gives
  `var(y) / (n + N)`. The **effective sample size** reported by `plan` and `replay` is `n` times the human-only
  variance over the ppi++ variance: the number of human labels that would do as well without a judge. The
  **labels needed** for a target width solve `var(y) * ((1 - rho^2) / n + rho^2 / M) = (width / (2 z))^2` for `n`,
  with `M` the number of rows.
- **ppi-stratified** bins the rows by judge score (one stratum per distinct score when there are at most 5,
  otherwise 5 quantile bins, merged until each has 10 labelled rows), runs ppi++ inside each stratum, and combines
  the strata by their shares `W_k` of all rows. Stratifying prediction-powered inference is studied by Fisch,
  Maynez, Hofer, Dhingra, Globerson and Cohen, "Stratified Prediction-Powered Inference for Hybrid Language Model
  Evaluation" ([arXiv:2406.04291](https://arxiv.org/abs/2406.04291)); the variance used here, with its second term
  for shares estimated from the rows at hand, is the one written in the table and is this package's own choice.
- **rogan-gladen** is the correction of Rogan and Gladen, "Estimating prevalence from the results of a screening
  test", American Journal of Epidemiology 107 (1), 71-76, 1978, for 0/1 labels and a 0/1 judge: `p` is the judge's
  positive rate on the unlabelled rows, `se` and `sp` its sensitivity and specificity on the labelled rows. The
  delta-method variance is
  `(p(1-p)/N + prev^2 * se(1-se)/n1 + (1-prev)^2 * sp(1-sp)/n0) / (se + sp - 1)^2`, with `n1` and `n0` the labelled
  rows of each class; taking `p` from the unlabelled rows only keeps the three proportions independent.
- **Intervals.** The default is the normal-approximation interval `estimate ± z * sqrt(variance)` with sample
  variances (`n - 1` denominators). `--bootstrap B` adds percentile bootstrap intervals: labelled and unlabelled
  rows are resampled separately and the estimator, `lam` included, is recomputed. With `--cluster-col` the
  variance sums each row's contribution within its cluster first (a cluster-robust variance with a `G / (G - 1)`
  factor) and the bootstrap resamples clusters. With `--weight-col` the two means become weighted means with
  linearised variances; the weights are one over the probability of being labelled for labelled rows and one over
  the probability of not being labelled for the others, which is what `plan --sample` writes.
- **Two systems.** `compare` pairs the rows of two systems by item and runs the same estimators on the differences
  (judge difference on every shared item, human difference where both are labelled).
- **Diagnostics.** Judge-human correlation and agreement, sensitivity and specificity for 0/1 data, the rectifier
  `mean(f - y)` with its interval, and a two-sample Kolmogorov-Smirnov test of the judge scores of labelled against
  unlabelled rows, with a warning when it rejects at 0.01: the labelled rows then do not look like a random subset.

The same setting, a few human labels and automatic labels on everything, is treated for model evaluation by
Boyeau, Angelopoulos, Yosef, Malik and Jordan, "AutoEval Done Right: Using Synthetic Data for Model Evaluation"
([arXiv:2403.07008](https://arxiv.org/abs/2403.07008)). Its estimator of a mean metric such as accuracy is PPI++,
the ppi++ row above, and it also reports an effective sample size. Its second part, ranking models from pairwise
comparisons with the Bradley-Terry model, is not implemented here.

Also not implemented: Zrnic and Candès, "Cross-Prediction-Powered Inference"
([arXiv:2309.16598](https://arxiv.org/abs/2309.16598)), which trains the predictor on the labelled data itself (the
judge here is given); and Chatzi, Straitouri, Thejaswi and Gomez Rodriguez, "Prediction-Powered Ranking of Large
Language Models" ([arXiv:2402.17826](https://arxiv.org/abs/2402.17826)), which builds rank sets for many models.

## Results on real public data

Data: the MT-Bench human judgments, https://huggingface.co/datasets/lmsys/mt_bench_human_judgments (licence
CC-BY-4.0 on the dataset card; Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
[arXiv:2306.05685](https://arxiv.org/abs/2306.05685)). It holds 3,355 pairwise votes by people and 2,400 pairwise
verdicts by GPT-4 on answers of six models to 80 questions. Every human vote has a GPT-4 verdict on the same
question, turn and pair of models. `data/` holds the derived tables (verdicts only, see `data/README.md`), written by

```
uv run --with pyarrow python scripts/fetch_mtbench.py --cache ../mtbench_cache --out data
```

The quantity is a model's win rate over the votes it took part in (win 1, tie 0.5, loss 0); the human label is the
person's vote and the judge score is GPT-4's verdict on the same comparison. The main table uses the model with the
most votes, gpt-3.5-turbo (1,478 votes). In each of 1,000 repetitions a random 5, 10 or 25 % of the human votes is
kept, every estimator is run, and its interval is checked against the win rate over all 1,478 human votes.

```
$ ppieval replay data/mtbench_winrate_gpt-3.5-turbo.csv --fractions 0.05 0.1 0.25 --reps 1000 --seed 0
# ppieval replay

- rows, all with a human label: 1478
- mean of all human labels (the target): 0.6519
- mean judge score: 0.6204
- repetitions per cell: 1000
- interval level: 95 %
- seed: 0

## 5 % of human labels kept (n = 74)

| method | coverage | mean width | vs human-only | mean error | effective n |
| --- | ---: | ---: | ---: | ---: | ---: |
| human-only | 0.951 | 0.1824 | 1.00 | -0.0013 | 74 |
| judge-only | 0.000 | 0.0413 | 0.23 | -0.0315 |  |
| ppi | 0.945 | 0.1909 | 1.05 | +0.0018 | 67 |
| ppi++ | 0.947 | 0.1598 | 0.88 | +0.0011 | 96 |
| ppi-stratified | 0.949 | 0.1629 | 0.89 | -0.0001 | 92 |

## 10 % of human labels kept (n = 148)

| method | coverage | mean width | vs human-only | mean error | effective n |
| --- | ---: | ---: | ---: | ---: | ---: |
| human-only | 0.957 | 0.1289 | 1.00 | -0.0009 | 148 |
| judge-only | 0.000 | 0.0413 | 0.32 | -0.0315 |  |
| ppi | 0.956 | 0.1379 | 1.07 | +0.0001 | 129 |
| ppi++ | 0.953 | 0.1139 | 0.88 | -0.0001 | 190 |
| ppi-stratified | 0.955 | 0.1150 | 0.89 | -0.0004 | 186 |

## 25 % of human labels kept (n = 370)

| method | coverage | mean width | vs human-only | mean error | effective n |
| --- | ---: | ---: | ---: | ---: | ---: |
| human-only | 0.975 | 0.0815 | 1.00 | -0.0003 | 370 |
| judge-only | 0.000 | 0.0413 | 0.51 | -0.0315 |  |
| ppi | 0.972 | 0.0958 | 1.18 | -0.0001 | 267 |
| ppi++ | 0.984 | 0.0738 | 0.91 | -0.0000 | 451 |
| ppi-stratified | 0.983 | 0.0741 | 0.91 | -0.0002 | 447 |
```

Reading it:

- ppi++ intervals are 0.88, 0.88 and 0.91 times as wide as human-only at 5, 10 and 25 % kept, with coverage 0.947,
  0.953 and 0.984. The effective sample size is 96, 190 and 451 human labels for 74, 148 and 370 kept.
- Plain ppi (`lam = 1`) is wider than human-only here, 1.05, 1.07 and 1.18 times, with coverage 0.945, 0.956 and
  0.972: GPT-4's verdicts are not close enough to the human votes for the untuned estimator to pay off.
- The judge-only interval uses no human label, so it is the same in every repetition. It does not contain the
  target: GPT-4's win rate for this model is 0.6204 against 0.6519 from people, and its interval is 0.0413 wide.
  Its coverage is 0.
- Coverage rises above 0.95 at 25 % kept for every valid method because the target is the mean over the file
  itself: the intervals are built for an infinite population and ignore that a quarter of the target is already
  observed. For the human-only interval the expected coverage is `2 * Phi(1.96 / sqrt(1 - 0.25)) - 1 = 0.976`.
- The Rogan-Gladen row is absent because ties make the labels 0, 0.5 and 1.

All six models at 10 % kept, 1,000 repetitions each (`python examples/mtbench_replay.py`):

| model | votes | human win rate | GPT-4 judge | judge-only covers | ppi++ coverage | ppi++ width vs human-only | effective n (of n) |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| alpaca-13b | 1018 | 0.2868 | 0.2333 | no | 0.945 | 0.87 | 134 (102) |
| claude-v1 | 996 | 0.6501 | 0.6827 | no | 0.955 | 0.89 | 126 (100) |
| gpt-3.5-turbo | 1478 | 0.6519 | 0.6204 | no | 0.948 | 0.88 | 189 (148) |
| gpt-4 | 1029 | 0.7328 | 0.8508 | no | 0.951 | 0.96 | 111 (103) |
| llama-13b | 1083 | 0.1542 | 0.1117 | no | 0.946 | 0.93 | 125 (108) |
| vicuna-13b-v1.2 | 1106 | 0.4801 | 0.4738 | yes | 0.951 | 0.87 | 146 (111) |

The judge-only interval contains the human win rate for one model of six. GPT-4 as judge puts GPT-4's own win rate
at 0.8508 where people put it at 0.7328; ppi++ keeps its coverage there but gains little (0.96 of the human-only
width). Votes are treated as independent rows although several votes share a question.

### Simulation

```
$ ppieval simulate --qualities 0.6 0.9 --fractions 0.1 --reps 1000 --seed 0
# ppieval simulate

- rows per repetition: 2000
- probability of a human 1 (the target): 0.5
- judge bias (added to sensitivity, taken from specificity): 0.1
- repetitions per cell: 1000
- interval level: 95 %
- seed: 0

## judge quality 0.6 (sensitivity 0.7, specificity 0.5), 10 % of human labels kept (n = 200)

| method | coverage | mean width | vs human-only | mean error | effective n |
| --- | ---: | ---: | ---: | ---: | ---: |
| human-only | 0.936 | 0.1386 | 1.00 | +0.0014 | 200 |
| judge-only | 0.000 | 0.0429 | 0.31 | +0.1001 |  |
| ppi | 0.941 | 0.1785 | 1.29 | +0.0024 | 120 |
| ppi++ | 0.942 | 0.1356 | 0.98 | +0.0017 | 209 |
| ppi-stratified | 0.942 | 0.1362 | 0.98 | +0.0015 | 207 |
| rogan-gladen | 0.981 | 0.6874 | 4.96 | +0.0028 | 8 |

## judge quality 0.9 (sensitivity 1, specificity 0.8), 10 % of human labels kept (n = 200)

| method | coverage | mean width | vs human-only | mean error | effective n |
| --- | ---: | ---: | ---: | ---: | ---: |
| human-only | 0.945 | 0.1386 | 1.00 | -0.0012 | 200 |
| judge-only | 0.000 | 0.0429 | 0.31 | +0.1002 |  |
| ppi | 0.950 | 0.0945 | 0.68 | +0.0001 | 428 |
| ppi++ | 0.956 | 0.0874 | 0.63 | +0.0005 | 500 |
| ppi-stratified | 0.957 | 0.0877 | 0.63 | -0.0002 | 497 |
| rogan-gladen | 0.948 | 0.1141 | 0.82 | -0.0010 | 288 |
```

With a weak judge (sensitivity + specificity = 1.2) ppi++ falls back to roughly the human-only width while plain ppi
is 1.29 times wider, and the Rogan-Gladen interval is nearly five times wider than human-only. With a good judge
ppi++ is 0.63 times the human-only width. The judge-only interval never covers: by construction its mean is 0.10
above the truth.

## Tests

`uv run --extra dev pytest -q` runs closed forms (`lam = 0` is the human-only estimate, `lam = 1` is ppi, a perfect
judge gives the full-sample variance, a useless judge leaves ppi++ at the human-only variance), comparisons with
scipy (normal interval, Kolmogorov-Smirnov statistic and p-value), hand-worked Rogan-Gladen and stratified cases,
and seeded simulations whose coverage must fall inside stated tolerances (0.95 ± 0.02 for 2,000 draws of the main
simulation). `uv run --extra dev --extra reference pytest -q tests/test_ppi_python.py` compares the point estimate
and interval with the ppi-python package (version 0.2.3 here, on Python 3.12 and 3.10). Two conventions differ and
the tests account for both: ppi-python uses `n` denominators where ppieval uses `n - 1`, and it clips the estimated
`lam` to [0, 1] where ppieval does not, so the two disagree on purpose for a negatively correlated judge and when
the best `lam` exceeds 1.

## Limitations

- The intervals are asymptotic. With few human labels, or a proportion close to 0 or 1, the normal approximation is
  poor; a stratum or a class in which all labels agree contributes zero variance.
- The labelled rows must be a uniform random subset of all rows, or come with correct weights. The
  Kolmogorov-Smirnov check only sees a difference in judge scores; labelling the easy items within each judge score
  goes unnoticed.
- Stratified PPI and Rogan-Gladen take neither weights nor clusters and are left out when either is given.
  Rogan-Gladen needs 0/1 labels and a 0/1 judge and is unstable when sensitivity + specificity is close to 1.
- With weights the variance is a linearisation that assumes rows were chosen independently; for the fixed-size
  stratified samples that `plan --sample` draws it is conservative. With weights or clusters `lam` minimises the
  estimated variance and takes the judge's variance from the labelled rows, not from all rows.
- The bootstrap is the percentile bootstrap only and is not available with weights.
- `replay` and the intervals have no finite-population correction, hence the over-coverage at large kept fractions.
- `compare` handles two systems at a time by paired differences; it does not rank many systems, and the
  Rogan-Gladen correction is not applied to differences.
- `plan` assumes the correlation and variance seen on the labelled rows. Stratified sampling works only on a file
  without human labels, with proportional or equal allocation; there is no Neyman allocation.
- Only means and proportions are estimated: no quantiles or regression coefficients.
- The comparison with ppi-python covers the mean estimator and runs in CI on Python 3.10 only.

## Related tools

- [ppi_py](https://github.com/aangelopoulos/ppi_py) (`pip install ppi-python`, MIT) implements prediction-powered
  inference for more estimands, including medians and linear and logistic regression coefficients.
- [evalstats](https://github.com/vipul21435/evalstats) covers error bars and paired comparisons when every item is
  labelled.
- [judgebias](https://github.com/vipul21435/judgebias) audits an LLM judge.

## Licence

MIT. Copyright (c) 2026 Vipul Raj Jha. The tables in `data/` derive from a dataset under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see `data/README.md` for the source and the changes made.

Built by Vipul Raj Jha · https://vipul21435.github.io
