# Changelog

## 0.1.0

First release.

- Estimators of a mean or proportion from a few human labels and judge scores on every row: human-only,
  judge-only (for contrast), PPI, PPI++ with the estimated tuning parameter, PPI stratified by judge score, and
  the Rogan-Gladen correction for 0/1 data with its delta-method interval.
- Normal-approximation intervals, percentile bootstrap intervals, cluster-robust variances and inverse-probability
  weights for the human-only, PPI and PPI++ estimators.
- The difference between two systems on shared items (`ppieval compare`).
- Planning: effective sample size, human labels needed for a target width, and uniform or stratified sampling of
  items to label with the weights the estimators take (`ppieval plan`).
- Diagnostics: judge-human correlation and agreement, sensitivity and specificity, the rectifier with its interval,
  and a two-sample Kolmogorov-Smirnov check that labelled rows look like a random subset.
- `ppieval simulate` and `ppieval replay`, with a replay on the MT-Bench human judgments in the README.
- Markdown and JSON output for every command; JSON Lines, CSV and TSV input.
