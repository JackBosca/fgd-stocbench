# StocBench paper metric reference

These are approximate values from the published legacy benchmark, for comparison
with training progress. They are reconstructed plotted coordinates, not raw
model-evaluation logs or results on the independent held-out release. Lower is
better for the error metrics. Predictive std norm is a spread diagnostic with a
nonzero stochastic reference, rather than an error to minimize.

## Sources and precision

Sources inspected on 2026-10-27:

- [Paper v1](https://arxiv.org/html/2608.22309v1), Figures 3-7 and Appendix B.4.
- [Official project-page chart data](https://github.com/tum-pbs/stocbench/blob/2a45f79ba064e89192172f1dcfc20a6113530527/sb-charts.js).
  Its header identifies the arrays as extracted from the arXiv v1 PGF figures.
- [Paper source archive](https://arxiv.org/src/2608.22309v1), including the
  `stoc_errors`, `stoc_std_norm`, `stoc_enstr_rollout`, `det_mean_err`,
  `det_std_norm`, `det_rmse_rollout`, and `det_enstr_rollout` PGF plots.
- [Original repository](https://github.com/tum-pbs/stocbench/tree/2e519f153738972fb41c7c996ada8e1a4f4cb7a9),
  particularly its metrics, reproduction script, and evaluation configuration.

The original repository's main tree and releases supplied no raw result table.
One-step values and the Figure 3(d) scalar come from the chart arrays. Stochastic
predictive std norms come from the corresponding PGF paths calibrated to axis
ticks. The rollout arrays were checked against their PGF paths and completed
where the web charts clip points. All 24 rollout curves have 50 integer horizons;
shared web-chart and reconstructed vector values agreed within `1e-5`. This is a
coordinate-consistency check, not a bound on statistical or scientific error.
No training-run uncertainty or statistical confidence intervals are available.
CSV precision retains extraction detail; use the rounded tables for interpretation.
No training, model evaluation, checkpoint selection, or final-test access occurred
while collecting these references.

## Inference budgets and metric definitions

One-step comparisons use 400 NFE for SI, DDIM, DDPM, and FM; 399 NFE for DPM-2;
1 NFE for CD-1 and ADD-FM; and 2 NFE for CD-2. Long-rollout figures label the
multistep methods as 400 NFE. NFE is network evaluations per generated transition,
not the number of physical rollout steps. EDM is the CD teacher, not another
reported comparison row.

- Mean error: relative spatial L2 error of the ensemble mean, averaged over cases.
- Stochastic std error: relative spatial L2 error of the ensemble std field.
- Deterministic std error: absolute spatial L2 norm of the ensemble std field;
  the reference std is zero. Dividing by 64 gives spatial RMS std on the 64x64 grid.
- Energy distance: spatial mean of pointwise marginal energy distances, using
  the square-root convention in the original metric implementation. It does not
  compare the full joint field law. It is absent from the deterministic metric config.
- Enstrophy-spectrum error: mean absolute difference of ensemble-averaged radial
  spectra over nonzero bins, then averaged over the requested physical horizons.
  It is an absolute error, not a relative error or percentage.
- Rollout RMSE: at each horizon, square root of mean squared error over starts
  and spatial sites. The scalar averages these per-horizon RMSEs arithmetically.
  No paired-trajectory RMSE is reported for the stochastic task.

## Stochastic one-step generation

The auxiliary predicted std L2 norm has reference value approximately **29.34**
(Figure 5); matching that norm alone does not establish calibrated variability.

| Baseline | Mean error | Std error | Energy distance | Predicted std L2 norm |
| --- | ---: | ---: | ---: | ---: |
| SI | 0.0478 | 0.0284 | 0.0410 | 28.90 |
| DPM-2 | 0.0535 | 0.0336 | 0.0457 | 28.64 |
| DDIM | 0.0533 | 0.0367 | 0.0460 | 28.52 |
| DDPM | 0.0519 | 0.0343 | 0.0443 | 28.71 |
| FM | 0.0362 | 0.0232 | 0.0317 | 29.18 |
| CD-1 | 0.0677 | 0.0322 | 0.0546 | 28.85 |
| CD-2 | 0.0623 | 0.0335 | 0.0515 | 28.79 |
| ADD-FM | 0.0396 | 0.0254 | 0.0339 | 29.47 |

## Stochastic rollout

Averages 1-10 and 1-50 use the first 10 or all 50 points of the same Figure 4
long-rollout evaluation. They are not separate 10-step evaluations. At 50 is
only the final horizon.

| Baseline | Spectrum avg. 1-10 | Spectrum avg. 1-50 | Spectrum at 50 | Figure 3(d) sweep scalar |
| --- | ---: | ---: | ---: | ---: |
| SI | 4.73 | 7.08 | 7.47 | 5.19 |
| DPM-2 | 10.02 | 13.18 | 11.98 | 10.26 |
| DDIM | 10.38 | 13.66 | 12.44 | 10.64 |
| DDPM | 1.92 | 3.68 | 2.98 | 2.78 |
| FM | 1.77 | 1.74 | 3.59 | 2.34 |
| CD-1 | 6.67 | 8.16 | 6.97 | 6.48 |
| CD-2 | 2.31 | 2.90 | 3.16 | 2.19 |
| ADD-FM | 1.65 | 1.84 | 3.83 | 2.28 |

Keep the Figure 3(d) sweep scalar separate from these long-rollout averages.
The original `configs/evaluate/stocbench.yaml` uses 10-step rollouts for the NFE
sweep and a separate 50-step pass at the last budget. The project-page description
calls the sweep chart a 50-step result. The original run manifests were not found,
so the published Figure 3(d) horizon is not certified here. This documentation
mismatch is why `summary.csv` names the column `fig3_sweep_enstrophy_error`
without asserting a physical horizon.

## Deterministic one-step generation

| Baseline | Mean error | Absolute std error |
| --- | ---: | ---: |
| SI | 0.0222 | 1.65 |
| DPM-2 | 0.0224 | 1.40 |
| DDIM | 0.0224 | 1.48 |
| DDPM | 0.0224 | 1.22 |
| FM | 0.0227 | 1.60 |
| CD-1 | 0.1141 | 9.71 |
| CD-2 | 0.0509 | 1.92 |
| ADD-FM | 0.0343 | 1.80 |

## Deterministic rollout

Averages come from the Figure 6 RMSE and Figure 7 spectrum-error curves.
The first-ten averages use those same 50-step runs.

| Baseline | RMSE avg. 1-10 | RMSE avg. 1-50 | RMSE at 50 | Spectrum avg. 1-10 | Spectrum avg. 1-50 | Spectrum at 50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SI | 0.145 | 0.830 | 1.242 | 1.47 | 1.90 | 1.92 |
| DPM-2 | 0.137 | 0.817 | 1.242 | 1.54 | 1.94 | 2.05 |
| DDIM | 0.135 | 0.814 | 1.243 | 1.54 | 1.97 | 2.16 |
| DDPM | 0.123 | 0.804 | 1.250 | 2.34 | 2.96 | 2.99 |
| FM | 0.141 | 0.837 | 1.255 | 2.85 | 3.64 | 3.97 |
| CD-1 | 0.637 | 1.183 | 1.350 | 23.71 | 24.61 | 24.69 |
| CD-2 | 0.384 | 1.011 | 1.235 | 1.66 | 2.33 | 2.39 |
| ADD-FM | 0.206 | 0.900 | 1.238 | 1.68 | 2.54 | 2.58 |

## Comparison limits

The paper's deterministic CD-2 uses a different second noise level from the
original reproduction script's `at:[0,2]` schedule. These are paper references,
not guaranteed outputs of that script. The explicitly selected legacy schedule
for this fork's baseline campaign remains unchanged; see
[validation readiness and batching](../heldout_protocol.md#validation-readiness-and-batching).

The released legacy loaders have training/validation/evaluation overlaps; see the
[split summary](../split_summary.md) and [held-out protocol](../heldout_protocol.md).
That audit does not establish the authors' original execution settings or quantify
bias in their published scores. Independent validation/test scores can differ in
both levels and rankings. These references do not set acceptance thresholds.

## CSV files

- [summary.csv](summary.csv): 32 rows, covering all eight baselines in all four
  variant/task combinations. Empty cells mean a metric is inapplicable or not
  reported. `one_step_nfe`, `long_rollout_nfe_label`, and `fig3_sweep_nfe` retain
  the distinct budget conventions. `*_mean10_from_50` averages the first ten
  horizons of the 50-step run; `*_mean50` averages all horizons; `*_at10` and
  `*_at50` are single horizons. The auxiliary stochastic spread norm is included.
- [curves.csv](curves.csv): reconstructed plotted NFE sweeps, rollout curves,
  and spectral-ratio curves. `x` is NFE for `mean`, `std`, `energy`, `enstrophy`,
  `det_mean`, and `det_std`; physical rollout step for `rollout`, `det_rollout`,
  and `det_rmse`; and wavenumber for `ratio` and `det_ratio`. A spectral ratio of
  one is ideal. `clipping_intersection` rows are display-boundary intersections,
  not measured budgets or wavenumbers. `reconstructed_flat_reference` rows use
  the actual 1- or 2-NFE budget. Plot clipping can omit other sweep/ratio points;
  absence is not evidence that a method was not evaluated there.

Validation used Python standard-library CSV/JSON parsing, checked all 32
variant/task rows and their applicable metrics, verified all 24 rollout curves
have horizons 1-50, and recomputed the reported averages from `curves.csv`.
