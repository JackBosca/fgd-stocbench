# StocBench fresh-validation baseline metrics

These are measured baseline-checkpoint results on the independent
`cfgd-stocbench-v1` validation partition, campaign
`validation-20261106T231802Z`. They are not the [paper-derived references](../paper_metrics/README.md)
and are not final-test scores. Lower is better for every error metric below.

## Sources and precision

Exported on 2026-11-10 directly from the saved metric NPZ artifacts, retaining
floating-point scalar values in the CSVs. Tables are rounded for readability.
All 16 jobs completed with Slurm exit code `0:0`. The export checked the intended
ensemble sizes, 16 per-condition scores, complete schedule artifact inventories,
finite values, and consistency of rollout scalars with their per-horizon curves.
No new inference, training, checkpoint selection, or test evaluation was run.
These are single-seed evaluations; no confidence intervals are provided.

Portable source paths are relative to `BENCHMARK_OUTPUT_ROOT/eval/`:
`<stoc|det>/<baseline>/validation-20261106T231802Z/artifacts/`.
The checkpoint root is separate from research outputs. Its `download-manifest.json`
pins `pfistse/stocbench-ckpts` revision
`10b872dbe4b33aa8dd30cdc2e65515c2ce652e8b` and the 12 downloaded payload hashes.
Each source run retains `protocol_evidence.json`, `.hydra/`, `config.yaml`,
`summary.md`, artifacts and plots. CSV rows identify exact run/artifact paths;
`summary.csv` also records Slurm job IDs (19735568–19735583).

Frozen manifest and checkpoint/settings-lock identities:

| Variant | Manifest SHA-256 | Lock SHA-256 |
|---|---|---|
| stoc | `d09b53cc3641e1f0f018aa21c16bcdb86acc8e50762efd9e6935a6f1f115221e` | `b1fac79bfa640067a721a7e6e626c2fe335b4e53f3cce8082f8dad2257b6ad87` |
| det | `892ab11d8b68faf3d664b770b1f3ca695a36d55dabbd69d3984f139707e0fca2` | `2f7e2ba1f9d7fdffe40e8e5830f21d823087cfae27db63fcbc04b6c0f0ffc86f` |

## Protocol and inference budgets

- Both variants: 16 one-step validation conditions and 500 rollout starts at frame 0.
- Stochastic: cases 20000–20015, 1,000 model samples and 5,000 simulator futures per condition.
- Deterministic: frame 0 → 1 of trajectories 0–15 in batch 10000, 32 model samples and one reference target per condition.
- Rollout trajectories come from batch 10000 in each variant. One model path is generated per start, with supplied interval forcing for deterministic flow.
- All jobs use seed 42 and inference batch size 32. The inference loop uses Lightning `test`, but the data partition is explicitly validation.
- The sweep computes one-step and 10-step rollout scores at every schedule. A separate 50-step rollout pass uses the last schedule.
- These tables use the last configured budget, not the best value selected independently for each metric.

| Baseline | Final schedule | Network evaluations per transition |
|---|---|---:|
| SI | `u:400` | 400 |
| DPM-2 | `u:400` | 799 |
| DDIM | `u:400` | 400 |
| DDPM | `u:400` | 400 |
| FM | `u:400` | 400 |
| CD-1 | `at:[0]` | 1 |
| CD-2 | `at:[0,2]` | 2 |
| ADD-FM | `u:1` | 1 |

DPM-2 uses `2 * solver_steps - 1` network evaluations: the 400-step endpoint
here costs 799 NFE, whereas the paper-reference one-step table uses 399 NFE.
Use `curves.csv` for comparisons at other budgets (for example DPM-2 at 99 NFE,
and FM/SI/DDIM/DDPM at 100 NFE). CD-2 retains the explicitly selected legacy
`at:[0,2]` schedule for both variants; the paper uses a different deterministic
second noise level. EDM is a teacher, not an additional comparison row.

Mean error is relative spatial L2. Std error is relative for stochastic flow
and absolute spatial L2 for deterministic flow (whose target spread is zero).
Energy distance uses the code's square-root pointwise-marginal convention, then
averages over grid sites. Spectrum error pools predicted spectra across starts
before absolute error against the partition reference, excludes the zero bin,
and averages over horizons. RMSE pools squared errors over starts/sites before
square root at each horizon, then averages the resulting per-horizon RMSEs.
No paired-trajectory RMSE or energy distance is reported for stochastic rollout
or deterministic one-step evaluation, respectively.

## Stochastic one-step generation

| Baseline | Mean error | Relative std error | Energy distance |
|---|---:|---:|---:|
| SI | 0.05002 | 0.04265 | 0.04663 |
| DPM-2 | 0.05517 | 0.04566 | 0.05106 |
| DDIM | 0.05668 | 0.04905 | 0.05223 |
| DDPM | 0.05325 | 0.04427 | 0.04878 |
| FM | 0.03715 | 0.03141 | 0.03585 |
| CD-1 | 0.06860 | 0.04598 | 0.05932 |
| CD-2 | 0.06276 | 0.04819 | 0.05515 |
| ADD-FM | 0.04138 | 0.03130 | 0.03807 |

16 conditions × 1,000 model samples; relative errors are fractions, so
`0.03715` is approximately 3.715%.

## Stochastic rollout

| Baseline | Spectrum avg. 1–10, separate pass | Spectrum avg. 1–50 | Spectrum at 50 |
|---|---:|---:|---:|
| SI | 6.4859 | 10.1139 | 9.8441 |
| DPM-2 | 10.0023 | 14.6307 | 16.5604 |
| DDIM | 11.1611 | 15.1443 | 17.7560 |
| DDPM | 3.1748 | 5.7586 | 7.2049 |
| FM | 2.1130 | 2.4181 | 3.6454 |
| CD-1 | 7.0220 | 9.6209 | 11.1741 |
| CD-2 | 2.2757 | 3.5583 | 5.3604 |
| ADD-FM | 1.6746 | 2.4188 | 2.1879 |

500 starts. The first column comes from the 10-step pass, not the first ten
points of the 50-step pass. Both are retained separately in the CSVs.

## Deterministic one-step generation

| Baseline | Mean error | Absolute std error |
|---|---:|---:|
| SI | 0.02135 | 1.50528 |
| DPM-2 | 0.02132 | 1.37006 |
| DDIM | 0.02145 | 1.32222 |
| DDPM | 0.02117 | 1.09980 |
| FM | 0.02153 | 1.45221 |
| CD-1 | 0.11301 | 8.74438 |
| CD-2 | 0.10844 | 8.15139 |
| ADD-FM | 0.03263 | 1.66055 |

16 conditions × 32 model samples; absolute std error is not a percentage.

## Deterministic rollout

| Baseline | RMSE avg. 1–10, separate pass | RMSE avg. 1–50 | Spectrum avg. 1–10, separate pass | Spectrum avg. 1–50 |
|---|---:|---:|---:|---:|
| SI | 0.12005 | 0.80263 | 1.1781 | 1.4670 |
| DPM-2 | 0.11305 | 0.79388 | 1.2266 | 1.4817 |
| DDIM | 0.11106 | 0.79015 | 1.2354 | 1.5325 |
| DDPM | 0.09699 | 0.77788 | 1.9756 | 2.4098 |
| FM | 0.11701 | 0.81149 | 2.4477 | 3.0133 |
| CD-1 | 0.61009 | 1.15829 | 23.1277 | 24.0929 |
| CD-2 | 0.59955 | 1.14846 | 20.9707 | 21.7236 |
| ADD-FM | 0.18262 | 0.87376 | 1.4901 | 2.3494 |

500 starts. Horizon averages are not final-frame scores. Endpoints and the
first-ten average of the separate 50-step run are available in `summary.csv`.

## CSV files

- [summary.csv](summary.csv): 32 rows, one per baseline and variant/task combination. One-step fields are empty on rollout rows and vice versa. Empty means inapplicable, not zero.
- `nfe` is the actual network-call count per generated transition; `sampling_schedule` retains the solver/noise-index setting.
- `*_mean10` comes from the separate 10-step pass. `*_mean10_from_50` averages the first ten points of the 50-step pass; `*_mean50` averages all 50. `*_at10`/`*_at50` are single horizons from that long pass.
- [curves.csv](curves.csv): complete measured budget sweeps and per-physical-horizon rollout curves. `pass=budget_sweep` has `x=NFE`; `pass=rollout_10` or `rollout_50` has `x=physical horizon`. `nfe` and `sampling_schedule` identify the inference budget of every curve.
- `source_artifact` paths in `curves.csv` and `run_relative_path` in `summary.csv` are relative to the same evaluation output root. Source artifacts retain per-condition results, fields and spectra; these small CSVs do not duplicate those larger arrays.

## Comparison limits

Use identical conditions, generated-member counts, rollout starts, spectral
references, normalization and aggregation for a controlled model comparison.
Recent development evaluations of cFGD used smaller/different budgets; their
published development scores are not automatically matched to this campaign.
Compare inference cost as well as accuracy, and retain the distinction between
validation selection and untouched final testing. Equal checkpoint parameter
counts do not establish matched training budgets or selection histories.

The [split summary](../split_summary.md) explains the legacy overlap and remaining
uncertainty about the paper's original runs. The [held-out workflow](../heldout_protocol.md)
owns generation and evaluation commands. These results do not establish what
data were used for the paper figures; paper reference scores are historical context.
