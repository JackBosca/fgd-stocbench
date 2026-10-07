# StocBench dataset guide

These notes describe the downloaded 64×64 data and this checkout's default loaders.

**Stochastic variant (`stoc`)**

The model receives current vorticity and generates a possible next-vorticity field. Forcing is unobserved, so repeated predictions for the same input should reproduce the conditional distribution of possible outcomes.

- Training uses `incns_stoc/64/traj_seed_42.npy`, `traj_seed_43.npy`, and `traj_seed_44.npy`.
- Each file contains `(500 trajectories, 200 frames, 1 channel, 64, 64)`. The channel is normalized vorticity. A filename's seed identifies a simulation batch, not one trajectory or a time step.
- Training pairs are consecutive frames: current vorticity → next vorticity. Each pair supplies one realized outcome, not an ensemble.
- Together, the files contain 1,500 trajectories, 300,000 frames, and 298,500 adjacent-frame pairs.
- One-step evaluation uses `step_seed_100.npz` through `step_seed_147.npz`: 48 conditioning cases, each with 5,000 reference next states. The numbering identifies cases, not time steps.

| NPZ array | Shape | Meaning |
|---|---|---|
| `init` | `(1,64,64)` | Current vorticity |
| `raw` | `(5000,1,1,64,64)` | Possible next fields from the same condition |
| `mean` | `(1,64,64)` | Average across those outcomes at each grid point |
| `std` | `(1,64,64)` | Standard deviation across those outcomes at each grid point |

The mean/std are ensemble statistics, not spatial averages or normalization constants. The current loader recomputes them from `raw`.

**Deterministic variant (`det`)**

The model receives current vorticity and the known forcing for the next interval, and predicts next vorticity. The forcing is held constant within that interval. The benchmark treats this transition as deterministic.

- Training uses `incns_det/64/traj_seed_42.npy`, `traj_seed_43.npy`, and `traj_seed_44.npy`.
- Each file contains `(500 trajectories, 200 frames, 2 channels, 64, 64)`. Channel 0 is normalized vorticity; channel 1 is the forcing applied from that frame to the next.
- The input has both channels; the target is next-frame vorticity only. The model does not predict forcing. Training counts are the same as for `stoc`.
- The extra `traj_seed_45.npy` supplies held-out evaluation trajectories: 500 trajectories, 100,000 frames, and 99,500 possible adjacent-frame pairs. The one-step loader selects 48 pairs using a fixed random ordering.
- The download also contains 48 `step_seed_100.npz`–`step_seed_147.npz` files, but the current deterministic loaders do not use them.

Those NPZ files contain `init` with shape `(2,64,64)`, and `mean`/`std` with shape `(1,64,64)`. There is no `raw` ensemble. `mean` is the correct next field; `std` is zero. The active loader constructs these same reference quantities from seed-45 trajectory pairs.

Zero variance means repeated predictions for a fixed input should agree. It does not mean the field is spatially uniform. A constant but incorrect prediction still fails the accuracy test. Mean/std are mathematically valid for a deterministic distribution and allow shared evaluation code.

**Invariant measure**

The **invariant measure** $\mu$ is the long-run probability distribution of the *entire vorticity field* $\omega$. It is stationary: if $\omega_t$ has distribution $\mu$, evolving the physical system by one snapshot interval leaves the distribution unchanged,
$$
\omega_t\sim\mu \quad\Longrightarrow\quad \omega_{t+\Delta t}\sim\mu.
$$
Individual fields still change. “Invariant” describes their distribution, not conservation of enstrophy along a trajectory. The spectrum is one statistic *computed from* that distribution: roughly, the average squared Fourier amplitude of vorticity, grouped by spatial wavenumber. The authors compare that statistic between model and simulator rollouts as a practical test of whether the model respects the invariant measure. Matching the spectrum does **not** establish that the full distributions match; other statistics could differ.

The paper shows separate invariant-measure spectrum evaluations for the stochastic and deterministic variants in Figures 4 and 7. Its one-step metrics instead assess next-state mean, spread, and distributional accuracy. The same learned one-step model is repeatedly applied to produce the rollouts.

**One-step and rollout evaluation**

One-step evaluation predicts the next field from a supplied state. Autoregressive evaluation repeatedly feeds predicted vorticity back into the same model.

- `stoc`: each transition samples a possible future. Rollouts test long-term flow statistics through the enstrophy spectrum; matching one particular reference trajectory is not the objective.
- `det`: each transition also receives the stored forcing for that interval. Rollouts can be compared with the corresponding solver trajectory, as well as its spectrum.

The current full evaluation uses 3,000 model samples per one-step condition and 3,000 rollout starts/windows. It sweeps inference budgets with 10-step rollouts, then evaluates 50-step rollouts at the last budget. Inference budget (NFE) counts network evaluations used to produce a next field; it is different from the number of physical rollout steps.

**Validation and held-out data**

| Evaluation | Validation during training | Full evaluation | Separation |
|---|---|---|---|
| `stoc`, one-step | Cases 100–115 | Cases 100–147 | Cases 116–147 are unseen during training and validation; the other 16 overlap validation |
| `stoc`, rollout | 1,000 starts from seed 42 | 3,000 starts from seed 42 | Starting fields come from training data |
| `det`, one-step | 48 pairs from seed 45 | Same 48 pairs | Held out from training, reused from validation |
| `det`, rollout | 32 windows from seed 45 | 3,000 windows from seed 45 | Held out from training; validation and evaluation selections overlap at the same rollout length |

Validation generates 1,000 predictions per stochastic condition and 32 per deterministic condition. Changing ensemble size does not create new conditioning cases.

Using familiar states as rollout starts can reveal drift and instability, but does not demonstrate generalization to unseen starting states. Validation does not update weights directly, but checkpoint or hyperparameter selection still uses its results. The defaults therefore do not provide a consistently independent three-way split. For an independent thesis test, reserve separate validation/test conditions or whole trajectories; avoid randomly splitting neighboring frame pairs.

**Model comparisons in paper v1**

The [paper](https://arxiv.org/pdf/2608.22309v1) evaluates both modes for both variants:

| Variant | One-step comparison | Rollout comparison |
|---|---|---|
| `stoc` | Mean/std errors and energy distance: Figure 3a–c; variability: Figure 5 left | Enstrophy statistics: Figures 3d and 4, including 50-step rollouts |
| `det` | Mean error: Figure 6 left; predictive variability: Figure 5 right | Trajectory RMSE: Figure 6 right; enstrophy statistics: Figure 7, over 50 steps |

Paper v1's Appendix A describes 5,000 additional stochastic rollout starts and 5,000 deterministic test trajectories with 50 state–forcing pairs. This checkout instead selects 3,000 starts/windows from trajectory seeds 42/45. Keep that difference explicit when describing your experimental protocol; the split details above describe the code, not a claim about the authors' original runs.

Sources: [data configs](src/stocbench/configs/data/), [dataset loaders](src/stocbench/data/), [validation settings](src/stocbench/configs/experiment/), [full evaluation settings](src/stocbench/configs/evaluate/stocbench.yaml), and [inspection notebook](inspect/inspect_data.ipynb).
