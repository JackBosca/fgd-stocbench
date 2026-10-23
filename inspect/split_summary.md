# StocBench splits: legacy defaults, fresh data, and unresolved provenance

The inspected repository defaults have overlapping validation/evaluation
selections. This is confirmed code behavior. Whether the paper's reported runs
used those selections is unresolved. The fresh release defines our own explicit
independent protocol; it does not establish that the paper's results were biased.

## Paths and unchanged training data

Paths below use portable root placeholders; select their machine-specific values
in local configuration:

- `LEGACY_ROOT`: the original `stocbench-data` directory, on scratch or retained storage.
- `RELEASE_ROOT`: the `stocbench-releases/cfgd-stocbench-v1` directory.
- `FROZEN_ROOT`: `RELEASE_ROOT/frozen`, the unified input root for the new protocol.

Use the scratch copy for computation and retain the durable release on project
storage. The frozen release contains original training files plus fresh held-out
files. Legacy seed 45 and cases 100–147 are excluded. Retained training files are
links to the original arrays; the staged scratch release contains actual copies.

For each variant, training preserves all trajectories in:

```text
FROZEN_ROOT/incns_stoc/64/traj_seed_42.npy
FROZEN_ROOT/incns_stoc/64/traj_seed_43.npy
FROZEN_ROOT/incns_stoc/64/traj_seed_44.npy
FROZEN_ROOT/incns_det/64/traj_seed_42.npy
FROZEN_ROOT/incns_det/64/traj_seed_43.npy
FROZEN_ROOT/incns_det/64/traj_seed_44.npy
```

Each file has 500 trajectories × 200 frames: 1,500 trajectories and 298,500
adjacent-frame pairs per variant. A filename's seed identifies a batch, not one
trajectory. Every frame/window retains its parent trajectory's partition;
every reference future retains its conditioning case's partition.

## 1. Stochastic one-step

**Legacy defaults:**

- Validation: `LEGACY_ROOT/incns_stoc/64/step_seed_100.npz` through `step_seed_115.npz`.
- Full evaluation: the same directory's `step_seed_100.npz` through `step_seed_147.npz`.
- All 16 validation conditions recur among the 48 evaluation conditions. Each
  condition has 5,000 simulated next states. Increasing model ensemble size does
  not create additional independent conditions.

**Generated and frozen:**

- Validation: `FROZEN_ROOT/incns_stoc/64/step_seed_20000.npz` through `step_seed_20015.npz`.
- Test: the same directory's `step_seed_21000.npz` through `step_seed_21047.npz`.
- These are 16/48 separately generated conditions, each with 5,000 simulator
  futures. Each NPZ contains `init`, `raw`, `mean`, and `std`. The conditions are
  generated separately from the rollout trajectories below.
- Model scoring uses 1,000 generated samples per validation condition and 3,000
  per test condition. These differ from the 5,000 simulator reference members.

**Unresolved:** Did the published runs use cases 100–115 for validation or
selection before reporting scores on cases 100–147? The paper describes 48
conditions and 5,000 reference futures each, but does not identify a separate
validation partition or its file identities.

## 2. Stochastic rollout

**Legacy defaults:**

- Validation and full evaluation both draw starting fields from
  `LEGACY_ROOT/incns_stoc/64/traj_seed_42.npy`, which is also training data.
- Validation uses 1,000 starts; full evaluation uses 3,000. The fixed permutation
  includes every validation start in the larger evaluation selection.
- The paper instead describes 5,000 additional starting states. The difference
  concerns both count and starting-state provenance.

**Generated and frozen:**

- Validation: `FROZEN_ROOT/incns_stoc/64/traj_seed_10000.npy`.
- Test: `FROZEN_ROOT/incns_stoc/64/traj_seed_10001.npy`.
- Each file contains 500 fresh trajectories × 200 frames, with shape
  `(500, 200, 1, 64, 64)`.
- Evaluation starts at frame 0 of all 500 trajectories in the selected partition,
  using the same starts for 10- and 50-step model rollouts.
- `enstrophy_validation.npy` and `enstrophy_test.npy` in the same directory
  contain the corresponding population simulator spectra for future frames 1–50.
  Stochastic scoring compares population spectra, not pathwise agreement with a
  single realized simulator future.

**Unresolved:** Did the published evaluation use the paper's additional 5,000
states, the code's 3,000 training-data starts, or another selection? Training-state
starts can support stability diagnostics, but do not establish generalization
from unseen starting states.

## 3. Deterministic one-step

**Legacy defaults:**

- Validation and full evaluation select exactly the same 48 adjacent-frame pairs
  from `LEGACY_ROOT/incns_det/64/traj_seed_45.npy` using a fixed permutation.
- These pairs are held out from gradient training, but reused between validation
  and full evaluation.
- `LEGACY_ROOT/incns_det/64/step_seed_100.npz` through `step_seed_147.npz` also
  exist, but these deterministic loaders do not use them.
- The paper describes 48 conditioning cases, without identifying their files or
  explaining a separate validation selection.

**Generated and frozen:**

- Validation: frame 0 → 1 from trajectory indices 0–15 of
  `FROZEN_ROOT/incns_det/64/traj_seed_10000.npy`.
- Test: frame 0 → 1 from trajectory indices 0–47 of
  `FROZEN_ROOT/incns_det/64/traj_seed_10001.npy`.
- The 16/48 conditions therefore come from distinct trajectories. Inputs contain
  current vorticity and interval forcing; targets are next-frame vorticity, with
  zero reference conditional variance.
- No new deterministic `step_seed_*.npz` files are needed. The references are
  obtained from those trajectory pairs. Model scoring uses 32 members per
  validation condition and 3,000 per test condition.

**Unresolved:** Did the published evaluation use the separate NPZ cases, seed-45
pairs, or another source? Were those same conditions used for validation or
selection? Checkpoint weights and the inspected metadata do not answer this.

## 4. Deterministic rollout

**Legacy defaults:**

- Validation uses 32 windows and full evaluation uses 3,000 windows from
  `LEGACY_ROOT/incns_det/64/traj_seed_45.npy`'s 500 trajectories.
- At the same rollout horizon, all validation windows recur in full evaluation.
  Changing from 10-step validation to 50-step evaluation still leaves overlapping
  stored frames; it does not establish separate partitions.
- The paper instead describes 5,000 test trajectories of 50 state–forcing pairs.
  The code's 3,000 windows are not 3,000 independent trajectories.

**Generated and frozen:**

- Validation: `FROZEN_ROOT/incns_det/64/traj_seed_10000.npy`.
- Test: `FROZEN_ROOT/incns_det/64/traj_seed_10001.npy`.
- Each file contains 500 fresh trajectories × 200 frames, with shape
  `(500, 200, 2, 64, 64)`. Channel 0 is vorticity; channel 1 is forcing for the
  interval beginning at that frame.
- Evaluation starts at frame 0 of all 500 partition trajectories, with 10/50
  steps. Model rollouts receive the stored forcing for each interval.
- The same directory's `enstrophy_validation.npy` and `enstrophy_test.npy` provide
  partition-specific simulator spectra. Evaluation also compares trajectories
  through RMSE.
- One-step and rollout scoring share some trajectories within a partition;
  validation and test never share trajectories.

**Unresolved:** Did the published runs use separate 5,000-trajectory data or
windows from seed 45? Which selections, if any, overlapped validation?

## Why store 200 frames for 10/50-step rollouts?

The protocol chose the original dataset's 200-frame trajectory format. That
length is not required by the current rollout scores:

| Use | Stored frames consumed |
|---|---|
| One-step scoring at frame 0 | 0 and 1 |
| 10-step rollout from frame 0 | 0–10: 11 frames |
| 50-step rollout from frame 0 | 0–50: 51 frames |
| Population spectral reference | Simulator future frames 1–50 |

The 200 frames provide 199 adjacent pairs per trajectory, hence 99,500 possible
validation pairs per variant, and allow later window/horizon analyses. They cost
additional simulation and storage; a shorter dataset would suffice for the fixed
frame-0 evaluations alone. Frames 51–199 do not enter the current primary rollout
scores. Additional windows from the same trajectory are correlated examples, not
additional independent trajectories. Test frames remain test-only. Changing
starts or horizons would require a separately declared protocol, not a silent
change to this frozen release. The 100 warmup intervals precede stored frame 0;
they are separate from both the 200 stored frames and the 10/50 scored steps.

## Definitive selections and provenance limits

`FROZEN_ROOT/stoc.json` and `FROZEN_ROOT/det.json` are the authoritative executable
definitions. `partitions` records trajectory/case membership; `evaluation` records
`stats_pairs`, `rollout_starts`, horizons, ensemble sizes, and spectrum filenames.
An evaluation start is `(batch seed, trajectory index, current frame)`.
`FROZEN_ROOT/release.json` records frozen-file checksums. The portable generation
specification is [cfgd-stocbench-v1.json](../protocols/cfgd-stocbench-v1.json).

The original [reproduction script](../scripts/reproduce_stocbench.sh) retains the
legacy defaults. Use the [held-out workflow](heldout_protocol.md) for the new
protocol. Existing baseline checkpoints can be evaluated as frozen external
models; the new scores must be recomputed, and training/selection histories must
be disclosed where known. Preserving training membership does not automatically
require retraining.

The checkpoint provenance audit in the [dataset guide](dataset_guide.md)
inspected complete serialized metadata sections of all 14 published checkpoints.
It found no dataset membership or evaluation-input provenance. Model settings,
teacher references, filenames, and epoch numbers cannot establish which examples
produced the published scores. Original configs, data identities, evaluation
artifacts, or author clarification are still needed. Different run overrides,
unpublished code, and documentation errors remain possible explanations.

Sources: [paper v1, Appendix A](https://arxiv.org/html/2608.22309v1#A1), legacy
[stochastic loader](../src/stocbench/data/stoc.py),
[deterministic loader](../src/stocbench/data/det.py),
[data configurations](../src/stocbench/configs/data/),
[validation configurations](../src/stocbench/configs/experiment/),
[full-evaluation settings](../src/stocbench/configs/evaluate/stocbench.yaml),
[frozen-release assembly](../src/stocbench/protocol.py), and
[held-out loader](../src/stocbench/data/heldout.py).
