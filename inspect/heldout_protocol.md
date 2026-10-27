# Independent StocBench validation and test

Implemented on `feat/stocbench-heldout-protocol`. No production generation or
Slurm jobs were run during implementation. Existing raw data and legacy
reproduction commands are unchanged.

## Fixed protocol

[Protocol configuration](../protocols/cfgd-stocbench-v1.json) preserves all original
training trajectories in seeds 42–44. Each variant gets 500 new validation and
500 new test trajectories, each with 200 frames. Stochastic validation/test
add 16/48 independent conditions with 5,000 reference futures each.

Batch IDs are 10000/10001; stochastic case IDs are 20000–20015/21000–21047.
Every frame/window retains its parent trajectory's partition. Deterministic
one-step scoring uses frame 0 of the first 16/48 distinct trajectories. Primary
rollouts start at frame 0 of all 500 partition trajectories, with 10/50 steps.
Test one-step ensembles contain 3,000 model samples per condition. Old seed 45
and cases 100–147 remain legacy development/reproduction data.

The new generator reuses the upstream spatial operators and Euler updates,
with compiled snapshot loops. Every simulation/member has an explicit stream
key derived from namespace, variant, partition, identity, phase and member.
Changing sharding/batching preserves streams; bitwise agreement across different
hardware or numerical-library versions is not promised. High-resolution
conditioning states, generation records, checksums and source hashes are retained.

Freezing verifies all shards, shapes, finite values, checksums, high/coarse state
alignment and exact cross-partition vorticity duplicates, including training
frames and reference futures. It then assembles compatible NPY/NPZ files and
cFGD-compatible membership manifests. Per-partition spectra use the same simulator
future frames as the rollout starts; the existing callback aggregates population
spectra before calculating error. This is a new protocol, not the old paper table.

## Environment

From this repository:

```bash
module load miniforge3/26.1.1-3
conda activate fgd
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

On 2026-10-19, CUDA-enabled JAX/jaxlib 0.4.38 and the matching CUDA 12 plugins
were installed into `fgd`, with `nvidia-cuda-nvcc-cu12==12.6.85`.
All pre-existing package versions were constrained during installation, preserving
NumPy 1.26.4 and the existing PyTorch stack. GPU execution and production
runtime/resource sizing remain unverified; validate with a pilot Slurm job first.
Launchers accept `STOCBENCH_ENV` and `STOCBENCH_MAMBA_MODULE` overrides and reject
CPU fallback for generation. No environment installation is performed by a job.

## Prepare and generate (commands for later execution)

Choose a fresh retained release directory outside the original dataset:

```bash
release=/projects/f2pde/stocbench-releases/cfgd-stocbench-v1
python -m stocbench.protocol plan \
  --legacy-root /projects/f2pde/stocbench-data --release "$release"
mkdir -p slurm/logs
```

Planning reads original headers and creates `plan.json`; it does not simulate.
Commit/freeze generator source before planning. If generator source changes,
create a new release rather than mixing implementations. Planning and freezing
refuse to overwrite existing releases.

Submit generation separately for each variant and partition. Example:

```bash
sbatch slurm/heldout/generate.sh --release "$release" \
  --variant stoc --partition validation --kind trajectory
sbatch slurm/heldout/generate.sh --release "$release" \
  --variant stoc --partition validation --kind case
```

Repeat trajectories for `stoc/test`, `det/validation`, `det/test`; repeat cases
for `stoc/test` only. Deterministic references come from trajectory pairs.
For parallel jobs add `--shard-count N --shard-index I` with each `I=0..N-1`.
`--batch-size` (default 8) and `--member-batch` (32) control memory, not membership.
Completed items are checksum-verified and skipped on retry. A failed in-progress
item is recomputed; unrelated/completed artifacts are never replaced. Do not run
two workers for the same assigned item concurrently.

Preview any launcher without submission, activation or computation:

```bash
STOCBENCH_DRY_RUN=1 bash slurm/heldout/generate.sh --release "$release" \
  --variant stoc --partition validation --kind trajectory
```

Launch from this repository root. The Slurm files request account `f2pde`, one
task, one GPU for generation/evaluation and no GPU for verification. Override
resources before the script name; walltime defaults are requests, not measured
runtime estimates. A submitted job verifies its allocated node and package origin.

## Verify and freeze

After every generation shard succeeds:

```bash
sbatch slurm/heldout/verify.sh freeze --release "$release"
sbatch slurm/heldout/verify.sh verify --release "$release"
```

Run the second command after freeze succeeds. `freeze` publishes `frozen/`
atomically; incomplete/corrupt releases cannot become evaluation inputs. Original
training files are linked, not rewritten. `verify` checks every frozen checksum,
including the original training arrays. If a process is forcibly killed during
freeze, inspect its inactive `.freeze.lock` before removing that individual empty
lock directory and retrying; do not clean the repository or release broadly.

For fast evaluation, copy `frozen/` to scratch using a separate allocated CPU
session/job and `rsync -aL` (dereference the original training links), then run
`verify` there. Keep `frozen/` as the child directory of the chosen scratch release
root. Membership files use relative dataset paths and work at either location.
Preserve the retained release; scratch is not the only copy.

## Lock checkpoints and evaluate

Copy [the checkpoint specification example](../protocols/baselines.example.json)
to external run storage. Fill actual local checkpoint paths and provenance,
including the EDM/FM teacher files needed to construct CD/ADD. Choose `variant`
and a baseline subset if desired. Download published weights separately and
record the resolved revision/epoch. The new wrapper never resolves `@max` or
silently downloads changing weights. `model_overrides` records any architecture
settings needed by local checkpoints; default settings retain published legacy
padding. `schedules` can override each baseline's schedule list before locking.

```bash
python -m stocbench.heldout lock --spec /absolute/path/checkpoints.json \
  --output /absolute/path/checkpoints.lock.json
sbatch slurm/heldout/evaluate.sh \
  --manifest /absolute/path/scratch-release/frozen/stoc.json \
  --lock /absolute/path/checkpoints.lock.json --baseline fm --partition validation \
  --run-dir /projects/f2pde/stocbench-outputs/eval/stoc/fm/validation-v1
```

Repeat for `si`, `fm`, `dpm2`, `ddim`, `ddpm`, `cd1`, `cd2`, `addfm` as selected
in the lock. Sampler defaults mirror the legacy reproduction script, including
its CD schedules; tune only on validation. Use separate locks for the variants.
Final test requires an explicit `--partition test` and a fresh run directory.
Keep the locked settings unchanged after inspecting test results.

### Validation readiness and batching

The eight comparisons per variant are `si`, `fm`, `dpm2`, `ddim`, `ddpm`, `cd1`,
`cd2`, and `addfm`: 16 jobs for both variants on validation alone. They need six
checkpoint files per variant (`si`, `fm`, `dm`, `edm`, `cd`, `add_fm`), including
the EDM/FM teachers. DDIM/DDPM/DPM-2 share DM weights; CD-1/CD-2 share CD weights.
Keep the legacy CD-2 schedule `at:[0,2]` for both variants in this comparison,
as explicitly selected. This differs from the paper's deterministic CD-2 noise
level; the choice belongs in the checkpoint lock's selection provenance.

Inference is already batched in `EvalCallback._sample`: `--batch-size` controls
GPU chunks of ensemble members and the held-out rollout loader's batch size.
The default is 32. Changing it can change random-number assignment and floating
point rounding, so it does not promise identical samples. Retain 32 for these
runs; no sampler, precision, random stream, or metric calculation is changed.
The 400-step stochastic validation pass retains intermediate samples for
mean/std curves, so GPU memory also depends on the ensemble and schedule length,
not only the chunk size. GPU throughput and peak memory still require a measured
allocation; the launcher's 24-hour request is not a runtime estimate.

Output plots use the saved model name when the run directory is a job ID or
other unregistered name. This preserves DPM-2's `2 * steps - 1` NFE axis and
DDIM/DDPM labels. The cause of the previous wrong axis was relying only on the
directory basename. The legacy seed-field panel is skipped when none of its
requested case IDs occur; the per-schedule mean/std grids still show held-out
cases. Metrics and their NPZ artifacts are unaffected.

CPU readiness checks on 2026-10-24 verified all 12 published checkpoint payloads
against their LFS sizes and SHA-256 hashes at repository revision
`10b872dbe4b33aa8dd30cdc2e65515c2ce652e8b`. All 16 baseline/variant configurations
composed, loaded their weights strictly, and produced finite two-member 64×64
samples in a bounded CPU check (two solver steps for the iterative models).
Validation selections contained 16 one-step conditions and 500 rollout starts;
both 10- and 50-step windows were checked. These checks are not GPU benchmarks
or full evaluations. Check reports belong in external development output storage.

With the downloaded files and per-variant locks under `CHECKPOINT_ROOT`, launch
validation only from the repository root. Set `FROZEN_ROOT`, `CHECKPOINT_ROOT`,
and `OUTPUT_ROOT` to the local scratch release, retained checkpoints, and intended
research output root. The lock paths below distinguish this selection from any
later test campaign:

```bash
mkdir -p slurm/logs
campaign="validation-$(date -u +%Y%m%dT%H%M%SZ)"
for variant in stoc det; do
  for baseline in si fm dpm2 ddim ddpm cd1 cd2 addfm; do
    sbatch --job-name="val-${variant}-${baseline}" slurm/heldout/evaluate.sh \
      --manifest "$FROZEN_ROOT/$variant.json" \
      --lock "$CHECKPOINT_ROOT/locks/cfgd-stocbench-v1-validation/$variant.lock.json" \
      --baseline "$baseline" --partition validation --batch-size 32 \
      --run-dir "$OUTPUT_ROOT/eval/$variant/$baseline/$campaign" || break 2
  done
done
```

This submits 16 independent one-GPU jobs. The frozen validation partition is
explicit even though the underlying Lightning API calls its inference loop
`test`; no final-test cases are selected. Each run directory must be new.

The wrapper checks checkpoint hashes and selected data hashes before inference,
saves the lock, manifest hash, command and source hashes, and launches the
existing metric/plot pipeline with explicit selections. Teacher paths are pinned
as well. One process/GPU per evaluation job avoids distributed duplication.
Validation and testing use different dataset/spectrum selections. Training with
`data=heldout_stoc`/`heldout_det` rejects `data.partition=test`.

Existing baseline weights can be reevaluated on the new independent test data.
This avoids the overlap in the legacy default selections; it does not establish
that the paper's reported runs used those selections. Published training and
evaluation membership remains unverified; see the
[checkpoint provenance audit](dataset_guide.md). Retraining is needed when
changing training membership or requiring matched training/selection histories.
Historical scores cannot be copied into the new comparison table.

## Monitoring and checks

```bash
bash slurm/heldout/watch.sh
bash slurm/heldout/watch.sh JOB_ID --follow
```

The read-only picker shows active/accounted jobs, states, elapsed times and logs;
`/text` filters and `r` refreshes. Logs report warmup, stored-frame and reference
member progress. This supplements existing local `slurm/watch.sh`; only the new
`slurm/heldout/` workflow is versioned.

Small checks (CPU only, with JAX available):

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

These check stream separation, batching/resume, upstream update agreement,
deterministic forcing alignment, freezing/corruption, manifest selections,
population metric aggregation and launcher previews. Full GPU generation and
real-checkpoint evaluation require later jobs. cFGD can read the membership
manifests; its separate evaluator still needs to consume the explicit evaluation
starts and per-partition spectrum paths before claiming this full protocol.
