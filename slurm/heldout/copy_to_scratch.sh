#!/bin/bash
#SBATCH --job-name=stocbench-copy-verify
#SBATCH --account=f2pde
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
source "${STOCBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/heldout/common.sh"
source_release=/projects/f2pde/stocbench-releases/cfgd-stocbench-v1
scratch_release=/scratch/jbosca/stocbench-releases/cfgd-stocbench-v1
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Submit through Slurm.' >&2; exit 2; }
[[ -f "$source_release/frozen/release.json" ]]
mkdir -p "$(dirname "$scratch_release")"
# Refuse to overwrite an existing scratch release.
mkdir "$scratch_release"
mkdir "$scratch_release/frozen"
srun --kill-on-bad-exit=1 --ntasks=1 rsync -aL --stats \
    "$source_release/frozen/" "$scratch_release/frozen/"
run_heldout verify verify --release "$scratch_release"
echo "Copy and verification succeeded: $scratch_release/frozen"
