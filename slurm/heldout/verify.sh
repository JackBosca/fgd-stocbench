#!/bin/bash
#SBATCH --job-name=stocbench-verify
#SBATCH --account=f2pde
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
source "${STOCBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/heldout/common.sh"
run_heldout verify "$@"
