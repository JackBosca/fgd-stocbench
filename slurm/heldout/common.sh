#!/bin/bash
# Sourced by the held-out launchers. No job is submitted by this file.
set -euo pipefail
STOCBENCH_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

run_heldout() {
    local stage=$1
    shift
    local -a command
    case "$stage" in
        generate) command=(python -m solver.generate_holdouts "$@") ;;
        verify) command=(python -m stocbench.protocol "$@") ;;
        evaluate) command=(python -m stocbench.heldout evaluate "$@") ;;
        *) echo "Unknown stage: $stage" >&2; return 2 ;;
    esac
    printf 'Stage: %s\nRepo: %s\nLaunch: ' "$stage" "$STOCBENCH_REPO"
    printf '%q ' srun --kill-on-bad-exit=1 --ntasks=1 "${command[@]}"
    printf '\n'
    [[ ${STOCBENCH_DRY_RUN:-0} != 1 ]] || return 0
    [[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Submit through Slurm; use STOCBENCH_DRY_RUN=1 to preview.' >&2; return 2; }
    [[ ${SLURM_NTASKS:-1} == 1 ]] || { echo 'Use one task per job; parallelize with independent shards/jobs.' >&2; return 2; }
    cd "$STOCBENCH_REPO"
    set +u
    module load "${STOCBENCH_MAMBA_MODULE:-miniforge3/26.1.1-3}"
    conda activate "${STOCBENCH_ENV:-fgd}"
    set -u
    export PYTHONPATH="$STOCBENCH_REPO/src:$STOCBENCH_REPO${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1 MPLBACKEND=Agg SLURM_EXPORT_ENV=ALL
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
    export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    if [[ "$stage" == generate ]]; then
        export JAX_PLATFORMS=cuda
    fi
    srun --kill-on-bad-exit=1 --ntasks=1 bash -c '
        set -euo pipefail
        node=$(hostname -s)
        scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -Fxq "$node" || {
            echo "Host $node is not in allocation $SLURM_JOB_ID" >&2; exit 2;
        }
        python -c "import sys, stocbench; print(sys.executable); print(stocbench.__file__)"
        if [[ "$1" == generate ]]; then
            python -c "import jax; print(jax.devices()); assert all(d.platform == '\''gpu'\'' for d in jax.devices())"
        fi
        shift
        exec "$@"
    ' _ "$stage" "${command[@]}"
}
