#!/bin/bash
set -euo pipefail
cd /home/jbosca/fgd-stocbench
mkdir -p slurm/logs

release=/projects/f2pde/stocbench-releases/cfgd-stocbench-v1

for partition in validation test; do
  for variant in stoc det; do
    sbatch slurm/heldout/generate.sh --release "$release" \
      --variant "$variant" --partition "$partition" --kind trajectory
  done

  sbatch slurm/heldout/generate.sh --release "$release" \
    --variant stoc --partition "$partition" --kind case
done