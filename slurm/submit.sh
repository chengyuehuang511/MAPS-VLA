#!/bin/bash
# Usage: bash slurm/submit.sh slurm/train/train_libero_minivlaoft.sh [extra sbatch args]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a  # auto-export all variables
source "${SCRIPT_DIR}/../secrets.env"
set +a
exec sbatch \
    --partition="${SLURM_PARTITION}" \
    --exclude="${SLURM_EXCLUDE_NODES}" \
    --export=ALL \
    "$@"
