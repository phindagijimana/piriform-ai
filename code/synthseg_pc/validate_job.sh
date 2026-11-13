#!/usr/bin/env bash
# Validate SynthSeg-PC checkpoints on real Hammers T1 (PC Dice only).
# Do not resolve paths from BASH_SOURCE — sbatch copies the script to /var/spool/slurmd/.
set -euo pipefail
PIRIFORM_ROOT="${SLURM_SUBMIT_DIR:-/path/to/piriform}"
SCRIPT_DIR="${PIRIFORM_ROOT}/synthseg_pc"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

FOLD="${SSP_FOLD:-0}"
RECIPE="${RECIPE:-v1}"
case "${RECIPE}" in
  v2)
    MODEL_DIR="${SSP_MODELS}/fold${FOLD}_v2"
    VAL_DIR="${SSP_MODELS}/fold${FOLD}_v2_val"
    ;;
  wb)
    MODEL_DIR="${SSP_MODELS}/fold${FOLD}_wb"
    VAL_DIR="${SSP_MODELS}/fold${FOLD}_wb_val"
    ;;
  *)
    MODEL_DIR="${SSP_MODELS}/fold${FOLD}"
    VAL_DIR="${SSP_MODELS}/fold${FOLD}_val"
    ;;
esac
mkdir -p "${VAL_DIR}" "${SSP_LOGS}" "${SSP_TABLES}"

export PYTHONPATH="${SYNTHSEG_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

TAG="${VAL_TAG:-fold${FOLD}_${RECIPE}}"
EXTRA=()
if [[ "${TOPOLOGY_FIX:-1}" == "1" ]]; then
  EXTRA+=(--topology-fix)
fi
"${SYNTHSEG_ENV}" "${SCRIPT_DIR}/validate.py" \
  --image-dir "${SSP_DATA}/val_images" \
  --gt-dir "${SSP_DATA}/val_labels" \
  --models-dir "${MODEL_DIR}" \
  --validation-dir "${VAL_DIR}" \
  --config "${SSP_CONFIG}" \
  --step-eval "${STEP_EVAL:-1}" \
  --tables-prefix "${SSP_TABLES}/synthseg_pc_${TAG}_epoch_pick" \
  "${EXTRA[@]}"
