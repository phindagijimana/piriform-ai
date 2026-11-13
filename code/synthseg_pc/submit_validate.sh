#!/usr/bin/env bash
# Submit real-T1 validation after SynthSeg-PC training.
# Usage: AFTER=<train_jobid> bash synthseg_pc/submit_validate.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"
PIRIFORM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${SSP_LOGS}"

CORES="${CORES:-4}"
MEM="${MEM:-32G}"
TIME="${TIME:-04:00:00}"
PARTITION="${PARTITION:-interactive}"
GRES="${GRES:-gpu:l40s.24g:1}"
AFTER="${AFTER:-}"
JOBNAME="${SYNTHSEG_VAL_JOBNAME:-synthseg-pc-val}"
export SSP_FOLD="${SSP_FOLD:-0}"
export STEP_EVAL="${STEP_EVAL:-1}"
# Optional: SSP_MODELS=.../models_smoke for post-smoke wiring check
export SSP_MODELS="${SSP_MODELS:-${SCRIPT_DIR}/models}"
export VAL_TAG="${VAL_TAG:-fold${SSP_FOLD}}"
export SYNTHSEG_ROOT SYNTHSEG_ENV SSP_DATA SSP_CONFIG SSP_MODELS SSP_TABLES SSP_LOGS VAL_TAG PYTHONPATH
echo "validate SSP_MODELS=${SSP_MODELS} fold=${SSP_FOLD} tag=${VAL_TAG}"

deps=()
if [[ -n "${AFTER}" ]]; then
  deps+=(--dependency="afterany:${AFTER}")
fi

JID=$(
  sbatch \
    --parsable \
    --job-name="${JOBNAME}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CORES}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --partition="${PARTITION}" \
    --gres="${GRES}" \
    --chdir="${PIRIFORM_ROOT}" \
    --output="${SSP_LOGS}/%x-%j.out" \
    --error="${SSP_LOGS}/%x-%j.err" \
    --export=ALL \
    "${deps[@]}" \
    "${SCRIPT_DIR}/validate_job.sh"
)
echo "submitted validate job ${JID}"
