#!/usr/bin/env bash
# Submit SynthSeg-PC training (Billot GMM, no L/R flip).
# Usage:
#   bash synthseg_pc/submit_train.sh
# Optional:
#   FOLD=0 DICE_EPOCHS=50 STEPS_PER_EPOCH=1000 CHAINS=2
#   SMOKE=1   → wl2=1, dice=2, steps=50 (wiring check)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"
PIRIFORM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${SSP_LOGS}"

CORES="${CORES:-8}"
MEM="${MEM:-64G}"
# interactive MaxTime is typically 12h — 24:00:00 → PartitionTimeLimit (never starts)
TIME="${TIME:-12:00:00}"
PARTITION="${PARTITION:-interactive}"
GRES="${GRES:-gpu:l40s.24g:1}"
AFTER="${AFTER:-}"
FOLD="${SSP_FOLD:-0}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  export WL2_EPOCHS=1
  export DICE_EPOCHS=2
  export STEPS_PER_EPOCH=50
  JOBNAME="synthseg-pc-smoke"
  CHAINS=1
  # Isolate weights so smoke never clobbers full fold training.
  export SSP_MODELS="${SCRIPT_DIR}/models_smoke"
else
  JOBNAME="${SYNTHSEG_JOBNAME:-synthseg-pc}"
  # ~50 min/epoch × remaining epochs → use several 12h hops
  CHAINS="${CHAINS:-4}"
fi

export SSP_FOLD="${FOLD}"
# Resume from dice_*: train_job forces WL2=0 even if this is 1.
export WL2_EPOCHS="${WL2_EPOCHS:-1}"
export DICE_EPOCHS="${DICE_EPOCHS:-50}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
export OUTPUT_SHAPE="${OUTPUT_SHAPE:-160}"
export SSP_MODELS="${SSP_MODELS:-${SCRIPT_DIR}/models}"
export SYNTHSEG_ROOT SYNTHSEG_ENV SSP_DATA SSP_CONFIG SSP_MODELS SSP_LOGS PYTHONPATH

if [[ ! -d "${SSP_DATA}/labels_train" ]]; then
  echo "Preparing FoV dense labels (fold ${FOLD})..."
  python3 "${SCRIPT_DIR}/prepare_labels.py" --mode "${SSP_MODE}" --fold "${FOLD}" --out "${SSP_DATA}"
fi

PREV=""
for ((c = 1; c <= CHAINS; c++)); do
  deps=()
  if [[ -n "${AFTER}" ]]; then
    deps+=(--dependency="afterany:${AFTER}")
  fi
  if [[ -n "${PREV}" ]]; then
    deps+=(--dependency="afterany:${PREV}")
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
      "${SCRIPT_DIR}/train_job.sh"
  )
  echo "submitted hop ${c}/${CHAINS} job ${JID} (${JOBNAME} fold${FOLD})"
  PREV="${JID}"
  AFTER=""
done
