#!/usr/bin/env bash
# 5-fold nnU-Net (1000 epochs, no L/R mirror). Interactive 12h; resubmit to continue.
# Usage: submit_nnunet_train.sh
# Optional env:
#   FOLDS=0-4  PARTITION=interactive  TIME=12:00:00
#   AFTER=188516_0:188516_1     wait for those jobs (TIMEOUT counts; afterany)
#   CHAINS=2                    number of 12h hops to queue (default 1)
#
# Overnight: do not start a second job on a fold that is already R/PD.
# Queue hops that wait, e.g.
#   FOLDS=0-2 CHAINS=2 AFTER=188516_0:188516_1:188516_2 submit_nnunet_train.sh
# Each hop: skip if checkpoint_final.pth exists, else --c from checkpoint_latest.pth.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainerNoMirroring}"
# shellcheck source=nnunet_exec.sh
source "${SCRIPT_DIR}/nnunet_exec.sh"
nnunet_env_setup

CORES="${CORES:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-12:00:00}"
PARTITION="${PARTITION:-interactive}"
GRES="${GRES:-gpu:l40s.24g:1}"
FOLDS="${FOLDS:-0-4}"
AFTER="${AFTER:-}"
CHAINS="${CHAINS:-1}"
JOBNAME="${JOBNAME:-nnunet-pc}"

if [[ ! -f "${nnUNet_raw}/${NNUNET_DATASET}/dataset.json" ]]; then
  echo "ERROR: missing ${nnUNet_raw}/${NNUNET_DATASET}/dataset.json (run prepare_nnunet.py)" >&2
  exit 1
fi

if ! [[ "${CHAINS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CHAINS must be a positive integer (got ${CHAINS})" >&2
  exit 1
fi

expand_folds() {
  local spec="${1}"
  local p a b i
  local -a parts
  IFS=',' read -ra parts <<< "${spec}"
  for p in "${parts[@]}"; do
    p="${p%%%*}"
    if [[ "${p}" == *-* ]]; then
      a="${p%-*}"
      b="${p#*-}"
      for ((i = a; i <= b; i++)); do
        printf '%s\n' "${i}"
      done
    else
      printf '%s\n' "${p}"
    fi
  done
}

# A hop with no AFTER would write the same fold_* as a live job.
if [[ -z "${AFTER}" ]]; then
  mapfile -t LIVE < <(squeue --me -n "${JOBNAME}" -t R,PD -h -o '%i' 2>/dev/null || true)
  while read -r f; do
    [[ -z "${f}" ]] && continue
    for job in "${LIVE[@]+"${LIVE[@]}"}"; do
      case "${job}" in
        *_"${f}" | *_"${f}"[%]*)
          echo "ERROR: fold ${f} already queued as ${job}; set AFTER=${job} (or wait). Do not overlap writers." >&2
          exit 1
          ;;
      esac
    done
  done < <(expand_folds "${FOLDS}")
fi

python3 "${SCRIPT_DIR}/make_nnunet_splits.py"

submit_one() {
  local dep="${1}"
  local -a extra=()
  if [[ -n "${dep}" ]]; then
    extra+=(--dependency="afterany:${dep}" --kill-on-invalid-dep=yes)
  fi
  sbatch --parsable \
    --export=ALL,CONFIG="${CONFIG}",TRAINER="${TRAINER}",BEYONDFS_MODULES="${SCRIPT_DIR}" \
    --job-name="${JOBNAME}" \
    --array="${FOLDS}" \
    --partition="${PARTITION}" \
    --gres="${GRES}" \
    --time="${TIME}" \
    --cpus-per-task="${CORES}" \
    --mem="${MEM}" \
    --output="${NNUNET_ROOT}/logs/${JOBNAME}-f%a-%j.out" \
    --error="${NNUNET_ROOT}/logs/${JOBNAME}-f%a-%j.err" \
    "${extra[@]}" \
    "${SCRIPT_DIR}/train_nnunet_job.sh"
}

prev="${AFTER}"
for ((hop = 1; hop <= CHAINS; hop++)); do
  jid="$(submit_one "${prev}")"
  if [[ -n "${prev}" ]]; then
    echo "hop=${hop} job=${jid} folds=${FOLDS} afterany:${prev}"
  else
    echo "hop=${hop} job=${jid} folds=${FOLDS}"
  fi
  prev="${jid}"
done
