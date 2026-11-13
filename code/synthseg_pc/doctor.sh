#!/usr/bin/env bash
# Confirm TF/Python 3.8 SynthSeg env + label paths (implementation.md checklist).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

ok=1
export PYTHONPATH="${SYNTHSEG_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
echo "SYNTHSEG_ENV=${SYNTHSEG_ENV}"
echo "SYNTHSEG_ROOT=${SYNTHSEG_ROOT}"
if [[ -x "${SYNTHSEG_ENV}" ]]; then
  "${SYNTHSEG_ENV}" - <<'PY'
import sys
print("python", sys.version.split()[0])
import tensorflow as tf
print("tensorflow", tf.__version__)
from SynthSeg.training import training  # noqa: F401
print("SynthSeg.training OK")
PY
else
  echo "FAIL: missing ${SYNTHSEG_ENV}"
  ok=0
fi

for p in "${SYNTHSEG_ROOT}" "${ATLAS_DIR}" "${SEG95PC_DIR}" "${CROP_HELPER}" "${SPLITS_JSON}"; do
  if [[ -e "${p}" ]]; then
    echo "OK  ${p}"
  else
    echo "FAIL ${p}"
    ok=0
  fi
done

n_seg=$(ls -1 "${SEG95PC_DIR}"/*.nii.gz 2>/dev/null | wc -l)
n_brain=$(ls -1 "${ATLAS_DIR}"/*_BrainExtractionBrain.nii.gz 2>/dev/null | wc -l)
echo "seg95pc=${n_seg} atlas_brains=${n_brain}"
[[ "${n_seg}" -eq 30 && "${n_brain}" -eq 30 ]] || ok=0

if [[ "${ok}" -eq 1 ]]; then
  echo "doctor: PASS"
  exit 0
fi
echo "doctor: FAIL"
exit 1
