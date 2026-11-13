#!/usr/bin/env bash
# Warp one Hammers atlas PC (125/126) onto a subject T1 — prior for FoV crop.
# Does NOT use subject JLF. Gold scoring still uses JLF separately.
#
# Usage: warp_atlas_pc_prior.sh T1.nii.gz OUT_LABELS.nii.gz [ATLAS_ID=a01]
set -euo pipefail
T1="${1:?usage: warp_atlas_pc_prior.sh T1.nii.gz OUT.nii.gz [a01]}"
OUT="${2:?}"
AID="${3:-a01}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

ATLAS_DIR="${PIRIFORM_DIR}/atlas"
ATLAS_T1="${ATLAS_DIR}/${AID}_BrainExtractionBrain.nii.gz"
ATLAS_PC="${ATLAS_DIR}/${AID}_PC_labels.nii.gz"
[[ -s "${ATLAS_T1}" && -s "${ATLAS_PC}" ]] || {
  echo "ERROR: missing atlas ${AID} under ${ATLAS_DIR}" >&2
  exit 1
}

FMRIPREP_SIF="${FMRIPREP_SIF:-/opt/ood_apps/images/fmriprep/fmriprep_25.2.3.sif}"
ANTS_BIN="${ANTS_BIN:-/app/.pixi/envs/fmriprep/bin}"
WORKDIR="$(dirname "${OUT}")/atlas_pc_reg_${AID}"
mkdir -p "${WORKDIR}"
PREFIX="${WORKDIR}/atlas2t1_"

run_ants() {
  local cmd=("$@")
  if command -v antsRegistrationSyNQuick.sh >/dev/null 2>&1; then
    env ANTSPATH="${ANTSPATH:-}" "${cmd[@]}"
  elif command -v apptainer >/dev/null 2>&1 && [[ -f "${FMRIPREP_SIF}" ]]; then
    apptainer exec --bind /mnt:/mnt "${FMRIPREP_SIF}" \
      env ANTSPATH="${ANTS_BIN}/" PATH="${ANTS_BIN}:${PATH}" "${cmd[@]}"
  elif command -v singularity >/dev/null 2>&1 && [[ -f "${FMRIPREP_SIF}" ]]; then
    singularity exec --bind /mnt:/mnt "${FMRIPREP_SIF}" \
      env ANTSPATH="${ANTS_BIN}/" PATH="${ANTS_BIN}:${PATH}" "${cmd[@]}"
  else
    echo "ERROR: ANTs not on PATH and no fmriprep SIF" >&2
    exit 1
  fi
}

# Reuse affine if present
if [[ ! -s "${PREFIX}0GenericAffine.mat" ]]; then
  echo "atlas_pc: register ${AID} → $(basename "${T1}") (affine)"
  # -t a = affine only (FoV prior; 60 mm crop tolerates residual error)
  run_ants antsRegistrationSyNQuick.sh -d 3 -f "${T1}" -m "${ATLAS_T1}" \
    -o "${PREFIX}" -t a -n "${SLURM_CPUS_PER_TASK:-4}"
fi
[[ -s "${PREFIX}0GenericAffine.mat" ]] || {
  echo "ERROR: registration failed (${PREFIX}0GenericAffine.mat missing)" >&2
  exit 1
}

echo "atlas_pc: warp PC labels → ${OUT}"
run_ants antsApplyTransforms -d 3 -i "${ATLAS_PC}" -r "${T1}" -o "${OUT}" \
  -n GenericLabel -t "${PREFIX}0GenericAffine.mat"
[[ -s "${OUT}" ]] || { echo "ERROR: warped prior missing ${OUT}" >&2; exit 1; }

# Fail closed if PC labels empty after warp
env -u PYTHONPATH -u PYTHONUSERBASE /usr/bin/python3 - "${OUT}" <<'PY'
import sys, nibabel as nib, numpy as np
p = sys.argv[1]
d = np.asanyarray(nib.load(p).dataobj)
n125, n126 = int((d == 125).sum()), int((d == 126).sum())
print(f"warped PC voxels: 125={n125} 126={n126}")
if n125 < 10 or n126 < 10:
    raise SystemExit(f"ERROR: empty/near-empty atlas_pc prior {p}")
PY
echo "atlas_pc prior ok: ${OUT}"
