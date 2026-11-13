#!/usr/bin/env bash
# Optional Billot SynthSR preprocess (lever 7). Writes OUT nifti or copies IN if unavailable.
# Usage: bash maybe_synthsr.sh IN.nii.gz OUT.nii.gz
set -euo pipefail
IN="${1:?usage: maybe_synthsr.sh in.nii.gz out.nii.gz}"
OUT="${2:?}"
mkdir -p "$(dirname "${OUT}")"

if [[ "${SYNTHSR:-0}" != "1" ]]; then
  cp -f "${IN}" "${OUT}"
  echo "synthsr=skip (SYNTHSR!=1) out=${OUT}"
  exit 0
fi

SYNTHSR_BIN=""
for cand in \
  "${FREESURFER_HOME:-}/bin/mri_synthsr" \
  "$(command -v mri_synthsr 2>/dev/null || true)" \
  /usr/local/freesurfer/bin/mri_synthsr
do
  [[ -n "${cand}" && -x "${cand}" ]] || continue
  SYNTHSR_BIN="${cand}"
  break
done

if [[ -z "${SYNTHSR_BIN}" ]]; then
  echo "WARN: mri_synthsr not found — using original T1 (set FREESURFER_HOME or disable SYNTHSR=1)" >&2
  cp -f "${IN}" "${OUT}"
  echo "synthsr=missing_bin out=${OUT}"
  exit 0
fi

echo "synthsr=run bin=${SYNTHSR_BIN} in=${IN}"
# SynthSR CLI: mri_synthsr --i in --o out (FS 7.3+)
"${SYNTHSR_BIN}" --i "${IN}" --o "${OUT}" --cpu || {
  echo "WARN: mri_synthsr failed — falling back to original T1" >&2
  cp -f "${IN}" "${OUT}"
  echo "synthsr=failed_fallback out=${OUT}"
  exit 0
}
echo "synthsr=ok out=${OUT}"
