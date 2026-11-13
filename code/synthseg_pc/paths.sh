# shellcheck shell=bash
# SynthSeg-PC paths. Source from project scripts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIRIFORM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLAU_ROOT="${CLAU_ROOT:-$(cd "${PIRIFORM_ROOT}/../clau_seg" && pwd)}"

export SYNTHSEG_ROOT="${SYNTHSEG_ROOT:-${CLAU_ROOT}/tools/SynthSeg}"
export SYNTHSEG_ENV="${SYNTHSEG_ENV:-${CLAU_ROOT}/tools/env/bin/python3.8}"
export PYTHONPATH="${SYNTHSEG_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export ATLAS_DIR="${ATLAS_DIR:-${PIRIFORM_ROOT}/Piriform_Segmentation/atlas}"
export SEG95PC_DIR="${SEG95PC_DIR:-${PIRIFORM_ROOT}/Piriform_Segmentation/hammers/ancillaries/seg/seg95pc}"
export SPLITS_JSON="${SPLITS_JSON:-${PIRIFORM_ROOT}/nnunet/splits/Dataset001_Piriform.json}"
export CROP_HELPER="${CROP_HELPER:-${CLAU_ROOT}/tools/bin/crop_to_fov}"

export SSP_DATA="${SSP_DATA:-${SCRIPT_DIR}/data}"
export SSP_CONFIG="${SSP_CONFIG:-${SCRIPT_DIR}/config}"
export SSP_MODELS="${SSP_MODELS:-${SCRIPT_DIR}/models}"
export SSP_TABLES="${SSP_TABLES:-${SCRIPT_DIR}/tables}"
export SSP_LOGS="${SSP_LOGS:-${SCRIPT_DIR}/logs}"

# FoV + dense remap (Mauri-style scaffolding). Steinbart PC = labels 1/2 after remap.
export SSP_MODE="${SSP_MODE:-fov}"   # fov | wb
export SSP_FOLD="${SSP_FOLD:-0}"
export SSP_FOV_MM="${SSP_FOV_MM:-60}"
