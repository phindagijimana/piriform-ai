#!/usr/bin/env bash
# Worker for SynthSeg-PC training (invoked by submit_train.sh / sbatch).
# Do not resolve paths from BASH_SOURCE — sbatch copies the script to /var/spool/slurmd/.
set -euo pipefail
PIRIFORM_ROOT="${SLURM_SUBMIT_DIR:-/path/to/piriform}"
SCRIPT_DIR="${PIRIFORM_ROOT}/synthseg_pc"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

mkdir -p "${SSP_LOGS}" "${SSP_MODELS}"
FOLD="${SSP_FOLD:-0}"
# RECIPE=v2 → models/fold0_v2 (fresh arch); RECIPE=wb → fold0_wb; else fold0
RECIPE="${RECIPE:-v1}"
case "${RECIPE}" in
  v2) MODEL_DIR="${SSP_MODELS}/fold${FOLD}_v2" ;;
  wb) MODEL_DIR="${SSP_MODELS}/fold${FOLD}_wb" ;;
  *) MODEL_DIR="${SSP_MODELS}/fold${FOLD}" ;;
esac
mkdir -p "${MODEL_DIR}"

if [[ ! -x "${SYNTHSEG_ENV}" ]]; then
  echo "ERROR: SynthSeg python missing: ${SYNTHSEG_ENV}" >&2
  exit 1
fi
if [[ ! -d "${SSP_DATA}/labels_train" ]]; then
  echo "ERROR: run prepare_labels.py first (SSP_DATA=${SSP_DATA})" >&2
  exit 1
fi

CKPT=""
if [[ -n "${CHECKPOINT:-}" ]]; then
  CKPT="${CHECKPOINT}"
elif ls "${MODEL_DIR}"/dice_*.h5 >/dev/null 2>&1; then
  CKPT="$(ls -1 "${MODEL_DIR}"/dice_*.h5 | sort -V | tail -1)"
  echo "Resuming from ${CKPT}"
fi

WL2="${WL2_EPOCHS:-1}"
if [[ -n "${CKPT}" && "${CKPT}" == *"/dice_"* ]]; then
  WL2=0
  echo "dice resume → WL2_EPOCHS=0 (continue to dice_epochs=${DICE_EPOCHS:-50})"
fi

EXTRA=()
if [[ -n "${CKPT}" ]]; then
  EXTRA+=(--checkpoint "${CKPT}")
fi

# v2 defaults (levers 3–4): richer GMM + bigger U-Net + longer dice
if [[ "${RECIPE}" == "v2" ]]; then
  UNET_FEAT_COUNT="${UNET_FEAT_COUNT:-32}"
  DICE_EPOCHS="${DICE_EPOCHS:-100}"
  SCALING_BOUNDS="${SCALING_BOUNDS:-0.3}"
  ROTATION_BOUNDS="${ROTATION_BOUNDS:-20}"
  SHEARING_BOUNDS="${SHEARING_BOUNDS:-0.02}"
  NONLIN_STD="${NONLIN_STD:-5.0}"
  BIAS_FIELD_STD="${BIAS_FIELD_STD:-1.0}"
elif [[ "${RECIPE}" == "wb" ]]; then
  # wb ablation: same capacity as v2, FoV labels replaced by full-brain maps
  UNET_FEAT_COUNT="${UNET_FEAT_COUNT:-32}"
  DICE_EPOCHS="${DICE_EPOCHS:-100}"
  SCALING_BOUNDS="${SCALING_BOUNDS:-0.3}"
  ROTATION_BOUNDS="${ROTATION_BOUNDS:-20}"
  SHEARING_BOUNDS="${SHEARING_BOUNDS:-0.02}"
  NONLIN_STD="${NONLIN_STD:-5.0}"
  BIAS_FIELD_STD="${BIAS_FIELD_STD:-1.0}"
  OUTPUT_SHAPE="${OUTPUT_SHAPE:-192}"
else
  UNET_FEAT_COUNT="${UNET_FEAT_COUNT:-24}"
  DICE_EPOCHS="${DICE_EPOCHS:-50}"
  SCALING_BOUNDS="${SCALING_BOUNDS:-0.2}"
  ROTATION_BOUNDS="${ROTATION_BOUNDS:-15}"
  SHEARING_BOUNDS="${SHEARING_BOUNDS:-0.012}"
  NONLIN_STD="${NONLIN_STD:-4.0}"
  BIAS_FIELD_STD="${BIAS_FIELD_STD:-0.7}"
fi

export PYTHONPATH="${SYNTHSEG_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

echo "SynthSeg-PC train recipe=${RECIPE} fold=${FOLD} wl2=${WL2} dice_epochs=${DICE_EPOCHS} steps=${STEPS_PER_EPOCH:-1000} unet_feat=${UNET_FEAT_COUNT} model_dir=${MODEL_DIR}"
"${SYNTHSEG_ENV}" "${SCRIPT_DIR}/train.py" \
  --labels-dir "${SSP_DATA}/labels_train" \
  --model-dir "${MODEL_DIR}" \
  --config "${SSP_CONFIG}" \
  --output-shape "${OUTPUT_SHAPE:-160}" \
  --wl2-epochs "${WL2}" \
  --dice-epochs "${DICE_EPOCHS}" \
  --steps-per-epoch "${STEPS_PER_EPOCH:-1000}" \
  --unet-feat-count "${UNET_FEAT_COUNT}" \
  --scaling-bounds "${SCALING_BOUNDS}" \
  --rotation-bounds "${ROTATION_BOUNDS}" \
  --shearing-bounds "${SHEARING_BOUNDS}" \
  --nonlin-std "${NONLIN_STD}" \
  --bias-field-std "${BIAS_FIELD_STD}" \
  --recipe "${RECIPE}" \
  "${EXTRA[@]}"
