#!/usr/bin/env bash
# One fold of nnU-Net Stage A. sbatch copies this file; use BEYONDFS_MODULES.
set -euo pipefail
SCRIPT_DIR="${BEYONDFS_MODULES:-/path/to/piriform/beyondfs/modules}"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainerNoMirroring}"
# shellcheck source=nnunet_exec.sh
source "${SCRIPT_DIR}/nnunet_exec.sh"
nnunet_env_setup

FOLD="${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}"
echo "start=$(date -Iseconds) host=$(hostname) fold=${FOLD} trainer=${TRAINER} compile=${nnUNet_compile}"
nvidia-smi -L || true

nnunet_exec python -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); sys.exit(0 if torch.cuda.is_available() else 1)"

if ! nnunet_exec python -c "import nnunetv2"; then
  nnunet_exec pip install --user nnunetv2
fi
nnunet_disable_find

if [[ ! -f "${nnUNet_preprocessed}/${NNUNET_DATASET}/nnUNetPlans.json" ]]; then
  echo "=== plan_and_preprocess ==="
  nnunet_exec nnUNetv2_plan_and_preprocess -d "${NNUNET_ID}" --verify_dataset_integrity
fi

python3 "${SCRIPT_DIR}/make_nnunet_splits.py"

if fold_done "${FOLD}"; then
  echo "fold ${FOLD} already has checkpoint_final.pth; skip"
  exit 0
fi

CONTINUE=()
if [[ -f "$(fold_dir "${FOLD}")/checkpoint_latest.pth" ]]; then
  CONTINUE=(--c)
  echo "resuming fold ${FOLD} from checkpoint_latest.pth"
fi

nnunet_exec python "${SCRIPT_DIR}/nnunet_entry.py" train "${NNUNET_ID}" "${CONFIG}" "${FOLD}" -tr "${TRAINER}" "${CONTINUE[@]}"
echo "end=$(date -Iseconds) fold=${FOLD}"
ls -la "$(fold_dir "${FOLD}")" || true
