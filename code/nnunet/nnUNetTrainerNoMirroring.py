# Do not put this path on PYTHONPATH. nnU-Net loads the packaged trainer:
# nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring
# Train: nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerNoMirroring
# Left/right PC (labels 1/2) must not be mirrored.
# L40S MIG FIND crash: beyondfs/modules/disable_cudnn_find.py + nnunet_exec.sh env.
