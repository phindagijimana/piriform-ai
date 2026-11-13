Outputs
=======

The derivative layout below is produced on the cluster. None of these
files are in the repository, because they are subject data.

nnU-Net
-------

.. code-block:: text

   nnunet/
   ├── nnUNet_raw/DatasetXXX_Name/
   │   ├── dataset.json              # channel names, label IDs, numTraining
   │   ├── imagesTr/<case>_0000.nii.gz
   │   └── labelsTr/<case>.nii.gz    # 0 background, 1 PC left, 2 PC right
   ├── nnUNet_preprocessed/DatasetXXX_Name/
   │   ├── nnUNetPlans.json          # patch size, spacing, network topology
   │   └── splits_final.json         # locked folds
   ├── nnUNet_results/DatasetXXX_Name/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/
   │   └── fold_k/{checkpoint_latest.pth, checkpoint_best.pth, checkpoint_final.pth, progress.png}
   ├── splits/DatasetXXX_Name.json   # copy of the locked split
   └── tables/*.tsv, *.json          # per-subject scores (never committed)

SynthSeg-PC
-----------

.. code-block:: text

   synthseg_pc/
   ├── config/
   │   ├── generation_labels.npy     # label IDs the generator paints
   │   ├── generation_classes.npy    # which labels share a GMM intensity class
   │   ├── segmentation_labels.npy   # labels the network predicts
   │   ├── evaluation_labels.npy     # labels that are scored (PC only)
   │   └── label_meta.json           # flipping=false, citations
   ├── data/{labels_train, val_images, val_labels}/
   └── models/fold0[_v2|_wb]/
       ├── train_run.json            # full recipe (hyperparameters)
       ├── wl2_XXX.h5, dice_XXX.h5   # one checkpoint per epoch
       └── ../fold0[_v2]_val/epoch_pick.json

Prior-crop inference
--------------------

.. code-block:: text

   <subject>/priorcrop_<model>_m<prior>/
   ├── fov/
   │   ├── prior_blob.{lh,rh}.nii.gz # 8 mm sphere at the prior centroid
   │   ├── {lh,rh}.crop.nii.gz       # 60 mm crops
   │   └── crop_centres.json         # prior mode, centroids (RAS mm)
   └── pred_native/
       ├── <subject>_priorcrop.nii.gz
       ├── <subject>_priorcrop_topo.nii.gz
       └── <subject>_priorcrop.json  # voxel counts, mm³ per side

Score tables
------------

Scores are tab-separated with one row per subject. The columns are
``dice_{L,R}``, ``jaccard_{L,R}``, ``asd_{L,R}_mm``, ``vol_gt_*``,
``vol_pred_*``, ``dvol_*``, ``n_cc_*``, ``n_holes_*``, ``largest_cc_frac_*``,
``dice_hemi``, and ``gold_labels``. Every scorer records which gold label
convention it detected (:ref:`label-id-bug`).
