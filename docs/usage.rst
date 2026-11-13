Usage
=====

This page lists the command-line entry points in the order they are run.
Paths default to the original cluster layout. Set them through environment
variables or edit the ``ROOT`` constants.

1. Prepare datasets
-------------------

.. code-block:: bash

   # Locked 5-fold splits (Stage A: copy; multi-site: --stratify by site prefix)
   python code/nnunet/make_nnunet_splits.py [--stratify] [--seed 2023]

   # Dataset002: 60 mm per-hemisphere crops around the manual PC
   python code/nnunet/prepare_nnunet_crop.py

   # Dataset006: Hammers + 3 offline photometric/resolution augmentations each
   python code/nnunet/prepare_nnunet_hammers_dr.py

   # Dataset007: crops + neighbouring labels + 5 stronger augmentations each
   python code/nnunet/prepare_nnunet_dense_crop_dr.py

   # SynthSeg-PC label maps: 60 mm box (fov) or whole brain (wb)
   python code/synthseg_pc/prepare_labels.py --mode fov --fold 0 --fov-mm 60

2. Train
--------

nnU-Net. Each Slurm "hop" runs for 12 h and resumes from the last checkpoint.

.. code-block:: bash

   nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
   NNUNET_ID=1 NNUNET_DATASET=Dataset001_Piriform FOLDS=0-4 CHAINS=4 \
       bash code/nnunet/submit_nnunet_train.sh

SynthSeg-PC (``RECIPE`` is ``v1``, ``v2``, or ``wb``):

.. code-block:: bash

   RECIPE=v2 bash code/synthseg_pc/submit_train.sh
   RECIPE=v2 bash code/synthseg_pc/submit_validate.sh   # epoch pick on real T1s

3. Inference with a prior crop
------------------------------

.. code-block:: bash

   # Deployable crop centre: register one Hammers atlas (affine) and warp its PC labels
   bash code/evaluation/warp_atlas_pc_prior.sh T1.nii.gz atlas_pc.nii.gz a01

   # Build per-hemisphere 60 mm crops around the prior centroids
   python code/evaluation/make_pc_crop.py --t1 T1.nii.gz --jlf atlas_pc.nii.gz \
       --prior-mode atlas_pc --out crops/fov

   # ... run the model on crops/fov/{lh,rh}.crop.nii.gz, then stitch back to native space
   python code/evaluation/make_pc_crop.py --stitch --t1 T1.nii.gz \
       --lh-pred lh_pred.nii.gz --rh-pred rh_pred.nii.gz --stitched pred_native.nii.gz

4. Quality control and scoring
------------------------------

.. code-block:: bash

   # Fail-closed field-of-view gate (exit code 2 on failure)
   python code/evaluation/qc_priorcrop.py --subject S --crop-root crops \
       --jlf reference.nii.gz --out qc.tsv

   # Three-way Hammers table (MAPER / JLF / nnU-Net)
   python code/evaluation/score_hammers_stage_a.py

   # Cohort scoring with automatic gold-label detection (125/126 or 1/2)
   python code/evaluation/score_priorcrop.py ...

Self-tests
----------

Several modules include a synthetic self-test that runs without patient data:

.. code-block:: bash

   python code/evaluation/make_pc_crop.py --self-test
   python code/evaluation/score_hammers_stage_a.py --self-test
