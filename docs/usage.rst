Running the code
================

The command-line entry points, in the order they are run. Paths point at a
placeholder root (``/path/to/``); set them through environment variables or
edit the ``ROOT`` constants. Imaging data are not included.

Environments
------------

Three separate environments are needed, because SynthSeg's TensorFlow 2.2
cannot coexist with the PyTorch build nnU-Net v2 requires.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Environment
     - Contents
     - Used for
   * - nnU-Net
     - Apptainer image with PyTorch + ``nnunetv2``
     - Planning, training, inference
   * - SynthSeg
     - Python 3.8, TensorFlow 2.2, SynthSeg source
     - SynthSeg-PC training and validation
   * - Registration
     - ANTs, MIRTK, NiftySeg
     - Brain extraction, atlas warps, MAPER, JLF

Training ran on NVIDIA L40S GPUs split into 24 GB MIG slices. On these slices
cuDNN has no usable 3D convolution engine, so every nnU-Net command goes
through ``code/nnunet/nnunet_entry.py``, which disables cuDNN
(:doc:`programming/nnunet_training`).

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
   python code/evaluation/paired_stats.py --self-test

Known-answer unit tests for the metric library (Dice, surface distance,
6-connected topology, the topology fix) and the self-tests above run in CI on
every push:

.. code-block:: bash

   pip install numpy scipy nibabel pytest
   python -m pytest -q tests
