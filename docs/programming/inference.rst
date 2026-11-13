Prior-crop inference
====================

Inference on a new scan has four steps: get a crop centre (the *prior*),
crop, predict, and stitch back. Then the result goes through a fail-closed
quality gate.

Deployable prior: atlas registration
------------------------------------

``code/evaluation/warp_atlas_pc_prior.sh`` registers one Hammers atlas to the
subject with an **affine** transform only. A 60 mm crop tolerates a few
millimetres of error, and affine registration is fast and hard to break. The
atlas PC labels are then warped with ``GenericLabel`` interpolation, which is
label-preserving. The script exits non-zero if the warped prior is empty:

.. literalinclude:: ../../code/evaluation/warp_atlas_pc_prior.sh
   :language: bash
   :start-at: # Reuse affine if present

Crop centres
------------

``code/evaluation/make_pc_crop.py`` turns a prior label map into one centre
per hemisphere. The centroid of the label voxels is mapped to world
coordinates:

.. literalinclude:: ../../code/evaluation/make_pc_crop.py
   :language: python
   :pyobject: centroid_ras

The oracle mode detects which label convention the reference uses, because
ADNI manuals are 1/2 while atlas/JLF outputs are 125/126:

.. literalinclude:: ../../code/evaluation/make_pc_crop.py
   :language: python
   :pyobject: _pc_labmap

Stitching back to native space
------------------------------

Each hemisphere crop is resampled back onto the subject grid with nearest
neighbour interpolation. The side that belongs to that crop wins, and the
other side only fills voxels that are still empty:

.. literalinclude:: ../../code/evaluation/make_pc_crop.py
   :language: python
   :pyobject: stitch

Field-of-view gate (fail closed)
--------------------------------

``code/evaluation/qc_priorcrop.py`` implements the inequality of
:ref:`fov-geometry` as hard thresholds:

.. literalinclude:: ../../code/evaluation/qc_priorcrop.py
   :language: python
   :start-at: MAX_PRIOR_TO_PC_MM
   :end-before: MIN_CROP_MEAN

It measures, per hemisphere, the prior-to-reference distance and the
fraction of reference PC voxels inside the crop. It also checks that the
prediction is not empty:

.. literalinclude:: ../../code/evaluation/qc_priorcrop.py
   :language: python
   :start-at: # Hard fail if either hemi
   :end-before: return row

The process exits with code 2 on failure, so a Slurm pipeline stops instead
of scoring a crop that never contained the structure.

Self-test
---------

``make_pc_crop.py --self-test`` builds a synthetic 40³ volume with one
voxel per hemisphere, crops, fakes predictions, stitches, and asserts that
both labels survive. That tests the geometry path end to end without any
patient data.
