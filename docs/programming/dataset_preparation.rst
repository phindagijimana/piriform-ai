Dataset preparation
===================

All dataset builders share three rules:

1. **Never modify an existing dataset.** Each variant gets a new
   ``DatasetXXX`` directory, so finished models stay reproducible.
2. **Reuse the locked folds.** Every variant derives its splits from
   Dataset001, so all comparisons are paired on the same held-out brains.
3. **Log counts, never pixels.** Scripts print voxel counts and paths only.

Locked, site-stratified splits
------------------------------

``code/nnunet/make_nnunet_splits.py``. If a locked split file exists, it is
copied verbatim. Otherwise cases are grouped by site prefix (``Hammers_``,
``ADNI_``) and dealt round-robin into folds, so each fold mixes sites in
proportion:

.. literalinclude:: ../../code/nnunet/make_nnunet_splits.py
   :language: python
   :pyobject: stratified_folds

Crop datasets: geometry in world coordinates
--------------------------------------------

``code/nnunet/prepare_nnunet_crop.py`` (Dataset002). The PC centroid is
computed in voxel space and mapped to scanner (RAS) millimetres with the image
affine, :math:`\mathbf{r} = A\,[i,j,k,1]^\top`. All geometry from then on is
in millimetres, which makes it independent of voxel size and orientation.
A small sphere around the centroid serves as the "prior blob" for the crop
helper:

.. literalinclude:: ../../code/nnunet/prepare_nnunet_crop.py
   :language: python
   :pyobject: sphere_mask

Labels are remapped from the atlas convention (125/126) to nnU-Net's
contiguous IDs (1/2). Any unexpected label aborts the build:

.. literalinclude:: ../../code/nnunet/prepare_nnunet_crop.py
   :language: python
   :pyobject: remap_crop

Splits are expanded so that **both hemispheres of a brain land in the same
fold**. Otherwise a model could train on a brain's left crop and be tested on
its right crop:

.. literalinclude:: ../../code/nnunet/prepare_nnunet_crop.py
   :language: python
   :pyobject: expand_splits

Offline augmentation (Dataset006 / 007)
---------------------------------------

The corruption function implements the equations in
:doc:`../theory/augmentation`. It changes intensities only, and labels are
copied unchanged:

.. literalinclude:: ../../code/nnunet/prepare_nnunet_hammers_dr.py
   :language: python
   :pyobject: corrupt

SynthSeg-PC label maps
----------------------

``code/synthseg_pc/prepare_labels.py`` builds the generator inputs. The
dense Hammers atlas is reduced to PC, amygdala, medial temporal lobe, and
"other". Left/right pairs share a generation class
(:ref:`generation-classes`):

.. literalinclude:: ../../code/synthseg_pc/prepare_labels.py
   :language: python
   :start-at: # Dense FoV remap
   :end-before: PC_NEED

.. literalinclude:: ../../code/synthseg_pc/prepare_labels.py
   :language: python
   :pyobject: remap_dense_arr

The configuration written alongside records ``flipping: false``. The
training script checks this before starting (:doc:`synthseg_training`).
