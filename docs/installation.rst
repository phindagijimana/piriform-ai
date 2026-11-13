Installation
============

The pipeline uses three independent software environments, because the
upstream tools have incompatible dependencies. TensorFlow 2.2 for SynthSeg
cannot coexist with the PyTorch build required by nnU-Net v2.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Environment
     - Contents
     - Used for
   * - **nnU-Net**
     - Apptainer image with PyTorch + ``nnunetv2``
     - Dataset planning, training, inference (Datasets 001–007)
   * - **SynthSeg**
     - Python 3.8, TensorFlow 2.2, Keras, SynthSeg source tree
     - SynthSeg-PC training, validation, prediction
   * - **Registration**
     - ANTs (via an fMRIPrep image), MIRTK AppImage, NiftySeg
     - Brain extraction, atlas warps, MAPER, JLF
   * - **Analysis** (host)
     - Python 3 with ``nibabel``, ``numpy``, ``scipy``, ``matplotlib``
     - Crops, scoring, QC, figures

Hardware
--------

* NVIDIA L40S partitioned with **MIG** into 24 GB slices
  (Slurm GRES ``gpu:l40s.24g:1``). nnU-Net ``3d_fullres`` fits in one slice.
* Multi-atlas methods (MAPER, JLF) are CPU-only. They are dominated by image
  registration and have no GPU code path.

.. warning::

   On MIG slices with this container, cuDNN offers **no usable 3D
   convolution engine**. nnU-Net must run with ``torch.backends.cudnn.enabled
   = False``. See :doc:`programming/nnunet_training` and :doc:`troubleshooting`.

nnU-Net environment
-------------------

.. code-block:: bash

   export nnUNet_raw=/path/to/piriform/nnunet/nnUNet_raw
   export nnUNet_preprocessed=/path/to/piriform/nnunet/nnUNet_preprocessed
   export nnUNet_results=/path/to/piriform/nnunet/nnUNet_results
   export nnUNet_compile=false        # torch.compile also fails on MIG + cuDNN

   # All nnU-Net commands go through the cuDNN-off entry point:
   python code/nnunet/nnunet_entry.py train 1 3d_fullres 0 -tr nnUNetTrainerNoMirroring

SynthSeg environment
--------------------

.. code-block:: bash

   export SYNTHSEG_ROOT=/path/to/SynthSeg        # github.com/BBillot/SynthSeg
   bash code/synthseg_pc/doctor.sh               # checks python, TF, GPU, paths

Building this documentation
---------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html
