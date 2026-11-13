nnU-Net training
================

Entry point with cuDNN disabled
-------------------------------

On MIG-partitioned L40S GPUs in this container, cuDNN could not find a 3D
convolution engine through either its new (v8) or legacy API
(:doc:`../troubleshooting`). Rather than patching nnU-Net, a thin entry point
sets the backend flags **before** nnU-Net is imported and then dispatches to
the standard console entry points:

.. literalinclude:: ../../code/nnunet/nnunet_entry.py
   :language: python

The pattern is a reusable one: a wrapper that pins global framework state,
logs what it pinned (``cudnn.enabled=False device=...``), and leaves the
upstream package untouched. PyTorch then uses its native CUDA convolution
kernels at ~168 s per epoch.

Trainer: no mirroring
---------------------

The trainer is nnU-Net's built-in ``nnUNetTrainerNoMirroring``, selected
with ``-tr``. It removes mirroring from training augmentation and test-time
augmentation (:ref:`mirroring-proof`):

.. code-block:: bash

   python code/nnunet/nnunet_entry.py train 1 3d_fullres $FOLD -tr nnUNetTrainerNoMirroring
   python code/nnunet/nnunet_entry.py train 1 3d_fullres $FOLD -tr nnUNetTrainerNoMirroring --c

Chained submission
------------------

``code/nnunet/submit_nnunet_train.sh`` submits a 5-fold job array and a
chain of follow-up "hops". Each hop waits for the previous one with
``afterany``. ``afterany`` rather than ``afterok`` matters here, because a hop
that hits its walltime ends in ``TIMEOUT``, which is not "ok":

.. literalinclude:: ../../code/nnunet/submit_nnunet_train.sh
   :language: bash
   :start-at: submit_one() {

A guard refuses to start a new chain on a fold that already has a running or
pending writer, since two jobs writing the same ``fold_k`` would corrupt its
checkpoints:

.. literalinclude:: ../../code/nnunet/submit_nnunet_train.sh
   :language: bash
   :start-at: # A hop with no AFTER
   :end-before: python3 "${SCRIPT_DIR}/make_nnunet_splits.py"

``sbatch --parsable`` returns only the job ID, so ``jid=$(submit_one ...)``
captures a clean value for the next dependency (:doc:`hpc`).

Worker
------

The array worker ``train_nnunet_job.sh`` skips a fold whose
``checkpoint_final.pth`` exists. Otherwise it resumes with ``--c`` from
``checkpoint_latest.pth``. That makes every hop **idempotent**: resubmitting
a finished chain does nothing.
