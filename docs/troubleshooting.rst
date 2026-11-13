Troubleshooting
===============

Each entry gives the symptom, the diagnosis, and the fix.

``FIND was unable to find an engine to execute this computation``
------------------------------------------------------------------

:Where: First nnU-Net training step on an L40S MIG slice.
:Attempts:
   1. ``torch.compile`` on: crash in compiled ``conv_transpose3d``.
   2. Compile off: same error in eager ``conv_transpose3d``.
   3. ``TORCH_CUDNN_V8_API_DISABLED=1``: FIND error gone, then ``no valid
      convolution algorithms available in CuDNN`` on ``conv3d``.
   4. ``torch.backends.cudnn.enabled = False``: **trains** at 168 s/epoch.
:Diagnosis: cuDNN in this container has no usable 3D convolution engine on
   MIG, through either API. Compilation was a red herring.
:Lesson: Change one variable at a time, and smoke-test one fold for two
   epochs before relaunching a 5-fold array.

``PartitionTimeLimit``
----------------------

:Cause: Requested walltime exceeds the partition maximum (24 h on a 12 h partition).
:Fix: Use 12 h hops with checkpoint chaining (:doc:`programming/hpc`).

``Job dependency problem``
--------------------------

:Cause: The chaining function printed log lines to stdout, so the captured
   job ID contained text.
:Fix: Logs go to stderr, and only the ID goes to stdout.

SynthSeg resume runs "Epoch 1/1" of warm-up
-------------------------------------------

:Cause: Resume logic did not skip the L2 warm-up phase.
:Fix: Force ``WL2_EPOCHS=0`` when the checkpoint is a ``dice_*`` file.

SynthSeg validation shape mismatch (64³ vs 61³)
-----------------------------------------------

:Cause: Resampling 0.94 mm crops to 1 mm during prediction.
:Fix: ``target_res=None``. Score with the resample-safe ``score_val.py``.

Helper files not found inside Slurm jobs
----------------------------------------

:Cause: ``sbatch`` runs a spooled copy of the script.
:Fix: Resolve paths from ``SLURM_SUBMIT_DIR``.

Empty predictions on a new cohort
---------------------------------

Check in this order:

1. **FoV gate.** Is the structure inside the crop? (:ref:`fov-geometry`)
2. **Preprocessing.** Is the brain extraction the same as in training?
   (:ref:`intensity-fingerprint`)
3. **Label IDs.** Does the scorer read the reference convention?
   (:ref:`label-id-bug`)

Empty MAPER output
------------------

:Cause: Vote collapse on a thin structure (:ref:`vote-collapse`).
:Fix: Use JLF. MAPER now fails loudly when fused labels are missing.

ADNI download (LONI IDA)
------------------------

* Diagnosis is a reporting field, not a filter. Search by subject ID.
* A "3T only" filter hid 25 of 30 subjects.
* Collection names are capped at 25 characters.
* Browser downloads did not reach the cluster. Use ``curl`` on the cluster
  with the IDA URL.
