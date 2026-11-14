Engineering and lessons
=======================

The problems that cost the most time were not modelling problems. This page
records how they were diagnosed and what they taught.

Debugging a GPU failure one variable at a time
----------------------------------------------

The first nnU-Net training step on an L40S MIG slice failed with
``FIND was unable to find an engine to execute this computation``.

1. ``torch.compile`` on: crash in compiled ``conv_transpose3d``.
2. Compile off: the same error in eager ``conv_transpose3d``.
3. ``TORCH_CUDNN_V8_API_DISABLED=1``: the FIND error is gone, replaced by
   ``no valid convolution algorithms available in CuDNN`` on ``conv3d``.
4. ``torch.backends.cudnn.enabled = False``: **trains**, at 168 s/epoch.

Diagnosis: in this container, cuDNN has no usable 3D convolution engine on
MIG through either API, and compilation was a red herring. The fix is a thin
entry point rather than a patched nnU-Net (:doc:`programming/nnunet_training`).
Each attempt was a two-epoch smoke test on one fold, not a relaunched 5-fold
array.

Training past a 12-hour wall-clock limit
----------------------------------------

A job needing :math:`H` hours on a partition capped at :math:`h` hours is split
into :math:`\lceil H/h\rceil` hops:

.. math::

   H_\text{fold} = 1000 \times 168\ \text{s} \approx 47\ \text{h}
   \;\Rightarrow\; \lceil 47/12 \rceil = 4\ \text{hops per fold}.

Each hop is an ``sbatch --dependency=afterany:<previous>`` that resumes from
the latest checkpoint. Making this reliable required:

* **Idempotent workers.** A hop on a finished fold exits cleanly.
* **Only the job ID on stdout.** A helper that also printed progress to stdout
  produced a "job ID" containing text, and Slurm rejected the chain with
  ``Job dependency problem``. Logs now go to stderr.
* **Explicit walltime per hop.** A 4 h default timed SynthSeg hops out; 24 h on
  a 12 h partition left jobs pending forever (``PartitionTimeLimit``).
* **Paths from** ``SLURM_SUBMIT_DIR``. ``sbatch`` runs a spooled copy of the
  script, so ``BASH_SOURCE`` points at the spool and helper files vanish.
* **Dry-run by default.** Orchestration scripts print the full submission
  plan and queue nothing unless ``RUN=1``, so GPU time is reviewed before it
  is spent.

Bugs that produced plausible output
-----------------------------------

* **Label IDs.** A scorer read the wrong label convention and returned
  believable Dice values (:ref:`label-id-bug`). Scorers now report which
  labels they read.
* **SynthSeg resume.** Resuming from a ``dice_*`` checkpoint re-ran one epoch
  of the L2 warm-up phase. Resume now forces ``WL2_EPOCHS=0``.
* **Validation shape mismatch (64³ vs 61³).** Prediction resampled 0.94 mm
  crops to 1 mm. Fixed with ``target_res=None`` and a resample-safe scorer.
* **Silent MAPER failure.** Vote collapse on a thin structure gave empty
  masks (:ref:`vote-collapse`); the pipeline now fails loudly.

Lessons
-------

1. **Build the baseline first, and use it to test the evaluation code.**
   Reproducing MAPER's published number made every later score trustworthy.
2. **The label protocol beats the architecture.** The most important training
   decision, disabling mirroring, was a one-flag change derived from anatomy
   (:ref:`mirroring-proof`).
3. **A strong default is a strong baseline.** nnU-Net out of the box beat
   multi-atlas methods in-domain; added complexity (Dataset007) did not beat
   the simpler crop model (Dataset002).
4. **"Fails on a new site" is a symptom, not a diagnosis.** Each failure had a
   testable cause (crop geometry, skull stripping, intensity range), and each
   control ruled out one hypothesis.
5. **Check the input before blaming the model.** A crop containing 0% of the
   structure cannot pass; a geometric gate catches it in seconds.
6. **Set thresholds before looking at results, and report negative results**
   with their evidence.
7. **Infrastructure is ML work.** The cuDNN incident cost more time than any
   modelling decision.
