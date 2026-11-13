For reviewers: the five-minute version
======================================

**Philbert Ndagijimana** · `phindagiji@gmail.com <mailto:phindagiji@gmail.com>`__ ·
`github.com/phindagijimana <https://github.com/phindagijimana>`__

The task
--------

Segment the human piriform cortex, a ~0.5 cm³, 2–3 mm thick strip of
olfactory cortex relevant to epilepsy surgery, on T1 MRI. The only published
automatic method (multi-atlas MAPER) agrees with manual tracing at Jaccard
≈ 0.50. I asked whether a trained 3D CNN can beat it, and whether it still
works on scans from other sites.

Findings
--------

1. **In-domain, the CNN wins.** On 30 manually labelled brains, each held
   out once, nnU-Net reached Dice 0.71 / 0.69 (left / right) vs 0.66 / 0.63
   for MAPER and 0.66 / 0.61 for joint label fusion. The gain in overlap and
   surface distance is significant (paired Wilcoxon, Holm-corrected
   p ≤ 0.003; nnU-Net better on 20–28 of 30 brains). The volume-error gain
   over MAPER is **not** significant (:ref:`paired-stats`).
2. **It did not transfer to a second site, and I found out why rather than
   tuning blindly.** On ADNI, Dice stayed near 0.02. Controlled follow-ups
   ruled out explanations one at a time: intensity harmonisation (0.04),
   ADNI-only training (fits the training brains, 0.02 on held-out), and
   correctly centred crops (still empty). The branch was stopped at a fail
   threshold fixed before training (:doc:`experiments`).
3. **A pooled validation metric hid a total failure.** The mixed-site
   training curve showed a pseudo-Dice of ~0.45 that looked like slow
   learning; scoring per site showed 0.68 on one site and 0.02 on the other
   (:doc:`training_dynamics`).
4. **Geometry mattered more than the model** on new data: a crop centred on
   the wrong structure contained 0% of the target, and a mismatched
   skull-stripping step produced empty masks. Both now fail closed in the
   inference pipeline (:ref:`fov-geometry`).
5. **Production stays classical.** No learned model is deployed until it is
   validated against manual labels on the target cohort.

What I built
------------

* The experimental protocol: success and fail thresholds set before
  training, locked and site-stratified splits, a scorer calibrated against a
  published result before any model was trained.
* Dataset builders for seven nnU-Net variants (whole-brain, crops,
  multi-site, harmonised, offline augmentation, dense neighbour labels) and
  the SynthSeg-PC label maps and training recipes.
* Training on a shared GPU cluster: a no-mirroring trainer (left–right flips
  are invalid for this task), a cuDNN workaround for MIG GPU slices, and
  checkpoint chaining across 12-hour Slurm limits.
* Inference: atlas-registered crop priors, stitching back to native space,
  and a fail-closed field-of-view gate.
* Evaluation: the metric library, per-site scoring, paired statistics, and a
  test suite that runs in CI. I also found and fixed a label-ID bug that had
  produced plausible-looking numbers (:ref:`label-id-bug`).

What I reused
-------------

The piriform labels (Steinbart et al.), the Hammers atlas and ADNI images,
nnU-Net and SynthSeg as training frameworks, and ANTs / MIRTK / NiftySeg for
registration and the multi-atlas baselines. Every source is cited in
:doc:`references`, and :doc:`literature` explains which paper drove which
decision.

What I would do next
--------------------

1. Draw 10–20 manual labels on the target cohort. Without them, any
   accuracy claim there is agreement with a reference, not accuracy.
2. Finish the synthetic-image model (SynthSeg-PC v2) with a full per-epoch
   validation sweep on real images.
3. Cut the training schedule: the curves plateau within ~100 of 1000 epochs.

Where to look
-------------

.. list-table::
   :widths: 30 70

   * - 10 minutes
     - :doc:`experiments` (hypothesis → result → decision for each step) and
       :doc:`lessons`
   * - The evidence
     - :doc:`results`, :doc:`training_dynamics`
   * - Depth
     - :doc:`theory/segmentation` onwards (math), :doc:`literature`
   * - Code
     - :doc:`programming/nnunet_training` onwards; ``code/`` and ``tests/``
