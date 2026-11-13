Experiment log
==============

Each experiment is recorded as *hypothesis → design → result → decision*.
Status key: **Final** means completed under the locked protocol.
**Provisional** means a mid-training checkpoint or small spot-check.
**Parked** means stopped at the pre-set fail threshold.

.. contents::
   :local:
   :depth: 1

E0 · Multi-atlas baselines (Final)
----------------------------------

:Hypothesis: Reproducing MAPER on Hammers will match the published agreement
   (Jaccard ≈ 0.50). If not, the evaluation code is wrong.
:Design: Leave-one-out on 30 brains, 29 atlases each, for MAPER and JLF.
   Same scorer as all later models.
:Result: MAPER Jaccard 0.495 / 0.462, JLF 0.494 / 0.445 (L/R). The left
   hemisphere matches the published bar.
:Decision: The scorer is calibrated. JLF becomes the production method.

E1 · Stage A: supervised nnU-Net, Dataset001 (Final)
----------------------------------------------------

:Hypothesis: A 3D U-Net trained on 24 brains per fold beats multi-atlas
   methods in-domain.
:Design: ``3d_fullres``, no mirroring, locked 5-fold, 1000 epochs, cuDNN off.
:Result: Dice 0.708 / 0.693 vs MAPER 0.659 / 0.627 and JLF 0.659 / 0.613.
   Overlap and surface distance are significantly better than both baselines
   (paired Wilcoxon, Holm-corrected p ≤ 0.003). Volume error is significantly
   better than JLF but **not** than MAPER (p = 0.44); see :ref:`paired-stats`.
:Decision: In-domain question answered. Move to multi-site.

E2 · Stage B: Hammers + ADNI, Dataset003 (Parked)
-------------------------------------------------

:Hypothesis: Adding 37 ADNI brains to training makes the model work on ADNI.
:Design: Retrain from scratch, splits stratified by site.
:Result: Hammers ~0.68, ADNI **~0.02**, with 81% of left-side Dice exactly 0.
   Predictions mostly empty (predicted/true volume ≈ 0.27).
:Decision: Diagnose before tuning.

E3 · Intensity harmonisation, Dataset004 (Parked)
-------------------------------------------------

:Hypothesis: The 12× intensity-scale difference is the cause.
:Design: Percentile harmonisation of ADNI to Hammers
   (:ref:`intensity-fingerprint`).
:Result: ADNI **~0.04**. Predictions are now non-empty but in the wrong place.
:Decision: Scale is a cause but not the only one.

E4 · ADNI-only control, Dataset005 (Parked)
-------------------------------------------

:Hypothesis: Hammers dominates the mixed training set, and ADNI alone is
   learnable.
:Design: Train on ADNI only.
:Result: Dice 0.017 / 0.001. Best training-log Dice 0.074 over 173 epochs.
:Decision: ADNI is not learnable with this pipeline. **Park the ADNI branch**
   (2026-10-21). Dataset006 (Hammers + offline augmentation) was cancelled
   with it.

E5 · Crop datasets, Dataset002 and Dataset007 (Provisional)
-----------------------------------------------------------

:Hypothesis: A 60 mm crop raises the foreground fraction ~60×
   (:ref:`class-imbalance`) and helps transfer. Neighbouring labels and stronger
   augmentation (007) help further.
:Design: Per-hemisphere crops around the manual PC; same folds.
:Result: With a correct crop and matched preprocessing, both produce
   anatomically placed masks on the target cohort. The more complex 007 did
   **not** beat 002.
:Decision: Prefer the simpler model with only 30 training brains.

E6 · Crop prior on the target cohort (Provisional)
--------------------------------------------------

:Hypothesis: Reusing the claustrum pipeline's putamen-seeded crop works for PC.
:Result: Near-failure for every model. The putamen is 45–50 mm from PC, so
   the crop contained none of it (:ref:`fov-geometry`).
:Decision: Use an atlas-registered prior. Add the fail-closed FoV gate.

E7 · Skull-strip match on the target cohort (Provisional)
---------------------------------------------------------

:Hypothesis: The model is robust to the cohort's own brain mask.
:Result: Near-empty predictions on the FastSurfer mask, even with the
   oracle crop. Normal-sized masks after re-stripping with the training
   pipeline.
:Decision: Inference must reproduce training preprocessing.

E8 · ADNI with crop models (Parked)
-----------------------------------

:Hypothesis: The ADNI failure is a localisation problem.
:Design: Atlas crop, then oracle crop (centred on the manual label), then
   oracle + matched strip.
:Result: All 0. Atlas registration missed by 57–120 mm, and even the
   oracle crop gave empty predictions.
:Decision: The ADNI gap is in image appearance, which motivates E9.

E9 · SynthSeg-PC v1 (Provisional)
---------------------------------

:Hypothesis: Synthetic-image training removes the dependence on matched
   preprocessing.
:Design: GMM generator on Hammers label maps, no flipping, epoch picked on
   real T1s (:doc:`theory/synthseg`).
:Result: Hammers 0.467 at epoch 13. On the target cohort it still produced
   masks **without intensity matching**, where supervised models were empty,
   but agreed with JLF less than the supervised crop model. ADNI 0 (crop
   missed by more than 25 mm).
:Decision: It is more robust but below the bar. Scale up (E10).

E10 · SynthSeg-PC v2 and wb (Paused)
------------------------------------

:Design: Wider U-Net (32), stronger deformation and bias, 100 epochs,
   topology fix, optional SynthSR; whole-brain ablation (wb).
:Status: v2 paused at epoch 6 for review. The next steps (ADNI with an
   oracle crop for v1, resume v2, unmatched scoring) are prepared as a
   dry-run plan.
