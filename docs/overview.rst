Overview
========

The problem
-----------

The piriform cortex (PC) is the largest part of the primary olfactory cortex.
It sits at the junction of the frontal and temporal lobes, next to the amygdala.

* **Epilepsy surgery.** In temporal lobe epilepsy, removing at least half of
  the PC increased the odds of seizure freedom about 16-fold
  :ref:`[Galovic 2019] <ref-galovic>`.
* **Clinical cohorts.** The only validated PC segmentation protocol was
  developed in patients with temporal lobe epilepsy and Alzheimer's disease
  :ref:`[Steinbart 2023] <ref-steinbart>`.

Why PC is hard to segment
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Property
     - Consequence (derived in :doc:`theory/segmentation`)
   * - Mean volume 486 / 461 mm³ (L/R) in the training brains
     - About 0.003% of a full-head image is foreground. Loss and metrics are
       dominated by boundary voxels.
   * - Thin (~2–3 mm), folded paleocortex
     - A one-voxel boundary error costs roughly a third of the Dice score
       (:ref:`thin-structure-dice`).
   * - Low contrast against amygdala and orbitofrontal cortex
     - Boundaries come partly from anatomical convention, not visible edges.
   * - Separate left and right classes
     - Standard left-right flip augmentation is *invalid*
       (:ref:`mirroring-proof`).

Datasets
--------

.. list-table::
   :header-rows: 1
   :widths: 18 8 44 30

   * - Dataset
     - n
     - What it provides
     - Role
   * - Hammers adult atlas
     - 30
     - T1 + Steinbart manual PC labels :ref:`[Hammers 2003] <ref-hammers>`
     - Training; in-domain test (5-fold CV)
   * - ADNI
     - 37
     - Steinbart manual PC labels; T1s from LONI IDA under the ADNI DUA
     - Second-site test
   * - Clinical target cohort
     - —
     - T1 only, no manual PC labels (private; not described here)
     - Deployment target; production uses JLF
   * - Bonn TLE
     - 19
     - Labels public, T1s not
     - Excluded (needs a data-sharing agreement)

Label conventions:

.. list-table::
   :header-rows: 1

   * - Source
     - Left PC
     - Right PC
   * - Hammers / Steinbart atlas; JLF output
     - 125
     - 126
   * - ADNI released manuals; nnU-Net training labels
     - 1
     - 2

Success criteria (set before training)
--------------------------------------

1. **In-domain:** beat the published MAPER agreement on Hammers (Jaccard ≈
   0.50), with all methods evaluated on the same brains.
2. **Multi-site:** per-hemisphere Dice ≥ **0.40** on ADNI. A Dice ≤ **0.10**
   after a fair attempt means the branch is parked.
3. **Target cohort:** non-empty, anatomically placed masks on the target
   cohort that agree with JLF. JLF is a reference here, not ground truth.

Project map
-----------

.. graphviz::

   digraph overview {
     rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=11];
     data [label="Data\nHammers 30 · ADNI 37\n· clinical target"];
     base [label="Classical baselines\nMAPER · JLF"];
     a [label="Stage A\nnnU-Net 5-fold\n(Dataset001)"];
     b [label="Stage B\nmulti-site\n(003–006)", fillcolor="#fde2e2"];
     crop [label="Crop variants\n(002, 007)"];
     ss [label="SynthSeg-PC\nv1 · v2 · wb", fillcolor="#fff4d6"];
     inf [label="Prior-crop inference\n+ FoV QC gate"];
     prod [label="Production\nJLF on target cohort", fillcolor="#e3f5e1"];
     data -> base -> a -> b; a -> crop -> inf; b -> ss -> inf; base -> prod;
   }
