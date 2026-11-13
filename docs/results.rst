Results
=======

In-domain: Hammers n=30 (Final)
-------------------------------

.. list-table::
   :header-rows: 1

   * - Method
     - Dice L / R
     - Jaccard L / R
     - ASD L / R (mm)
     - \|ΔV\| L / R (mm³)
   * - **nnU-Net** (5-fold)
     - **0.708 / 0.693**
     - **0.551 / 0.534**
     - **0.57 / 0.56**
     - **75 / 82**
   * - MAPER (leave-one-out)
     - 0.659 / 0.627
     - 0.495 / 0.462
     - 0.66 / 0.70
     - 84 / 94
   * - JLF (leave-one-out)
     - 0.659 / 0.613
     - 0.494 / 0.445
     - 0.66 / 0.70
     - 150 / 171
   * - *Published MAPER bar*
     - —
     - *≈ 0.50*
     - —
     - —

.. figure:: ../figures/hammers_three_way.png
   :width: 100%

   nnU-Net has higher overlap and lower surface distance than both
   multi-atlas methods. Its volume error is lower than JLF's but not
   significantly different from MAPER's (:ref:`paired-stats`).
   Sub-millimetre ASD for all methods confirms that most of the remaining
   Dice error is a boundary effect (:ref:`thin-structure-dice`).

.. _paired-stats:

Is the in-domain difference significant?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each brain was held out from nnU-Net training (5-fold) and from the atlas
set (leave-one-out), so the three methods are compared on the same 30 brains.
``code/evaluation/paired_stats.py`` computes, per metric and hemisphere, the
mean paired difference (nnU-Net minus baseline), a percentile-bootstrap 95%
confidence interval over brains (10,000 resamples), and a Wilcoxon
signed-rank test. P-values are Holm-corrected across all 16 comparisons.

.. list-table::
   :header-rows: 1

   * - nnU-Net vs
     - Metric (L / R)
     - Mean difference (L / R)
     - 95% CI (L / R)
     - nnU-Net better (L / R)
     - Holm p (L / R)
   * - MAPER
     - Dice
     - +0.049 / +0.066
     - [0.030, 0.069] / [0.047, 0.085]
     - 23/30 / 28/30
     - 7e-4 / 6e-6
   * - MAPER
     - ASD (mm)
     - −0.09 / −0.14
     - [−0.13, −0.05] / [−0.19, −0.11]
     - 22/30 / 28/30
     - 0.001 / 1e-6
   * - MAPER
     - \|ΔV\| (mm³)
     - −8 / −12
     - [−26, +9] / [−26, +2]
     - 21/30 / 17/30
     - **0.44 / 0.44**
   * - JLF
     - Dice
     - +0.049 / +0.080
     - [0.029, 0.070] / [0.058, 0.103]
     - 21/30 / 25/30
     - 7e-4 / 7e-6
   * - JLF
     - ASD (mm)
     - −0.10 / −0.14
     - [−0.14, −0.05] / [−0.18, −0.10]
     - 20/30 / 26/30
     - 0.003 / 2e-5
   * - JLF
     - \|ΔV\| (mm³)
     - −75 / −89
     - [−111, −35] / [−124, −51]
     - 24/30 / 24/30
     - 0.005 / 0.001

Jaccard gives the same conclusions as Dice. The overlap and boundary gains
are robust: every confidence interval excludes zero and nnU-Net wins on most
brains. The volume-error advantage over MAPER is **not** established. Caveat:
brains in the same fold share a model, so cases are not fully independent;
the bootstrap treats them as independent and may slightly understate the
interval width.

Second site: ADNI n=37 (Parked)
-------------------------------

.. figure:: ../figures/site_gap.png
   :width: 100%

   The same recipe learned Hammers but never learned ADNI.

.. list-table::
   :header-rows: 1

   * - Model
     - ADNI Dice
     - Hammers Dice
   * - Dataset003: Hammers + ADNI
     - ~0.02
     - ~0.68
   * - Dataset004: + harmonisation
     - ~0.04 (misplaced)
     - ~0.68
   * - Dataset005: ADNI only
     - 0.017 / 0.001
     - —
   * - Dataset002 with oracle crop
     - 0 (empty)
     - —
   * - SynthSeg-PC v1, atlas crop
     - 0 (crop missed)
     - 0.467

Clinical target cohort
----------------------

Results on the private clinical cohort are not reported here. Two qualitative
findings shaped the pipeline: the crop location mattered more than the model
(:ref:`fov-geometry`), and inference only worked when brain extraction matched
training (:ref:`intensity-fingerprint`).

Production
----------

Production segmentation uses JLF. No deep-learning model is deployed until one
is validated against manual labels on the target cohort.

Compute
-------

Stage A took about :math:`5 \times 47 \approx 235` GPU-hours on 24 GB MIG
slices. JLF took 11–16 CPU-minutes per brain.
