Literature review and design decisions
======================================

Before training anything, I read the piriform, multi-atlas, and
segmentation literature to answer three questions: *what exactly is the
structure*, *what is the bar to beat*, and *which published method fits the
data I actually have*. This page records what each paper contributed and the
decision it led to. Full citations are in :doc:`references`.

.. contents::
   :local:
   :depth: 1

How sources were screened
-------------------------

* **Evidence that can change a decision:** peer-reviewed papers and official
  tool documentation (for example, the nnU-Net repository).
* **Cited for context only:** conference abstracts, results on a different
  label definition, and models without released weights or labels.
* **Same definition or no comparison.** A Dice score is only comparable if
  the label protocol is the same. Numbers on other protocols are cited, not
  used as a target.

1. What is the structure?
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Source
     - What it contributed
     - Decision
   * - :ref:`[Pereira 2005] <ref-pereira>`
     - First MRI volumetry protocol for PC. Excluded frontal PC because its
       borders are unclear on MRI.
     - Protocols differ in extent, so labels from different protocols must
       not be mixed.
   * - :ref:`[Vaughan 2014] <ref-vaughan>`
     - PC is thin paleocortex on both banks of the endorhinal sulcus, highly
       epileptogenic.
     - Expect a thin, low-contrast target (:ref:`thin-structure-dice`).
   * - :ref:`[Galovic 2019] <ref-galovic>`
     - Resecting more than half of PC was associated with ~16× higher odds of
       seizure freedom.
     - The clinical reason to measure PC accurately.
   * - :ref:`[Steinbart 2023] <ref-steinbart>`
     - The only validated protocol for automatic PC volumetry (frontal +
       temporal PC), drawn on the 30 Hammers brains. MAPER vs manual Jaccard
       about 0.50 in controls, 0.40 in TLE, 0.34 in AD. PC atrophy is
       ipsilateral to hippocampal sclerosis.
     - Use **only** Steinbart-protocol labels (Hammers 30, ADNI 37). Left and
       right are separate classes. Beat Jaccard 0.50 on *this* definition.
   * - :ref:`[Leon-Rojas 2021] <ref-leonrojas>`
     - A different PC protocol for surgical planning. Right PC about 18%
       larger and more anterior than left.
     - Cite for asymmetry only; do not train on its labels. Asymmetry is one
       more reason left–right flipping is invalid
       (:ref:`mirroring-proof`).

2. What is the bar to beat?
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Source
     - What it contributed
     - Decision
   * - :ref:`[Heckemann 2006] <ref-heckemann06>`, :ref:`[Rueckert 1999] <ref-rueckert>`
     - Multi-atlas label propagation with vote fusion; the B-spline free-form
       registration underneath it.
     - Understand where the baseline's errors come from
       (:ref:`vote-collapse`).
   * - :ref:`[Heckemann 2010] <ref-heckemann>`
     - MAPER: tissue-class-guided registration, majority vote. The method
       Steinbart used.
     - Reproduce it first as a **calibration** of my scorer (E0 in
       :doc:`experiments`).
   * - :ref:`[Wang 2013] <ref-wang>`, :ref:`[Avants 2011] <ref-avants>`
     - Joint label fusion with correlation-aware weights; ANTs SyN
       registration.
     - Second classical baseline, and the production method.
   * - :ref:`[Cardoso 2015] <ref-cardoso>`
     - GIF, a graph-based multi-atlas method (used by Leon-Rojas).
     - Not used: different atlas and protocol.
   * - :ref:`[Urriola 2025] <ref-urriola>`
     - Conference abstract: MedNeXt-L Dice ~0.90 for PC on 57 scans.
     - **Not a target.** Different labels, abstract only, no labels or
       weights released. Dice 0.90 is not comparable to Steinbart Jaccard
       0.50.

3. Which network?
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Source
     - What it contributed
     - Decision
   * - :ref:`[Ronneberger 2015] <ref-ronneberger>`
     - The U-Net encoder–decoder with skip connections.
     - The architecture family (:doc:`theory/unet`).
   * - :ref:`[Isensee 2021] <ref-isensee>`
     - nnU-Net configures preprocessing, architecture, and training from the
       dataset fingerprint and is a strong baseline across many tasks.
     - With 30 labelled brains, use the **unmodified** nnU-Net recipe
       (``3d_fullres``, 5-fold, 1000 epochs) and change only what the anatomy
       forces: no mirroring. Do not invent an architecture.
   * - :ref:`[Isensee 2024] <ref-isensee24>`
     - Many claimed architecture gains vanish under rigorous validation;
       residual-encoder presets are the recommended next step.
     - ResEnc only *after* a default baseline exists and stalls. Avoid an
       architecture "bake-off" on 30 brains.
   * - :ref:`[Roy 2023] <ref-roy>`
     - MedNeXt, a ConvNeXt-style 3D segmentation network.
     - Deferred. A second architecture is a later comparison, not the first
       experiment.

4. Small target, large image
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Source
     - What it contributed
     - Decision
   * - :ref:`[Lyu 2023] <ref-lyu>`
     - Hippocampus: crop a region around the structure, then segment, in
       native space without warping to a template.
     - Dataset002: 60 mm crops around PC (:ref:`class-imbalance`), and
       prior-crop inference (:doc:`programming/inference`).
   * - :ref:`[Mauri 2025] <ref-mauri>`
     - Claustrum, another thin sheet: field-of-view crop around an atlas
       prior, surface distance and topology metrics, epoch selection on real
       images.
     - Adopt the crop prior, the metrics (:doc:`theory/metrics`), the
       topology fix, and real-image epoch selection.

5. Synthetic training and domain shift
--------------------------------------

The second-site failure (E2–E4 in :doc:`experiments`) sent me back to the
literature on domain shift.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Source
     - What it contributed
     - Decision
   * - :ref:`[Ben-David 2010] <ref-bendavid>`
     - Target error is bounded by source error plus a domain-divergence
       term.
     - Low in-domain error guarantees little on a new site
       (:doc:`theory/domain_shift`).
   * - :ref:`[Guan 2022] <ref-guan>`
     - Survey of domain adaptation for medical imaging.
     - Frame ADNI as a held-out site; adaptation methods only after simpler
       controls fail.
   * - :ref:`[Tobin 2017] <ref-tobin>`, :ref:`[Shorten 2019] <ref-shorten>`, :ref:`[Pérez-García 2021] <ref-torchio>`
     - Domain randomisation; image augmentation practice and tooling.
     - Offline intensity augmentation (Datasets 006 and 007,
       :doc:`theory/augmentation`). Called "domain-randomised", not
       "multi-site".
   * - :ref:`[Billot 2023] <ref-billot>`
     - SynthSeg: train only on images synthesised from label maps with random
       contrast and resolution.
     - SynthSeg-PC (:doc:`theory/synthseg`), with flipping disabled.
   * - :ref:`[Mauri 2025] <ref-mauri>`
     - SynthSeg-style training for a thin structure, with left hemispheres
       flipped to the right.
     - Take the training scheme; **leave** the flipping (PC laterality
       matters) and the 0.35 mm working resolution, which would mean
       upsampling 1 mm labels and inventing boundary voxels.
   * - :ref:`[Iglesias 2023] <ref-iglesias>`
     - SynthSR turns heterogeneous clinical scans into 1 mm T1-like images.
     - Optional preprocessing for poor clinical T1s; implemented as an option,
       not yet used for results.

What I deliberately did not do
------------------------------

* **Train on labels from other protocols** (Leon-Rojas, Urriola). Mixing
  definitions would make every score uninterpretable.
* **Compare to Dice 0.90 from an abstract.** The comparison target is
  Steinbart's MAPER agreement on the same protocol.
* **Start with a custom or newer architecture.** The nnU-Net papers argue
  that a well-configured baseline beats most custom designs, and rigorous
  comparisons on 30 brains are underpowered (:doc:`theory/metrics`).
* **Copy the claustrum recipe wholesale.** Mauri et al. needed synthetic
  images because their labels had no matched in-vivo T1. I had matched T1 and
  labels, so SynthSeg-PC came *after* supervised training failed to
  transfer, as a targeted response to that failure.
