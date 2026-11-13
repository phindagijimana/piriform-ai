Domain shift
============

Definitions
-----------

The **source** domain :math:`S` (Hammers) and **target** domain :math:`T`
(ADNI, a clinical cohort) are joint distributions over images and labels. Under
**covariate shift**,

.. math::

   p_S(x) \neq p_T(x), \qquad p_S(y\mid x) = p_T(y\mid x).

The anatomy-to-label rule is the same, but scanners, protocols, and
preprocessing make images look different. Empirical risk minimisation (Eq.
:eq:`erm`) only controls the **source** risk :math:`R_S(f)`. A classical bound
:ref:`[Ben-David 2010] <ref-bendavid>` makes the gap explicit:

.. math::

   R_T(f) \;\le\; R_S(f) \;+\; \tfrac12\, d_{\mathcal{H}\Delta\mathcal{H}}(S, T) \;+\; \lambda^\star ,

where :math:`d_{\mathcal{H}\Delta\mathcal{H}}` measures how distinguishable the
two input distributions are to the model class, and :math:`\lambda^\star` is the
error of the best joint hypothesis. A low source error guarantees little when
the domains are easy to tell apart, which is the case when one site's
intensities are 12× another's. :ref:`[Guan 2022] <ref-guan>` surveys methods
that reduce this term.

Three ways to reduce the divergence term:

1. **Make the target look like the source**: matched preprocessing and
   intensity harmonisation.
2. **Make the source cover the target**: multi-site training, augmentation,
   and synthetic training (:doc:`synthseg`).
3. **Make the input comparable**: put the structure at the same place in
   the field of view, which is the crop prior below.

.. _intensity-fingerprint:

Intensity fingerprint
---------------------

nnU-Net z-scores each image with its own mean and standard deviation. That
removes global gain, but not differences in the *shape* of the intensity
distribution: skull-stripping edges, bias residue, and tissue contrast. Two
measured effects:

* **Raw intensity scale.** The ADNI 99th percentile was about **12×** the
  Hammers value.
* **Skull-strip mismatch.** The training brains were extracted with ANTs.
  Inference on the target cohort's FastSurfer mask gave near-empty masks.
  Re-extracting the same scan with the training pipeline gave
  normal-sized masks.

Percentile harmonisation (Dataset004) linearly rescales the brain voxels of
each target image so that its 50th and 99th percentiles match the source.
:math:`p^S` is the median, across the 30 Hammers brains, of each brain's
percentile:

.. math::

   x' = p^{S}_{50} + \big(x - p^{T}_{50}\big)\,
        \frac{p^{S}_{99} - p^{S}_{50}}{p^{T}_{99} - p^{T}_{50}} .

This made ADNI predictions non-empty but **misplaced** (Dice ~0.04).
Intensity scale was one cause, but not the only one.

.. _fov-geometry:

Field-of-view geometry
----------------------

A crop model sees a cube of side :math:`L = 60` mm centred at the prior
:math:`c`. Let PC have centroid :math:`\mu` and half-extent :math:`r` along an
axis. The whole structure is inside the crop only if

.. math::

   \|c - \mu\|_\infty + r \;\le\; \frac{L}{2} = 30\ \text{mm}.

A centre farther than :math:`30 + r` mm away contains **no** PC at all.

* The **putamen** centroid is about **45–50 mm** from PC. With a putamen prior,
  0% of PC fell inside the crop, and no model could pass.
* Affine atlas registration (``atlas_pc``) placed the centre within the gate
  on the target cohort. On ADNI, some registrations failed by 57–120 mm.

The **field-of-view gate** (:doc:`../programming/inference`) turns this
inequality into a fail-closed check: prior-to-reference distance
:math:`\le 25` mm, and at least 50% of reference PC voxels inside the crop.

Controls as hypothesis tests
----------------------------

Every ADNI experiment was designed to rule out one explanation:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Hypothesis
     - Test
     - Result
   * - The labels are mis-paired with images
     - Local overlays of manual labels on T1s
     - Rejected. Labels sit on PC
   * - Intensity scale alone
     - Dataset004, percentile harmonisation
     - Rejected. Non-empty but misplaced
   * - Hammers dominates the mixed training set
     - Dataset005, ADNI only
     - Rejected. ADNI is not learnable with this pipeline (best log Dice 0.074)
   * - Localisation only
     - Oracle crop centred on the manual label
     - Rejected. Prediction still empty

What remains is an image-domain difference that neither harmonisation nor
crop placement removes. That is the scenario synthetic-image training
targets (:doc:`synthseg`).

Decision rule
-------------

Pass and fail bands were fixed before each check:

.. math::

   \text{band}(\bar D) =
   \begin{cases}
     \text{pass} & \bar D \ge 0.40 \\
     \text{fail} & \bar D \le 0.10 \\
     \text{inconclusive} & \text{otherwise}
   \end{cases}

where :math:`\bar D` is the mean per-hemisphere Dice. ADNI was in the fail
band under every variant, and the branch was parked on 2026-10-21.
