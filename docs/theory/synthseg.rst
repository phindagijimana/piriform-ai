SynthSeg: learning from synthetic images
========================================

Idea
----

A supervised model learns :math:`p(y\mid x)` under the source image
distribution :math:`p_S(x)`, and nothing forces it to behave on a different
scanner's :math:`p_T(x)` (:doc:`domain_shift`). SynthSeg
:ref:`[Billot 2023] <ref-billot>` removes the dependence on any particular
:math:`p(x)` by training on images **generated from label maps**, with a new
random appearance at every step. The network never sees a real image during
training.

Generative model
----------------

Given a training label map :math:`L : \Omega \to \{0,\dots,C\}`, one training
pair :math:`(x, y)` is produced as follows. The code values in brackets are
the SynthSeg defaults, then the v1 / v2 values used here.

**1. Spatial deformation.** :math:`\phi = \phi_\text{aff}\circ\phi_\text{nl}`.

* Affine: per-axis scaling :math:`s\sim\mathcal{U}(1-a, 1+a)` with
  :math:`a` = 0.2 / 0.3; rotations :math:`\sim\mathcal{U}(-\theta,\theta)`
  with :math:`\theta` = 15° / 20°; shearing bounds 0.012 / 0.02.
* Non-linear: a stationary velocity field :math:`v` is sampled on a coarse
  grid (4% of the image size), with voxel-wise
  :math:`v \sim \mathcal{N}(0, \sigma^2)` and
  :math:`\sigma \sim \mathcal{U}(0, \sigma_\text{nl})`, where
  :math:`\sigma_\text{nl}` = 4 / 5. The velocity field is upsampled and
  integrated (scaling and squaring) into a diffeomorphism
  :math:`\phi_\text{nl} = \exp(v)`.

The deformed label map is :math:`L' = L\circ\phi^{-1}` (nearest neighbour).

**2. Gaussian mixture intensities.** Each generation class :math:`c` gets
fresh parameters

.. math::

   \mu_c \sim \mathcal{U}(25, 225), \qquad \sigma_c \sim \mathcal{U}(5, 25),

and every voxel is sampled independently:

.. math::
   :label: gmm

   x(v) \sim \mathcal{N}\big(\mu_{c(L'(v))},\ \sigma^2_{c(L'(v))}\big).

The contrast between any two tissues is therefore random. PC might be
brighter than white matter in one sample and darker in the next.

**3. Bias field.** Sample :math:`B_0 \sim \mathcal{N}(0, \sigma_B^2)` on a grid
at 2.5% of the image size (:math:`\sigma_B` = 0.7 / 1.0), upsample it to
:math:`B`, and multiply:

.. math::

   x \leftarrow x \cdot \exp(B).

**4. Resolution.** With ``randomise_res=True``, sample an acquisition
resolution (up to 4 mm isotropic or 8 mm along one axis), blur with a
Gaussian matched to that resolution, downsample, and resample back to the
training grid.

**5. Intensity normalisation and gamma.** Clip, rescale to :math:`[0,1]`,
and apply :math:`x \leftarrow x^{\exp(\gamma)}`, with
:math:`\gamma\sim\mathcal{N}(0, 0.5^2)`.

The target is :math:`y = L'` restricted to the segmentation labels.

.. _generation-classes:

Generation classes and laterality
---------------------------------

``generation_classes`` decides which labels **share** GMM parameters in
Eq. :eq:`gmm`. For the PC field of view (:doc:`../programming/synthseg_training`):

.. list-table::
   :header-rows: 1

   * - Label
     - Meaning
     - Generation class
   * - 0
     - Background
     - 0
   * - 1, 2
     - PC left, PC right
     - **1** (shared)
   * - 3, 4
     - Amygdala L/R
     - 2 (shared)
   * - 5, 6
     - Medial temporal lobe L/R
     - 3 (shared)
   * - 7
     - All other tissue
     - 4

Sharing a class between left and right is deliberate. If left and right PC
were drawn with independent intensities, the network could learn the
shortcut "the brighter PC is the left one", which is meaningless on real
scans. With a shared class, the **only** cue that separates labels 1 and 2 is
spatial context, which is the cue that generalises. The same logic forbids
flipping (:ref:`mirroring-proof`).

Network and optimisation
------------------------

* 5-level 3D U-Net, two :math:`3^3` convolutions per level, ELU activations,
  base width 24 (v1) or 32 (v2), doubling per level.
* Adam, learning rate :math:`10^{-4}`, batch size 1, 1000 steps per epoch,
  output patch :math:`160^3`.
* Loss: L2 warm-up, then soft Dice (:doc:`losses`).

.. _epoch-selection:

Epoch selection on real images
------------------------------

Synthetic-image Dice measures how well the network inverts the *generator*,
not how well it segments real scans. Following Mauri et al.
:ref:`[Mauri 2025] <ref-mauri>`, every saved checkpoint :math:`e` is run on
the **real** held-out Hammers T1s, and the selected epoch is

.. math::

   e^\star = \arg\max_{e}\; \frac{1}{|\mathcal{H}|}\sum_{h\in\mathcal{H}} \mathrm{Dice}_h(e),

where :math:`\mathcal{H}` is the set of scored hemispheres. Each 60 mm crop
contains one side, so each crop contributes one hemisphere score. This uses
validation data for model selection, so the in-domain number at :math:`e^\star`
is mildly optimistic. The cross-site numbers are unaffected.

Recipes in this project
-----------------------

.. list-table::
   :header-rows: 1

   * - Recipe
     - U-Net width
     - Scaling / rotation / shear
     - :math:`\sigma_\text{nl}`
     - :math:`\sigma_B`
     - Dice epochs
     - Labels
   * - v1
     - 24
     - 0.2 / 15° / 0.012
     - 4.0
     - 0.7
     - 50
     - 60 mm PC box
   * - v2
     - 32
     - 0.3 / 20° / 0.02
     - 5.0
     - 1.0
     - 100
     - 60 mm PC box
   * - wb
     - 32
     - 0.3 / 20° / 0.02
     - 5.0
     - 1.0
     - 100
     - Whole brain (output 192³)

A wider network cannot load a narrower checkpoint, so v2 trains from scratch
in a new model directory.

Relation to Mauri et al. (claustrum)
------------------------------------

The claustrum is another thin sheet of grey matter, and Mauri et al.
reached a cross-validated Dice of 0.632 with this recipe. The differences
here: *in-vivo* 1 mm T1 training labels instead of ultra-high-resolution
ex-vivo labels, **no flipping** (they flip all hemispheres to one side),
and a PC-specific crop prior.
