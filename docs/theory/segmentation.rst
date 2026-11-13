Segmentation as a learning problem
==================================

Formulation
-----------

A T1-weighted MRI is a function on a discrete 3D grid

.. math::

   x : \Omega \rightarrow \mathbb{R}, \qquad \Omega \subset \mathbb{Z}^3,

with a voxel-to-world affine :math:`A \in \mathbb{R}^{4\times4}` that maps voxel
indices :math:`(i,j,k)` to scanner coordinates (RAS, millimetres). A
segmentation is a label map :math:`y : \Omega \rightarrow \{0, 1, 2\}`
(background, left PC, right PC).

A segmentation network :math:`f_\theta` outputs, for each voxel :math:`v`, a
vector of logits :math:`z_v \in \mathbb{R}^K`, turned into class probabilities
by the softmax

.. math::

   p_{v,k} = \frac{e^{z_{v,k}}}{\sum_{j=1}^{K} e^{z_{v,j}}}, \qquad
   \hat{y}(v) = \arg\max_k p_{v,k}.

Training minimises an empirical risk over the :math:`N` labelled images,

.. math::
   :label: erm

   \theta^\star = \arg\min_\theta \frac{1}{N}\sum_{n=1}^{N}
   \mathcal{L}\big(f_\theta(x_n), y_n\big),

where :math:`\mathcal{L}` is the Dice + cross-entropy loss of
:doc:`losses`. With :math:`N = 24` training brains per fold, the main risks are
over-fitting and failing to generalise beyond the source scanner
(:doc:`domain_shift`).

.. _class-imbalance:

Class imbalance
---------------

Let :math:`\rho = |\mathrm{PC}| / |\Omega|` be the foreground fraction.
Hammers voxels are about :math:`0.94^3 \approx 0.83` mm³, and the mean PC
volume is 486 mm³ (left), so one hemisphere's PC has about 590 voxels.

.. list-table::
   :header-rows: 1

   * - Field of view
     - Volume
     - :math:`\rho` (one hemisphere)
   * - Full head, :math:`256^3` voxels
     - :math:`\approx 1.4\times10^{7}` mm³
     - :math:`\approx 3.5\times10^{-5}`
   * - 60 mm crop
     - :math:`2.16\times10^{5}` mm³
     - :math:`\approx 2.2\times10^{-3}`

Cropping raises :math:`\rho` by about :math:`60\times`. That is the motivation
for the crop datasets (002, 007) and the 60 mm field of view used by Mauri et
al. :ref:`[Mauri 2025] <ref-mauri>`.

.. _thin-structure-dice:

Why thin structures have a low Dice ceiling
-------------------------------------------

Model PC locally as a slab of thickness :math:`t` and area :math:`S`. Suppose
a prediction has the right shape but is displaced by :math:`\delta` along the
slab normal. Then :math:`|A \cap B| = S(t-\delta)` and
:math:`|A| = |B| = St`, so

.. math::
   :label: slab

   \mathrm{Dice} = \frac{2S(t-\delta)}{2St} = 1 - \frac{\delta}{t}
   \qquad (0 \le \delta \le t).

For PC, :math:`t \approx 2\text{–}3` mm. A one-voxel shift
(:math:`\delta \approx 0.94` mm) alone lowers Dice to about
**0.6–0.7**. The same shift on a 10 mm-thick structure costs less than 10%.
Eq. :eq:`slab` explains three observations in this project:

* State-of-the-art PC agreement sits around Dice 0.65–0.70 (Jaccard ≈ 0.50),
  which is much lower than for hippocampus or putamen.
* **Surface distance** (ASD) is a necessary companion metric. A prediction
  with Dice 0.6 can still be within 1 mm of the true boundary everywhere
  (:doc:`metrics`).
* Multi-atlas **majority voting collapses** on PC, because the same
  sub-millimetre misregistration erodes the overlap between atlases
  (:ref:`vote-collapse`).

Bayes-optimal prediction and laterality
---------------------------------------

Under a pixel-wise loss, the optimal predictor outputs the posterior
:math:`p(y_v = k \mid x)`. Left and right PC look the same locally, so the
only evidence that separates class 1 from class 2 is **position** relative to
the midline. Any training procedure that destroys positional information
about laterality makes the two classes indistinguishable. This is why
mirroring is excluded (:ref:`mirroring-proof`), and why SynthSeg-PC gives
both PCs the **same** intensity distribution (:ref:`generation-classes`).
