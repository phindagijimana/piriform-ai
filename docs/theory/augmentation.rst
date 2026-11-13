Data augmentation
=================

Augmentation as a transformation group
--------------------------------------

Data augmentation draws a random transformation :math:`T` and trains on the
pair :math:`(T_x x,\; T_y y)`. For a *spatial* transform :math:`\phi`
(rotation, scaling, elastic deformation), image and labels are resampled with
the same map, :math:`T_x x = x\circ\phi^{-1}` and :math:`T_y y = y\circ\phi^{-1}`,
using linear interpolation for :math:`x` and nearest neighbour for :math:`y`.
For a *photometric* transform :math:`\psi` (gamma, bias, noise), only the
image changes: :math:`T_x x = \psi(x)` and :math:`T_y y = y`.

An augmentation is **valid** if the augmented pair could plausibly have come
from the true data distribution. Equivalently, the labelling function
commutes with the transform:

.. math::

   \text{label}(T_x x) = T_y\,\text{label}(x).

.. _mirroring-proof:

Why left–right mirroring is invalid here
----------------------------------------

Let :math:`M` reflect the volume across the mid-sagittal plane. The
default nnU-Net and SynthSeg pipelines apply :math:`M` to **both** the
image and the label map with probability :math:`\tfrac12`, keeping label
*values* unchanged.

Take a voxel :math:`v` in the right PC. In the original data it carries
label 2. After mirroring, the anatomy of the *left* PC lands at :math:`v`,
and it carries label 1. Because the left and right PC look alike, the
network receives (approximately) the same input with label 2 half the time
and label 1 the other half. Under a pixel-wise loss, the Bayes-optimal output
is then

.. math::

   p(y_v = 1 \mid x) = p(y_v = 2 \mid x) = \tfrac12 \cdot p(y_v \in \{1,2\} \mid x),

so the model **cannot learn laterality**, and :math:`\arg\max` between 1 and 2
becomes arbitrary. Mirroring is valid only if the label permutation
:math:`\sigma: 1 \leftrightarrow 2` is applied together with :math:`M`, that
is :math:`T_y y = \sigma \circ y \circ M`. That variant also assumes the left
and right PC are mirror-symmetric, which is the very thing epilepsy studies
want to measure. This project therefore **disables mirroring entirely**:

* nnU-Net: ``nnUNetTrainerNoMirroring`` (training *and* test-time augmentation).
* SynthSeg-PC: ``flipping=False``, and the training script refuses to start
  if the label metadata says otherwise (:doc:`../programming/synthseg_training`).

Photometric augmentation (offline, Datasets 006 and 007)
--------------------------------------------------------

For Datasets 006 and 007, extra training images were generated **offline**
from each real T1. The labels are untouched. Inside the brain mask
:math:`\mathcal{B}`, the corruption ``corrupt()`` applies, in order:

**Gamma.** With :math:`l, h` the 1st and 99th percentiles and
:math:`\gamma \sim \mathcal{U}(0.6, 1.6)`:

.. math::

   x \leftarrow l + (h-l)\left(\mathrm{clip}\!\left(\frac{x-l}{h-l},0,1\right)\right)^{\gamma}.

**Multiplicative bias field** (probability 0.9). Draw
:math:`z \sim \mathcal{N}(0,1)` on a grid 16× coarser than the image,
upsample it trilinearly to :math:`\tilde z`, and set

.. math::

   b = 1 + 0.25\,\frac{\tilde z - \overline{\tilde z}}{\mathrm{sd}(\tilde z)}, \qquad x \leftarrow x\cdot b .

This mimics the smooth receive-coil inhomogeneity of MRI.

**Gaussian noise** (probability 0.85): :math:`x \leftarrow x + \epsilon`, with
:math:`\epsilon \sim \mathcal{N}(0, \sigma^2)` and :math:`\sigma = 0.02\,P_{99}(x)`.

**Resolution loss** (probability 0.5). Downsample by
:math:`s\sim\mathcal{U}(0.4, 0.75)` with linear interpolation, then upsample
back. This mimics thicker slices and lower in-plane resolution.

Dataset007 widens the ranges (:math:`\gamma\sim\mathcal{U}(0.5,1.8)`,
:math:`s\sim\mathcal{U}(0.35,0.7)`) and makes 5 variants per crop instead of 3.

.. note::

   Validation folds contain **only original images**, and every augmented
   copy of a brain stays in the same fold as its original. Otherwise a
   corrupted copy of a validation brain would leak into training.

Offline augmentation vs on-the-fly synthesis
--------------------------------------------

Offline augmentation multiplies the dataset by a fixed factor (×4 or ×6).
The network still sees the *same* underlying anatomy and the same family of
real contrasts. SynthSeg :doc:`synthseg` instead draws a **new random contrast
at every training step** from a generative model, which is a much larger
input distribution. The trade-off is an image appearance that is often
unrealistic. Domain randomisation :ref:`[Tobin 2017] <ref-tobin>` argues
that this is a feature: if training appearance varies enough, the real target
domain becomes just one more variation.
