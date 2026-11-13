The U-Net and nnU-Net
=====================

3D convolution
--------------

A 3D convolutional layer with kernel :math:`w \in \mathbb{R}^{C_\text{out}\times
C_\text{in}\times k^3}` computes

.. math::

   h^{(\ell+1)}_{c}(v) = \sum_{c'=1}^{C_\text{in}} \sum_{u \in \{-\lfloor k/2\rfloor..\lfloor k/2\rfloor\}^3}
   w_{c,c',u}\, h^{(\ell)}_{c'}(s\,v + u) + b_c ,

with stride :math:`s`. Convolution is **translation-equivariant**. Shifting the
input shifts the output, so the same PC detector works anywhere in the
image. The flip side is that equivariance alone carries no notion of "left"
or "right". That information has to come from context the network can see
within its receptive field.

Receptive field
---------------

For a stack of layers with kernel sizes :math:`k_\ell` and strides
:math:`s_\ell`, the receptive field grows as

.. math::

   r_\ell = r_{\ell-1} + (k_\ell - 1)\prod_{m<\ell} s_m, \qquad r_0 = 1.

Each down-sampling stage doubles the effective step. A 5-level U-Net with two
:math:`3^3` convolutions per level therefore sees on the order of 100+ voxels
at the bottleneck. That is enough to include the midline, the temporal horn,
and the amygdala, which are the cues that define PC and its side.

Encoder–decoder with skip connections
-------------------------------------

The U-Net :ref:`[Ronneberger 2015] <ref-ronneberger>` has an **encoder**
that halves resolution and doubles channels at each level, a **decoder** that
upsamples (transposed convolution), and **skip connections** that
concatenate encoder features into the decoder at matching resolution:

.. math::

   d^{(\ell)} = \mathrm{Conv}\Big(\big[\, \mathrm{Up}(d^{(\ell+1)}) \;\Vert\; e^{(\ell)} \big]\Big).

Coarse levels contribute *where* (context, side, neighbouring structures).
Skip connections restore *exactly where* (sub-voxel boundary detail). A thin
structure needs both.

Normalisation and activation (nnU-Net defaults)
-----------------------------------------------

Instance normalisation normalises each channel of each sample over space:

.. math::

   \mathrm{IN}(h)_c = \gamma_c\,\frac{h_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c,

followed by LeakyReLU, :math:`\phi(a)=\max(a, 0.01a)`. Instance
normalisation suits the small batch sizes (2 patches) that 3D volumes
force.

.. _nnunet-heuristics:

How nnU-Net configures itself
-----------------------------

nnU-Net :ref:`[Isensee 2021] <ref-isensee>` replaces manual architecture
design with rules computed from a **dataset fingerprint**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Fingerprint property
     - Derived setting
   * - Median voxel spacing
     - Target spacing for resampling
   * - Median image shape (after resampling)
     - Patch size, as large as fits the GPU memory budget
   * - Patch size
     - Number of pooling stages (until feature maps are ~4 voxels per axis)
   * - Modality = MRI
     - Per-image **z-score** normalisation, :math:`x' = (x-\mu_x)/\sigma_x`

Fixed training rules: SGD with Nesterov momentum 0.99, initial learning rate
:math:`\eta_0 = 10^{-2}`, polynomial decay

.. math::

   \eta_t = \eta_0\left(1 - \frac{t}{T}\right)^{0.9},

1000 epochs of 250 iterations, Dice + cross-entropy loss, and **deep
supervision**. The loss is applied at several decoder resolutions with weights
halving at each coarser level:

.. math::

   \mathcal{L}_\text{DS} = \sum_{i=0}^{D-1} \omega_i\, \mathcal{L}\big(\hat{y}^{(i)}, y^{\downarrow i}\big),
   \qquad \omega_i \propto 2^{-i}, \quad \sum_i\omega_i = 1 .

The 2024 re-benchmark :ref:`[Isensee 2024] <ref-isensee24>` found that a
well-configured U-Net still matches newer architectures on 3D medical
segmentation. With 24 training brains per fold, data and protocol were the
bottleneck, not architecture.

Configuration used here
-----------------------

* ``3d_fullres``: 3D patches at full resolution. PC is a thin 3D surface, so
  2D slices lose through-plane continuity.
* ``nnUNetTrainerNoMirroring``: the default trainer with every mirroring axis
  removed from training augmentation and test-time augmentation
  (:ref:`mirroring-proof`).
* Retraining instead of fine-tuning for new datasets, following the nnU-Net
  guidance that each dataset fingerprint gets its own plan.
