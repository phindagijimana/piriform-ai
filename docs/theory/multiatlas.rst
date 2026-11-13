Multi-atlas segmentation
========================

Multi-atlas methods are the classical alternative to deep learning. They
have no trained weights. They transfer labels from :math:`N` expert-labelled
**atlases** :math:`\{(A_i, L_i)\}_{i=1}^N` to a new target image :math:`T` by
registration, then **fuse** the transferred labels.

Registration
------------

For each atlas, find a spatial transform :math:`\phi_i` that aligns the
atlas to the target:

.. math::

   \phi_i = \arg\min_{\phi}\; \mathcal{S}\big(T,\, A_i\circ\phi\big) + \lambda\,\mathcal{R}(\phi),

where :math:`\mathcal{S}` is an image dissimilarity (e.g. negative normalised
cross-correlation or mutual information) and :math:`\mathcal{R}` penalises
non-smooth deformations. The propagated label map is
:math:`\tilde L_i = L_i\circ\phi_i`.

* **MAPER** :ref:`[Heckemann 2010] <ref-heckemann>` uses MIRTK free-form
  deformations and adds tissue-class (GM/WM/CSF) information to the
  similarity term. This makes registration robust to enlarged ventricles.
* **ANTs SyN** :ref:`[Avants 2011] <ref-avants>` uses symmetric
  diffeomorphic registration with cross-correlation. It is the registration
  inside the JLF pipeline.

Majority voting (MAPER)
-----------------------

.. math::

   \hat y(v) = \arg\max_{k}\; \sum_{i=1}^{N} \mathbb{1}\big[\tilde L_i(v) = k\big].

For a binary decision (PC vs background), a voxel is labelled PC only if
**more than half** the atlases agree.

.. _vote-collapse:

Vote collapse on thin structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`q_v` be the probability that a warped atlas covers voxel
:math:`v` with PC. If atlas errors were independent, the vote count would be
:math:`n_v \sim \mathrm{Binomial}(N, q_v)`, and PC survives only where
:math:`q_v > \tfrac12`. By the slab argument of Eq. :eq:`slab`, a residual
misregistration :math:`\delta` on a structure of thickness :math:`t` gives
:math:`q_v \lesssim 1-\delta/t`. When :math:`\delta` approaches :math:`t/2`
(about 1–1.5 mm for PC), :math:`q_v` drops below one half almost everywhere.

This happened in practice: every warped atlas contained PC, but at no voxel
did more than a small minority agree, far short of the 16/30 majority vote,
so the fused output was **empty**. The pipeline now fails loudly when fused
labels are missing.

Joint label fusion (JLF)
------------------------

JLF :ref:`[Wang 2013] <ref-wang>` replaces equal votes with spatially varying
weights:

.. math::

   \hat y(v) = \arg\max_{k}\; \sum_{i=1}^{N} w_i(v)\, \mathbb{1}\big[\tilde L_i(v) = k\big],
   \qquad \sum_i w_i(v) = 1 .

Let :math:`\delta_i(v)\in\{0,1\}` indicate that atlas :math:`i` is wrong at
:math:`v`. JLF chooses :math:`w` to minimise the expected squared labelling
error, which depends on **pairwise** error correlations:

.. math::

   \mathbb{E}\Big[\big(\textstyle\sum_i w_i\delta_i\big)^2\Big] = w^\top M_v\, w,
   \qquad M_v(i,j) = p\big(\delta_i(v)\,\delta_j(v) = 1 \mid T, A_i, A_j\big).

Minimising :math:`w^\top M_v w` subject to :math:`\mathbf{1}^\top w = 1`
(Lagrange multipliers) gives the closed form

.. math::
   :label: jlf

   w_v = \frac{M_v^{-1}\mathbf{1}}{\mathbf{1}^\top M_v^{-1}\mathbf{1}} .

:math:`M_v` is estimated from image patches :math:`\mathcal{N}(v)` around
:math:`v`, because atlases whose intensities differ from the target in the
same places tend to make the same errors:

.. math::

   M_v(i,j) \approx \Big[\sum_{u\in\mathcal{N}(v)} |A_i(u) - T(u)|\,|A_j(u) - T(u)|\Big]^{\beta},

with a small ridge term :math:`\alpha I` added for invertibility. Two
properties matter for PC:

* **Local weighting.** Atlases that match the target well *near PC* dominate,
  even if they match poorly elsewhere.
* **Correlation-aware.** Ten atlases that make the same mistake count as
  roughly one vote, not ten. That is why JLF degrades gracefully where majority
  voting collapses.

JLF is the **production** method for the target cohort.

Cost
----

Both methods need :math:`N` non-rigid registrations per target. That is
**CPU-bound**, with no GPU path in these tools. JLF took about 11–16 minutes
per brain here. The network, by contrast, runs one forward pass, paying the
cost once at training time.
