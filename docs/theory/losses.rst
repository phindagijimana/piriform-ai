Loss functions
==============

Cross-entropy and its imbalance problem
---------------------------------------

The voxel-wise cross-entropy, averaged over :math:`|\Omega|` voxels, is

.. math::

   \mathcal{L}_\text{CE} = -\frac{1}{|\Omega|}\sum_{v\in\Omega}\sum_{k} g_{v,k}\log p_{v,k},

where :math:`g_{v,k}\in\{0,1\}` is the one-hot target. Its gradient with
respect to the logits is the familiar

.. math::

   \frac{\partial \mathcal{L}_\text{CE}}{\partial z_{v,k}} = \frac{1}{|\Omega|}\,(p_{v,k} - g_{v,k}).

Every voxel contributes equally. With a foreground fraction of
:math:`\rho\approx10^{-3}\text{–}10^{-5}` (:ref:`class-imbalance`), a network
that predicts "background everywhere" already reaches a loss close to the
optimum, and the foreground gradient signal is diluted by a factor
:math:`\rho`.

Soft Dice
---------

For class :math:`k`, define soft intersection and sizes

.. math::

   I = \sum_v p_{v} g_{v}, \qquad P = \sum_v p_v, \qquad G = \sum_v g_v ,

and the soft Dice coefficient and loss

.. math::
   :label: softdice

   D = \frac{2I + \epsilon}{P + G + \epsilon}, \qquad \mathcal{L}_\text{Dice} = 1 - D .

Its gradient with respect to a voxel probability is

.. math::

   \frac{\partial D}{\partial p_v} = \frac{2 g_v}{P+G} - \frac{2I}{(P+G)^2}
   = \frac{2\big(g_v(P+G) - I\big)}{(P+G)^2}.

The denominator scales with the **size of the structure** (:math:`P+G`), not
with :math:`|\Omega|`. Background voxels far from PC (:math:`g_v=0`,
:math:`p_v\approx0`) contribute nothing to :math:`P`, :math:`G`, or :math:`I`.
The Dice loss is therefore insensitive to how much background the patch
contains, which is the property a 0.5 cm³ target needs.

Combined loss
-------------

nnU-Net optimises

.. math::

   \mathcal{L} = \mathcal{L}_\text{CE} + \mathcal{L}_\text{Dice},

so that cross-entropy provides well-behaved, calibrated gradients early in
training, while Dice keeps the foreground from being swamped. This is wrapped
in deep supervision (:ref:`nnunet-heuristics`).

SynthSeg's two-phase loss
-------------------------

SynthSeg trains in two phases :ref:`[Billot 2023] <ref-billot>`:

1. **Warm-up** (``wl2_epochs``): a weighted L2 loss on the pre-softmax
   activations, which stabilises the first steps when synthetic contrasts are
   extreme.
2. **Dice phase** (``dice_epochs``): soft Dice, Eq. :eq:`softdice`, summed
   over all segmentation labels. One checkpoint ``dice_XXX.h5`` is saved per
   epoch.

When training resumes from a ``dice_XXX.h5`` checkpoint, the warm-up must
**not** run again. In the first resume attempt, the log showed "Resuming from
dice_016" followed by "Epoch 1/1" of the L2 phase: the trained Dice-phase
weights were sent back through warm-up with a different objective. The
training worker now forces ``WL2_EPOCHS=0`` whenever the checkpoint is a
Dice checkpoint (:doc:`../programming/synthseg_training`).
