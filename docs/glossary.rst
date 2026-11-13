Glossary
========

.. glossary::

   ASD
      Average symmetric surface distance, in mm (:doc:`theory/metrics`).

   Atlas
      A brain image together with its expert labels.

   Checkpoint chaining
      Successive Slurm jobs that each resume from the previous job's checkpoint.

   cuDNN
      NVIDIA's deep-learning kernel library. Disabled here because it had no
      3D convolution engine on MIG slices.

   Dice / Jaccard
      Overlap scores between prediction and reference. :math:`J = D/(2-D)`.

   Domain shift
      A difference between training and deployment input distributions
      (:doc:`theory/domain_shift`).

   FoV
      Field of view: the box of image the model sees.

   GMM
      Gaussian mixture model. SynthSeg draws a random mean and variance per
      tissue class to synthesise images.

   JLF
      Joint label fusion. Multi-atlas fusion with correlation-aware local
      weights. Production method here.

   MAPER
      Multi-atlas propagation with enhanced registration, using majority-vote
      fusion. The published PC benchmark.

   MIG
      Multi-Instance GPU. One physical GPU partitioned into isolated slices.

   Mirroring
      Left–right flip augmentation. Invalid when left and right are separate
      classes.

   nnU-Net
      Self-configuring U-Net framework for medical image segmentation.

   Oracle
      A prior that uses information unavailable at deployment, used as an
      upper bound only.

   PC
      Piriform cortex.

   Prior
      The source of the crop centre (atlas, putamen, JLF, oracle).

   SynthSeg
      Training on synthetic images generated from label maps with random
      contrast.

   Walltime
      Maximum runtime of a cluster job (12 h here).
