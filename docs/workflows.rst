Processing workflows
====================

This page describes each processing pipeline as a sequence of nodes: what
goes in, what each step does, and what comes out. The theory behind each
step is linked from the node descriptions.

.. contents::
   :local:
   :depth: 1

Preprocessing
-------------

.. graphviz::

   digraph prep {
     rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=11];
     t1 [label="Raw T1", shape=note, fillcolor="#ffffff"];
     n4 [label="Bias correction\n(N4 / N3)"];
     bet [label="ANTs brain extraction\nOASIS-30 template"];
     out [label="Brain-extracted T1", shape=note, fillcolor="#ffffff"];
     t1 -> n4 -> bet -> out;
   }

Bias correction uses N3 :ref:`[Sled 1998] <ref-sled>` (Hammers and ADNI
releases ship N3-corrected) or N4 :ref:`[Tustison 2010] <ref-tustison10>`.
The training images were brain-extracted with the ANTs pipeline
:ref:`[Tustison 2014] <ref-tustison14>` and the OASIS-30 template
:ref:`[Klein 2012] <ref-klein>`, obtained from TemplateFlow
:ref:`[Ciric 2022] <ref-ciric>`. **Inference must use the same extraction.**
Running the model on a different brain mask (FastSurfer
:ref:`[Henschel 2020] <ref-henschel>`) gave near-empty predictions. The theory is in
:ref:`intensity-fingerprint`.

Workflow A: supervised nnU-Net (Datasets 001–007)
-------------------------------------------------

.. graphviz::

   digraph nnunet {
     rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=11];
     raw [label="imagesTr / labelsTr\n(T1, PC labels 1/2)", shape=note, fillcolor="#ffffff"];
     split [label="make_nnunet_splits.py\nlocked 5-fold (24/6)"];
     plan [label="nnUNetv2_plan_and_preprocess\nfingerprint → plans"];
     train [label="nnunet_entry.py train\n3d_fullres · NoMirroring · cuDNN off"];
     hop [label="12 h Slurm hop\n--c resume", shape=ellipse, fillcolor="#fff4d6"];
     ckpt [label="checkpoint_final.pth\n(per fold)", shape=note, fillcolor="#ffffff"];
     pred [label="5-fold CV predictions"];
     score [label="score_hammers_stage_a.py\nDice · Jaccard · ASD · |ΔV|"];
     raw -> split -> plan -> train -> hop -> train; train -> ckpt -> pred -> score;
   }

*Nodes:*

``make_nnunet_splits``
   Writes ``splits_final.json``. Splits are **locked**, so every dataset
   variant uses the same held-out brains and comparisons are paired. For
   crop datasets, both hemispheres of a brain stay in the same fold to avoid
   leakage (:doc:`programming/dataset_preparation`).

``plan_and_preprocess``
   nnU-Net reads the dataset fingerprint (spacing, size, intensity) and
   derives the patch size, network depth, and normalisation
   (:ref:`nnunet-heuristics`).

``train``
   1000 epochs × 250 iterations of SGD on the Dice + cross-entropy loss
   (:doc:`theory/losses`) with mirroring disabled (:ref:`mirroring-proof`).

``hop``
   Slurm chaining with ``afterany`` dependencies (:doc:`programming/nnunet_training`).

Workflow B: SynthSeg-PC
-----------------------

.. graphviz::

   digraph synthseg {
     rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#fff4d6", fontname="Helvetica", fontsize=11];
     lab [label="Hammers dense label maps\n(PC, amygdala, MTL, other)", shape=note, fillcolor="#ffffff"];
     prep [label="prepare_labels.py\n60 mm box · generation classes"];
     gen [label="Generator (every step)\ndeform → GMM → bias → blur/downsample"];
     net [label="5-level U-Net\nsoft Dice"];
     h5 [label="dice_XXX.h5\n(every epoch)", shape=note, fillcolor="#ffffff"];
     val [label="validate.py\npredict REAL Hammers T1s"];
     pick [label="epoch_pick.json\nargmax hemi-Dice", shape=note, fillcolor="#ffffff"];
     lab -> prep -> gen -> net -> h5 -> val -> pick;
   }

The network never sees a real image during training. The generator is
specified in :doc:`theory/synthseg`, and the epoch-selection rule in
:ref:`epoch-selection`.

Workflow C: prior-crop inference and QC
---------------------------------------

.. graphviz::

   digraph infer {
     rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#e3f5e1", fontname="Helvetica", fontsize=11];
     t1 [label="Subject T1\n(matched brain extraction)", shape=note, fillcolor="#ffffff"];
     reg [label="warp_atlas_pc_prior.sh\natlas a01 → T1 (affine)"];
     crop [label="make_pc_crop.py\n60 mm cube per hemisphere"];
     model [label="Model prediction\n(nnU-Net or SynthSeg-PC)"];
     topo [label="Topology fix\nlargest component + fill holes"];
     stitch [label="make_pc_crop.py --stitch\nback to native grid"];
     qc [label="qc_priorcrop.py\nFoV gate (fail closed)", fillcolor="#fde2e2"];
     out [label="PC mask + volumes", shape=note, fillcolor="#ffffff"];
     t1 -> reg -> crop -> model -> topo -> stitch -> qc -> out;
   }

The crop centre is the **prior**. Four modes are implemented:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Mode
     - Centre
     - Use
   * - ``atlas_pc``
     - Centroid of one Hammers atlas PC after affine registration
     - **Deployable.** Independent of the reference
   * - ``jlf``
     - Centroid of the subject's own JLF output
     - Upper bound only (circular)
   * - ``oracle``
     - Centroid of the subject's manual label
     - Ablation that separates localisation errors from appearance errors
   * - ``putamen``
     - FreeSurfer putamen centroid (claustrum pipeline)
     - **Fails for PC.** See :ref:`fov-geometry`

Workflow D: classical baselines
-------------------------------

.. graphviz::

   digraph malf {
     rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eeeeee", fontname="Helvetica", fontsize=11];
     t [label="Target T1", shape=note, fillcolor="#ffffff"];
     a [label="29 atlases\n(leave-one-out)", shape=note, fillcolor="#ffffff"];
     r [label="Non-rigid registration\n×29"];
     w [label="Warped label maps"];
     mv [label="MAPER\nmajority vote"];
     jf [label="JLF\njoint weighted fusion"];
     t -> r; a -> r -> w; w -> mv; w -> jf;
   }

The theory is in :doc:`theory/multiatlas`.
