SynthSeg-PC training
====================

Training wrapper
----------------

``code/synthseg_pc/train.py`` wraps ``SynthSeg.training.training``. It
validates its inputs, records the full recipe to ``train_run.json``, and
**refuses to train with flipping**:

.. literalinclude:: ../../code/synthseg_pc/train.py
   :language: python
   :start-at: meta = json.loads(meta_p.read_text())
   :end-before: args.model_dir.mkdir

The call into SynthSeg maps each documented generator parameter
(:doc:`../theory/synthseg`) to an argument:

.. literalinclude:: ../../code/synthseg_pc/train.py
   :language: python
   :start-at:     training(
   :end-before: if __name__

Recipe selection and resume
---------------------------

The Slurm worker ``train_job.sh`` picks the model directory from
``RECIPE``, so a wider v2 network never tries to load a v1 checkpoint:

.. literalinclude:: ../../code/synthseg_pc/train_job.sh
   :language: bash
   :start-at: RECIPE="${RECIPE:-v1}"
   :end-before: mkdir -p "${MODEL_DIR}"

On resume, the newest ``dice_*.h5`` is found with a version sort
(``dice_010`` after ``dice_009``). If it is a Dice checkpoint, the L2
warm-up is skipped (:doc:`../theory/losses`):

.. literalinclude:: ../../code/synthseg_pc/train_job.sh
   :language: bash
   :start-at: CKPT=""
   :end-before: EXTRA=()

Per-recipe hyperparameters are environment defaults, and any of them can be
overridden at submit time:

.. literalinclude:: ../../code/synthseg_pc/train_job.sh
   :language: bash
   :start-at: # v2 defaults
   :end-before: export PYTHONPATH

Epoch selection on real images
------------------------------

``code/synthseg_pc/validate.py`` predicts every ``dice_*.h5`` on the real
held-out Hammers crops at **native resolution** (``target_res=None``). An
early version resampled to 1 mm and produced a :math:`64^3` vs :math:`61^3`
shape mismatch. It then scores each epoch and keeps the best by hemisphere
Dice (:ref:`epoch-selection`):

.. literalinclude:: ../../code/synthseg_pc/validate.py
   :language: python
   :start-at: # pick best by hemi Dice
   :end-before: if args.tables_prefix

Each crop contains one side only, so the scorer scores only the side named in
the file:

.. literalinclude:: ../../code/synthseg_pc/score_val.py
   :language: python
   :pyobject: hemi_labs
