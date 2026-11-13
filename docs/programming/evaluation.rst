Evaluation code
===============

The metric library is ``code/evaluation/mauri_metrics.py``. It implements
:doc:`../theory/metrics` with NumPy and SciPy only.

Dice
----

.. literalinclude:: ../../code/evaluation/mauri_metrics.py
   :language: python
   :pyobject: dice_mask

Surface distance
----------------

Boundary voxels are the mask minus its 6-connected erosion. Coordinates are
scaled by the voxel size, so distances are in millimetres even for
anisotropic voxels:

.. literalinclude:: ../../code/evaluation/mauri_metrics.py
   :language: python
   :pyobject: average_surface_distance_mm

Topology and the topology fix
-----------------------------

.. literalinclude:: ../../code/evaluation/mauri_metrics.py
   :language: python
   :pyobject: topology_stats

.. literalinclude:: ../../code/evaluation/mauri_metrics.py
   :language: python
   :pyobject: apply_topology_fix

Pass / fail bands
-----------------

The summariser averages all scored hemispheres and assigns the band that
was fixed before the experiment. (Upstream, ``score_one`` skips a hemisphere
whose reference is empty, so two empty masks are never counted as a perfect
score.)

.. literalinclude:: ../../code/synthseg_pc/score_val.py
   :language: python
   :pyobject: summarize

.. _label-id-bug:

Case study: a bug that produced plausible numbers
-------------------------------------------------

The cohort scorer originally assumed gold labels 125/126. ADNI manuals use
1/2, so the gold masks were read as empty and **every ADNI Dice was 0**.
That number was believable for a hard cross-site test. It was caught by
checking the gold volumes in the output table. The fix detects the
convention per file and records it in a ``gold_labels`` column:

.. code-block:: python

   # Gold PC IDs: JLF 125/126, ADNI manuals 1/2
   if int((gold == 125).sum()) + int((gold == 126).sum()) > 0:
       pc_labs = {"lh": 125, "rh": 126}
   elif int((gold == 1).sum()) + int((gold == 2).sum()) > 0:
       pc_labs = {"lh": 1, "rh": 2}
   else:
       pc_labs = {"lh": 125, "rh": 126}
       flags.append("no_gold_pc_labels")

After the fix, the affected result was still 0, now correctly attributed to
a crop that missed PC. The lesson is to make every scorer **report what it
read**, so that "zero" can be told apart from "nothing".

Unit test with a known answer
-----------------------------

``score_hammers_stage_a.py --self-test`` builds tiny synthetic masks with a
hand-computed answer (intersection 18, union 27, so Jaccard 2/3 and Dice 0.8)
and asserts the scorer reproduces it.
