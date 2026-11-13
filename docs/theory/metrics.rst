Evaluation metrics
==================

All metrics are computed **per hemisphere**, since left and right PC are
separate clinical measurements. Let :math:`A` be the predicted voxel set and
:math:`B` the reference set for one hemisphere.

Overlap
-------

.. math::

   \mathrm{Dice}(A,B) = \frac{2|A\cap B|}{|A| + |B|}, \qquad
   \mathrm{Jaccard}(A,B) = \frac{|A\cap B|}{|A\cup B|}.

**Relation.** Write :math:`I = |A\cap B|` and :math:`U = |A\cup B|`. Then
:math:`|A| + |B| = U + I`, so :math:`D = 2I/(U+I)` and :math:`J = I/U`.
Substituting :math:`I = JU` gives

.. math::

   D = \frac{2J}{1+J}, \qquad J = \frac{D}{2-D}.

The map is monotonic, so Dice and Jaccard always **rank methods
identically**. Dice is quoted for the ML literature, and Jaccard for the PC
literature (Steinbart et al. report Jaccard).

Empty-set convention: if the reference is empty, the case is **not scored**,
rather than scoring two empty sets as Dice = 1.

Surface distance
----------------

Let :math:`\partial A` be the boundary voxels of :math:`A` (voxels with at
least one 6-connected neighbour outside :math:`A`), in physical
coordinates (mm). The average symmetric surface distance is

.. math::

   \mathrm{ASD}(A,B) = \frac12\left(
     \frac{1}{|\partial A|}\sum_{a\in\partial A}\min_{b\in\partial B}\|a-b\|
     + \frac{1}{|\partial B|}\sum_{b\in\partial B}\min_{a\in\partial A}\|a-b\|\right).

By Eq. :eq:`slab`, a one-voxel shift costs a third of the Dice on PC but
only ~1 mm of ASD. ASD therefore separates *near misses* (low Dice, ASD < 1
mm) from *misplacements* (ASD of tens of mm).

.. note::

   Implementation detail: for speed, each surface is randomly subsampled
   to at most 2000 points with a fixed seed, and nearest neighbours are
   computed by chunked brute force. The result is deterministic. Subsampling
   the *target* set can only increase a nearest-neighbour distance, so on
   large surfaces the estimate is slightly conservative (biased upward).
   One hemisphere's PC has only ~600 voxels, nearly all of them on the
   boundary, so PC surfaces are usually below the 2000-point cap and are not
   subsampled at all.

Volume
------

.. math::

   V(A) = |A|\cdot\prod_{i=1}^{3}\Delta_i, \qquad |\Delta V| = |V(A) - V(B)|,

with voxel spacings :math:`\Delta_i` in mm. Clinical PC studies use volume,
so :math:`|\Delta V|` is the most clinically relevant number.

Topology
--------

The number of connected components (``scipy.ndimage.label`` default,
i.e. 6-connectivity) :math:`n_\text{cc}`, the fraction of the
mask in the largest component, and the number of hole voxels (voxels added by
hole filling) flag anatomically implausible, fragmented outputs. The
**topology fix** keeps the largest component per label and fills holes. On a
structure that should be one connected piece, this can only remove false
positives (islands) and fill false negatives (holes).

Aggregation
-----------

The headline summary is the **hemisphere Dice**: the mean over all scored
hemispheres (both sides for whole-brain predictions, one side per crop). It is
compared against the pass/fail bands of :doc:`domain_shift`.

Statistical precision
---------------------

With :math:`n=30` and a between-subject SD of left Dice of about 0.06, the
standard error of a method mean is about :math:`0.06/\sqrt{30}\approx0.011`.
The nnU-Net vs MAPER difference in left Dice (0.049) is therefore about four
standard errors. Because every method is scored on the same brains, a paired
test is tighter still: :ref:`paired-stats` reports Wilcoxon signed-rank tests
with bootstrap confidence intervals and Holm correction.
