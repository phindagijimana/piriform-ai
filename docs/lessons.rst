Lessons learned
===============

Training models
---------------

1. **Build the baseline first, and use it to test the evaluation code.**
   Reproducing MAPER's published number made every later score trustworthy.
2. **The label protocol beats the architecture.** The most important training
   decision, disabling mirroring, was a one-flag change derived from anatomy
   (:ref:`mirroring-proof`).
3. **A strong default is a strong baseline.** nnU-Net out of the box beat 15
   years of multi-atlas engineering in-domain.
4. **Complexity is not free with small data.** Dataset007 did not beat
   Dataset002.
5. **Understand your metric's ceiling.** Eq. :eq:`slab` shows why PC Dice
   saturates near 0.7, and why ASD must be reported alongside it.

Generalisation
--------------

6. **"Fails on a new site" is a symptom, not a diagnosis.** Each failure on
   the target cohort had a testable cause: crop geometry, skull-stripping, or
   intensity range.
7. **Check the input before blaming the model.** A crop that contains 0% of
   the structure cannot pass, and a geometric gate catches that in seconds.
8. **Design controls that rule out one hypothesis each.** The ADNI-only and
   oracle-crop experiments each eliminated a whole class of explanations.
9. **Synthetic training trades in-domain accuracy for robustness.**

Running the project
-------------------

10. **Set thresholds before looking at results.**
11. **A negative result is a result.** Document it with its evidence.
12. **Silent bugs are the dangerous ones.** Make every scorer report what it
    read. Make every step fail on empty output.
13. **Infrastructure is ML work.** The cuDNN incident cost more time than
    any modelling decision.
14. **Dry-run and review before spending compute.**

What I would do differently
---------------------------

* Collect 10–20 manual labels on the **target** cohort early. That is the only
  way to turn "agrees with JLF" into "is accurate".
* Audit per-site intensity statistics on day one.
* Report bootstrap confidence intervals and paired tests (e.g. Wilcoxon
  signed-rank on per-brain Dice) for every comparison.
* Choose crop priors per structure from the start.

Responsible handling of clinical data
-------------------------------------

* Imaging data stayed on-premises under the data-use agreements. No
  images, overlays, or identifiers were given to external AI services. Image
  analysis ran in local scripts that report aggregate numbers.
* Results on the target cohort are labelled as agreement with a reference
  method, not clinical accuracy.
* The validated classical pipeline remains in production until a learned
  model is validated against target-cohort manual labels.
