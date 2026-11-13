# piriform-ai: training deep-learning models to segment the piriform cortex

[![docs](https://github.com/phindagijimana/piriform-ai/actions/workflows/docs.yml/badge.svg)](https://phindagijimana.github.io/piriform-ai/) [![tests](https://github.com/phindagijimana/piriform-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/phindagijimana/piriform-ai/actions/workflows/tests.yml)

**Philbert Ndagijimana** · [phindagiji@gmail.com](mailto:phindagiji@gmail.com) · [github.com/phindagijimana](https://github.com/phindagijimana)

**Documentation: <https://phindagijimana.github.io/piriform-ai/>** · Short on time? Read [For reviewers: the five-minute version](https://phindagijimana.github.io/piriform-ai/reviewers.html).

The piriform cortex is a ~0.5 cm³, 2–3 mm thick strip of olfactory cortex; in temporal lobe epilepsy, how much of it is removed during surgery predicts seizure freedom ([Galovic et al., 2019](https://doi.org/10.1001/jamaneurol.2019.0204)). The only published automatic method, multi-atlas MAPER, reaches Jaccard ≈ 0.50 against manual tracing ([Steinbart et al., 2023](https://doi.org/10.1002/hbm.26274)). This project asks whether a trained 3D CNN can beat it, and whether it still works on scans from other sites.

## Results

| Question | Answer |
|---|---|
| Can nnU-Net learn the structure from 30 labelled brains? | **Yes.** Dice 0.71 / 0.69 (L/R) vs MAPER 0.66 / 0.63 and JLF 0.66 / 0.61 on the same held-out brains; the overlap gain is significant (paired Wilcoxon, Holm-corrected p ≤ 0.003). |
| Does it transfer to a second site (ADNI)? | **No.** Dice stayed at 0.02–0.04; controlled follow-ups ruled out causes one at a time, and the branch was stopped at a threshold set in advance. |
| Does synthetic-image training (SynthSeg) fix the site dependence? | **Not yet.** Non-empty masks without intensity matching, but Dice 0.47, below the bar. |

![Stage A results](figures/hammers_three_way.png)

## Repository

```
docs/      Sphinx source for the documentation site (theory, code walkthroughs, experiment log)
code/      nnU-Net dataset builders and training, SynthSeg-PC, inference, evaluation (see code/README.md)
tests/     known-answer tests for the metric library, run in CI
figures/   aggregate plots and the scripts that make them
```

```bash
pip install numpy scipy nibabel pytest && python -m pytest -q tests        # tests
pip install -r docs/requirements.txt && sphinx-build -W -b html docs docs/_build/html   # docs
```

The repository holds code, methods, and aggregate results only. Imaging data, labels, model weights, and per-subject results are governed by data-use agreements and are not included.

## Citation

Ndagijimana P. *piriform-ai: training deep-learning models to segment the piriform cortex on brain MRI.* 2026. https://github.com/phindagijimana/piriform-ai ([BibTeX and upstream methods to cite](https://phindagijimana.github.io/piriform-ai/citing.html); [`CITATION.cff`](CITATION.cff)).
