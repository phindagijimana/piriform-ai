# Code

A snapshot of the working scripts behind the write-up. They are shared to show the pipeline structure. They are **not** a packaged tool: paths point at the original cluster layout, and the data are not included.

## `nnunet/`

| File | Purpose |
|---|---|
| `make_nnunet_splits.py` | Locked 5-fold splits, reused across all dataset variants |
| `prepare_nnunet_crop.py` | Dataset002: 60 mm per-hemisphere crops around PC |
| `prepare_nnunet_dense_crop_dr.py` | Dataset007: crops with neighbouring labels and offline augmentation |
| `prepare_nnunet_hammers_dr.py` | Dataset006: offline domain-randomization variant |
| `nnunet_entry.py` | nnU-Net entry point with cuDNN disabled (MIG workaround) |
| `nnUNetTrainerNoMirroring.py` | Note on using nnU-Net's built-in no-mirroring trainer |
| `train_nnunet_job.sh`, `submit_nnunet_train.sh` | Slurm job and chained-resubmit helper |

## `synthseg_pc/`

| File | Purpose |
|---|---|
| `prepare_labels.py` | Builds the label maps (PC box or whole brain) for the generator |
| `train.py`, `train_job.sh`, `submit_train.sh` | SynthSeg training with flipping off; chained 12 h jobs; resume logic |
| `validate.py`, `validate_job.sh`, `score_val.py`, `submit_validate.sh` | Scores every checkpoint on real T1s and picks the best epoch |
| `apply_topology_nifti.py` | Largest-component and hole-filling post-processing |
| `predict_one.py` | Single-image inference helper |
| `maybe_synthsr.sh` | Optional SynthSR pre-step (no-op if unavailable) |
| `doctor.sh`, `paths.sh` | Environment check and path resolution (Slurm-spool safe) |

## `evaluation/`

| File | Purpose |
|---|---|
| `mauri_metrics.py` | Dice, Jaccard, ASD, volume error, topology counts |
| `score_hammers_stage_a.py` | Three-way MAPER / JLF / nnU-Net scoring on Hammers |
| `make_pc_crop.py` | Crop builder with priors (atlas, putamen, oracle) and label-ID handling |
| `warp_atlas_pc_prior.sh` | Registers the atlas PC to a subject to get a deployable crop centre |
| `qc_priorcrop.py` | Field-of-view gate: rejects crops that miss the structure |
| `score_priorcrop.py` | Cohort scoring with automatic gold-label detection (125/126 or 1/2) |

Scripts that hard-code subject lists were left out of this snapshot.
