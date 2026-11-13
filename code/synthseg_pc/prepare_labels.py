#!/usr/bin/env python3
"""Prepare Hammers dense label maps for SynthSeg-PC (Billot GMM trainer).

Literature:
  Billot et al., Nat Commun 2023 — GMM on-the-fly from label maps; generation_* arrays.
  Mauri et al., Hum Brain Mapp 2025;46:e70303 — dense FoV scaffolding; score structure only;
    epoch pick on real T1 Dice; no claim that FS mri_synthseg has PC.
  Steinbart et al., Hum Brain Mapp 2023 — PC = Hammers 125/126; laterality → no L/R flip.
  Hammers et al., Hum Brain Mapp 2003 — n30 atlas.

Modes:
  fov (default): per-hemi 60 mm crop; remap Amyg/ATLmed/other + PC 1/2 (Dataset007 scaffold).
  wb: full-brain seg95pc (all Hammers IDs); eval still 125/126 only.

Does not print voxels. Writes counts + paths only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent
PIRIFORM = ROOT.parent
ATLAS = PIRIFORM / "Piriform_Segmentation/atlas"
SEG95PC = PIRIFORM / "Piriform_Segmentation/hammers/ancillaries/seg/seg95pc"
SPLITS = PIRIFORM / "nnunet/splits/Dataset001_Piriform.json"
CROP_HELPER = Path(
    "/path/to/clau_seg/tools/bin/crop_to_fov"
)

# Dense FoV remap (same scaffolding as Dataset007 / Mauri neighbors)
REMAP = {125: 1, 126: 2, 3: 3, 4: 4, 5: 5, 6: 6}
# generation_classes: L/R of same tissue share GMM intensity class (Billot); PC still distinct seg IDs
FOV_GENERATION_LABELS = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
FOV_GENERATION_CLASSES = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
FOV_SEGMENTATION_LABELS = FOV_GENERATION_LABELS.copy()  # keep dense loss; score 1/2 only
FOV_EVAL_LABELS = np.array([1, 2], dtype=np.int32)
PC_NEED = {"lh": 125, "rh": 126}


def sphere_mask(shape, affine, center_ras, radius_mm=8.0) -> np.ndarray:
    ijk = np.array(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"))
    ijk = np.stack(ijk, axis=-1).reshape(-1, 3)
    xyz = nib.affines.apply_affine(affine, ijk)
    keep = np.linalg.norm(xyz - center_ras, axis=1) <= radius_mm
    out = np.zeros(shape, dtype=np.uint8)
    out.reshape(-1)[keep] = 1
    return out


def subject_paths(i: int) -> tuple[Path, Path, Path]:
    brain = ATLAS / f"a{i:02d}_BrainExtractionBrain.nii.gz"
    pc = ATLAS / f"a{i:02d}_PC_labels.nii.gz"
    dense = SEG95PC / f"a{i}.nii.gz"
    return brain, pc, dense


def hammers_id(case: str) -> int:
    # Hammers_004 → 4
    return int(case.split("_")[1])


def load_fold(fold: int) -> tuple[list[int], list[int]]:
    splits = json.loads(SPLITS.read_text())
    tr = [hammers_id(c) for c in splits[fold]["train"]]
    va = [hammers_id(c) for c in splits[fold]["val"]]
    return tr, va


def remap_dense_arr(data: np.ndarray) -> np.ndarray:
    out = np.zeros(data.shape, dtype=np.int16)
    for old, new in REMAP.items():
        out[data == old] = new
    other = (data != 0) & (out == 0)
    out[other] = 7
    return out


def crop_volume(src: Path, prior_ras: np.ndarray, dest: Path, fov_mm: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_mask = dest.with_name(dest.name + ".prior.nii.gz")
    img = nib.load(str(src))
    mask = sphere_mask(img.shape[:3], img.affine, prior_ras)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(tmp_mask))
    r = subprocess.run(
        [
            sys.executable,
            str(CROP_HELPER),
            str(src),
            str(tmp_mask),
            str(dest),
            "--threshold",
            "0.5",
            "--fov-mm",
            str(fov_mm),
            str(fov_mm),
            str(fov_mm),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    tmp_mask.unlink(missing_ok=True)
    if r.returncode != 0:
        raise SystemExit(f"crop_to_fov failed {src} -> {dest}: {r.stderr}")


def pc_centroid(pc_path: Path, lab: int) -> np.ndarray:
    img = nib.load(str(pc_path))
    data = np.asanyarray(img.dataobj)
    ijk = np.argwhere(data == lab)
    if ijk.size == 0:
        raise SystemExit(f"missing label {lab} in {pc_path.name}")
    return nib.affines.apply_affine(img.affine, ijk.mean(axis=0))


def write_npy(config: Path) -> dict:
    config.mkdir(parents=True, exist_ok=True)
    np.save(config / "generation_labels.npy", FOV_GENERATION_LABELS)
    np.save(config / "generation_classes.npy", FOV_GENERATION_CLASSES)
    np.save(config / "segmentation_labels.npy", FOV_SEGMENTATION_LABELS)
    np.save(config / "evaluation_labels.npy", FOV_EVAL_LABELS)
    meta = {
        "mode": "fov",
        "flipping": False,
        "n_neutral_labels": None,
        "generation_labels": FOV_GENERATION_LABELS.tolist(),
        "generation_classes": FOV_GENERATION_CLASSES.tolist(),
        "segmentation_labels": FOV_SEGMENTATION_LABELS.tolist(),
        "evaluation_labels": FOV_EVAL_LABELS.tolist(),
        "remap": {str(k): v for k, v in REMAP.items()},
        "other_tissue": 7,
        "cite": [
            "Billot et al. Nat Commun 2023",
            "Mauri et al. Hum Brain Mapp 2025;46:e70303",
            "Steinbart et al. Hum Brain Mapp 2023 (125/126, no L/R flip)",
        ],
    }
    (config / "label_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def write_npy_wb(config: Path, labels: np.ndarray) -> dict:
    config.mkdir(parents=True, exist_ok=True)
    # With flipping=False, order is free; keep 0 first then sorted.
    gen = labels.astype(np.int32)
    # Each label its own GMM class (conservative); L/R PC still separate seg IDs.
    classes = np.arange(len(gen), dtype=np.int32)
    # Pair known L/R hammers neighbors for shared intensity (3/4, 5/6, 125/126)
    pairs = [(3, 4), (5, 6), (125, 126)]
    lab_to_i = {int(v): i for i, v in enumerate(gen)}
    next_c = 0
    classes = np.full(len(gen), -1, dtype=np.int32)
    classes[0] = 0  # background
    next_c = 1
    used = {0}
    for a, b in pairs:
        if a in lab_to_i and b in lab_to_i:
            classes[lab_to_i[a]] = next_c
            classes[lab_to_i[b]] = next_c
            used.add(a)
            used.add(b)
            next_c += 1
    for i, v in enumerate(gen):
        if classes[i] < 0:
            classes[i] = next_c
            next_c += 1
    seg = gen.copy()
    ev = np.array([125, 126], dtype=np.int32)
    np.save(config / "generation_labels.npy", gen)
    np.save(config / "generation_classes.npy", classes)
    np.save(config / "segmentation_labels.npy", seg)
    np.save(config / "evaluation_labels.npy", ev)
    meta = {
        "mode": "wb",
        "flipping": False,
        "n_neutral_labels": None,
        "n_generation_labels": int(len(gen)),
        "evaluation_labels": ev.tolist(),
        "cite": [
            "Billot et al. Nat Commun 2023",
            "Steinbart et al. Hum Brain Mapp 2023",
        ],
    }
    (config / "label_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _dense_crop_remap(dense: Path, ras: np.ndarray, dest_lab: Path, fov_mm: float) -> dict:
    tmp_lab = dest_lab.with_suffix(".tmp.nii.gz")
    crop_volume(dense, ras, tmp_lab, fov_mm)
    rem = remap_dense_arr(np.rint(np.asanyarray(nib.load(str(tmp_lab)).dataobj)).astype(np.int16))
    img = nib.load(str(tmp_lab))
    nib.save(nib.Nifti1Image(rem, img.affine, img.header), str(dest_lab))
    tmp_lab.unlink(missing_ok=True)
    return {int(k): int(v) for k, v in zip(*np.unique(rem, return_counts=True))}


def prepare_fov(out: Path, fold: int, fov_mm: float, config_dir: Path | None = None) -> dict:
    train_ids, val_ids = load_fold(fold)
    lab_tr = out / "labels_train"
    img_va = out / "val_images"
    lab_va = out / "val_labels"
    for d in (lab_tr, img_va, lab_va):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    summary: dict = {"fold": fold, "train_ids": train_ids, "val_ids": val_ids, "cases": []}

    for i in train_ids:
        brain, pc, dense = subject_paths(i)
        for p in (brain, pc, dense):
            if not p.is_file():
                raise SystemExit(f"missing {p}")
        for hemi in ("lh", "rh"):
            name = f"Hammers_{i:03d}_{hemi}.nii.gz"
            ras = pc_centroid(pc, PC_NEED[hemi])
            counts = _dense_crop_remap(dense, ras, lab_tr / name, fov_mm)
            summary["cases"].append({"id": i, "hemi": hemi, "split": "train", "counts": counts})

    for i in val_ids:
        brain, pc, dense = subject_paths(i)
        for hemi in ("lh", "rh"):
            name = f"Hammers_{i:03d}_{hemi}.nii.gz"
            ras = pc_centroid(pc, PC_NEED[hemi])
            crop_volume(brain, ras, img_va / name.replace(".nii.gz", "_0000.nii.gz"), fov_mm)
            counts = _dense_crop_remap(dense, ras, lab_va / name, fov_mm)
            summary["cases"].append({"id": i, "hemi": hemi, "split": "val", "counts": counts})

    cfg = out.parent / "config" if config_dir is None else config_dir
    meta = write_npy(cfg)
    summary["n_train_maps"] = len(list(lab_tr.glob("*.nii.gz")))
    summary["n_val_images"] = len(list(img_va.glob("*.nii.gz")))
    summary["config"] = str(cfg)
    summary["label_meta"] = meta
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mode": "fov",
                "fold": fold,
                "n_train_maps": summary["n_train_maps"],
                "n_val_images": summary["n_val_images"],
                "config": str(cfg),
            }
        )
    )
    return summary


def prepare_wb(out: Path, fold: int, config_dir: Path | None = None) -> dict:
    train_ids, val_ids = load_fold(fold)
    lab_tr = out / "labels_train"
    img_va = out / "val_images"
    lab_va = out / "val_labels"
    for d in (lab_tr, img_va, lab_va):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    all_labs: set[int] = set()
    for i in train_ids:
        _, _, dense = subject_paths(i)
        dest = lab_tr / f"Hammers_{i:03d}.nii.gz"
        shutil.copy2(dense, dest)
        u = np.unique(np.asanyarray(nib.load(str(dest)).dataobj))
        all_labs.update(int(x) for x in u)

    for i in val_ids:
        brain, pc, dense = subject_paths(i)
        shutil.copy2(brain, img_va / f"Hammers_{i:03d}_0000.nii.gz")
        # Val GT = PC manuals only for scoring; SynthSeg validate needs full seg space —
        # use dense map and evaluation_labels=[125,126].
        shutil.copy2(dense, lab_va / f"Hammers_{i:03d}.nii.gz")

    labels = np.array(sorted(all_labs), dtype=np.int32)
    if labels[0] != 0:
        labels = np.concatenate([[0], labels])
    cfg = out.parent / "config" if config_dir is None else config_dir
    meta = write_npy_wb(cfg, labels)
    summary = {
        "mode": "wb",
        "fold": fold,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "n_train_maps": len(train_ids),
        "n_val_images": len(val_ids),
        "n_labels": int(len(labels)),
        "config": str(cfg),
        "label_meta": meta,
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("mode", "fold", "n_train_maps", "n_val_images", "n_labels")}))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("fov", "wb"), default="fov")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fov-mm", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="where to write generation_*.npy (default: <out>/../config)",
    )
    args = ap.parse_args()
    if not CROP_HELPER.is_file() and args.mode == "fov":
        raise SystemExit(f"missing crop helper {CROP_HELPER}")
    if args.mode == "fov":
        prepare_fov(args.out, args.fold, args.fov_mm, config_dir=args.config_dir)
    else:
        prepare_wb(args.out, args.fold, config_dir=args.config_dir)


if __name__ == "__main__":
    main()
