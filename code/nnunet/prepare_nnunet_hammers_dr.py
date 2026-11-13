#!/usr/bin/env python3
"""Dataset006: Hammers-only + offline intensity/resolution synth-aug (domain-randomized).

Scientifically: train on Hammers manuals only; ADNI is an *external* unseen-site test.
Each real (T1, label) pair is kept; aug variants corrupt the T1 only (same label).

Does not touch Dataset001 weights or Dataset005 ADNI-only.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path("/path/to/piriform")
SRC = ROOT / "nnunet/nnUNet_raw/Dataset001_Piriform"
OUT = ROOT / "nnunet/nnUNet_raw/Dataset006_PiriformHammersDR"
LOCKED_A = ROOT / "nnunet/splits/Dataset001_Piriform.json"
# Stage A also used Dataset001_PiriformHammers naming in some trees
LOCKED_ALT = ROOT / "nnunet/splits/Dataset001_PiriformHammers.json"
N_AUG = 3  # plus original → 4 views / brain
SEED = 2026


def corrupt(td: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric + mild resolution synth on brain voxels (label geometry untouched)."""
    out = td.astype(np.float32).copy()
    brain = out > 0
    if not brain.any():
        return out

    # gamma
    g = float(rng.uniform(0.6, 1.6))
    lo, hi = np.percentile(out[brain], (1, 99))
    span = max(hi - lo, 1e-3)
    x = np.clip((out[brain] - lo) / span, 0, 1)
    out[brain] = (np.power(x, g) * span + lo).astype(np.float32)

    # smooth multiplicative bias (very low-freq)
    if rng.random() < 0.9:
        z = rng.normal(0, 1, size=[max(s // 16, 1) for s in out.shape]).astype(np.float32)
        # upsample with zoom
        from scipy.ndimage import zoom

        factors = [out.shape[i] / z.shape[i] for i in range(3)]
        field = zoom(z, factors, order=1)
        field = field - field[brain].mean()
        field = 1.0 + 0.25 * field / (np.std(field[brain]) + 1e-3)
        out[brain] *= field[brain]

    # noise
    if rng.random() < 0.85:
        sig = 0.02 * float(np.percentile(out[brain], 99))
        out[brain] += rng.normal(0, sig, size=int(brain.sum())).astype(np.float32)

    # simulate low-res then upsample
    if rng.random() < 0.5:
        from scipy.ndimage import zoom

        factor = float(rng.uniform(0.4, 0.75))
        small = zoom(out, factor, order=1)
        out = zoom(small, [out.shape[i] / small.shape[i] for i in range(3)], order=1).astype(
            np.float32
        )
        # zoom can change shape by 1 voxel — crop/pad
        out2 = np.zeros_like(td, dtype=np.float32)
        slices = tuple(slice(0, min(out.shape[i], td.shape[i])) for i in range(3))
        out2[slices] = out[slices]
        out = out2
        out[~brain] = 0

    out = np.clip(out, 0, None)
    out[~brain] = 0
    return out


def load_stage_a_splits() -> list[dict]:
    for p in (LOCKED_A, LOCKED_ALT):
        if p.is_file():
            return json.loads(p.read_text())
    # fall back to Dataset001 preprocessed
    pre = ROOT / "nnunet/nnUNet_preprocessed/Dataset001_Piriform/splits_final.json"
    if pre.is_file():
        return json.loads(pre.read_text())
    raise SystemExit("no Stage A locked splits found")


def main() -> int:
    if not SRC.is_dir():
        raise SystemExit(f"missing {SRC}")
    try:
        import scipy.ndimage  # noqa: F401
    except ImportError as e:
        raise SystemExit("scipy required for bias/low-res aug") from e

    if OUT.exists():
        shutil.rmtree(OUT)
    images = OUT / "imagesTr"
    labels = OUT / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    rng = np.random.default_rng(SEED)
    bases = sorted((SRC / "labelsTr").glob("Hammers_*.nii.gz"))
    if len(bases) != 30:
        raise SystemExit(f"expected 30 Hammers, got {len(bases)}")

    manifest = []
    for lab in bases:
        case = lab.name.replace(".nii.gz", "")
        img_p = SRC / "imagesTr" / f"{case}_0000.nii.gz"
        img = nib.load(str(img_p))
        td = np.asanyarray(img.dataobj).astype(np.float32)
        # original
        shutil.copy2(img_p, images / f"{case}_0000.nii.gz")
        shutil.copy2(lab, labels / f"{case}.nii.gz")
        manifest.append({"case": case, "base": case, "aug": 0})
        for k in range(1, N_AUG + 1):
            name = f"{case}_aug{k}"
            out = corrupt(td, rng)
            hdr = img.header.copy()
            hdr.set_data_dtype(np.float32)
            nib.Nifti1Image(out, img.affine, hdr).to_filename(str(images / f"{name}_0000.nii.gz"))
            shutil.copy2(lab, labels / f"{name}.nii.gz")
            manifest.append({"case": name, "base": case, "aug": k})

    dataset = {
        "channel_names": {"0": "T1"},
        "labels": {"background": 0, "PC_left": 1, "PC_right": 2},
        "numTraining": len(manifest),
        "file_ending": ".nii.gz",
        "licence": "Hammers Steinbart PC manuals; research use",
        "dataset_name": "Dataset006_PiriformHammersDR",
        "reference": "Domain-randomized Hammers (synth aug); ADNI external test only",
        "description": (
            "30 Hammers originals + 3 photometric/resolution augs each (120 cases). "
            "Labels unchanged. Not multi-site; ADNI must not enter imagesTr."
        ),
    }
    (OUT / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    (OUT / "aug_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Splits from Stage A: val = originals only; train = train bases + their augs
    sa = load_stage_a_splits()
    splits = []
    for fold in sa:
        val_bases = [c for c in fold["val"] if not c.endswith(("_aug1", "_aug2", "_aug3"))]
        # strip any accidental aug names
        val_bases = [c.split("_aug")[0] if "_aug" in c else c for c in val_bases]
        val_bases = sorted(set(val_bases))
        train_bases = sorted(set(fold["train"]) - set(val_bases))
        train_bases = [c.split("_aug")[0] if "_aug" in c else c for c in train_bases]
        train_bases = sorted(set(train_bases) - set(val_bases))
        train_cases = []
        for b in train_bases:
            train_cases.append(b)
            for k in range(1, N_AUG + 1):
                train_cases.append(f"{b}_aug{k}")
        splits.append({"train": sorted(train_cases), "val": sorted(val_bases)})

    pre = ROOT / "nnunet/nnUNet_preprocessed/Dataset006_PiriformHammersDR"
    pre.mkdir(parents=True, exist_ok=True)
    text = json.dumps(splits, indent=4) + "\n"
    (pre / "splits_final.json").write_text(text)
    lock = ROOT / "nnunet/splits/Dataset006_PiriformHammersDR.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(text)
    for i, sp in enumerate(splits):
        print(f"fold {i}: train={len(sp['train'])} val={len(sp['val'])} val={sp['val']}")
    print(f"wrote {len(manifest)} cases -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
