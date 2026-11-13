#!/usr/bin/env python3
"""Dataset007: partial Mauri combo — dense neighbors + 60 mm FoV + stronger offline DR.

NOT full SynthSeg-PC (no GMM on-the-fly trainer). Inference prior-crop is separate
(`submit_nnunet_prior_crop.sh`). Hammers manuals only. Score PC classes 1/2 only
(`score_dataset007_mauri.py` / `mauri_metrics.py`). Doc: nnunet/tables/dataset007_dense_crop_dr.md.

Label remap inside each crop (Hammers n30r95 IDs):
  125→1 PC_L, 126→2 PC_R,
  3→3 Amyg_L, 4→4 Amyg_R,
  5→5 ATLmed_L, 6→6 ATLmed_R,
  any other nonzero→7 other_tissue (Mauri-style fill / WM buffer proxy).

Per-hemi 60 mm around manual PC (same as Dataset002). Offline DR: 5 augs/case
(stronger than Dataset006's 3). Val = originals only.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path("/path/to/piriform")
ATLAS = ROOT / "Piriform_Segmentation/atlas"
SEG95 = ROOT / "Piriform_Segmentation/hammers/ancillaries/seg/seg95pc"
OUT = ROOT / "nnunet/nnUNet_raw/Dataset007_PiriformDenseCropDR"
LOCKED1 = ROOT / "nnunet/splits/Dataset001_Piriform.json"
LOCKED7 = ROOT / "nnunet/splits/Dataset007_PiriformDenseCropDR.json"
CROP_HELPER = Path(
    "/path/to/clau_seg/tools/bin/crop_to_fov"
)
PC = {"lh": 125, "rh": 126}
FOV_MM = 60.0
N_AUG = 5  # stronger than Dataset006 (3)
SEED = 2027
# Hammers n30r95 → consecutive training IDs
REMAP = {
    125: 1,
    126: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
}


def sphere_mask(shape, affine, center_ras, radius_mm=8.0) -> np.ndarray:
    ijk = np.array(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"))
    ijk = np.stack(ijk, axis=-1).reshape(-1, 3)
    xyz = nib.affines.apply_affine(affine, ijk)
    keep = np.linalg.norm(xyz - center_ras, axis=1) <= radius_mm
    out = np.zeros(shape, dtype=np.uint8)
    out.reshape(-1)[keep] = 1
    return out


def corrupt(td: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stronger photometric + resolution synth than Dataset006 (wider gamma, more low-res)."""
    out = td.astype(np.float32).copy()
    brain = out > 0
    if not brain.any():
        return out
    g = float(rng.uniform(0.5, 1.8))
    lo, hi = np.percentile(out[brain], (1, 99))
    span = max(hi - lo, 1e-3)
    x = np.clip((out[brain] - lo) / span, 0, 1)
    out[brain] = (np.power(x, g) * span + lo).astype(np.float32)
    if rng.random() < 0.95:
        from scipy.ndimage import zoom

        z = rng.normal(0, 1, size=[max(s // 16, 1) for s in out.shape]).astype(np.float32)
        factors = [out.shape[i] / z.shape[i] for i in range(3)]
        field = zoom(z, factors, order=1)
        field = field - field[brain].mean()
        field = 1.0 + 0.35 * field / (np.std(field[brain]) + 1e-3)
        out[brain] *= field[brain]
    if rng.random() < 0.9:
        sig = 0.03 * float(np.percentile(out[brain], 99))
        out[brain] += rng.normal(0, sig, size=int(brain.sum())).astype(np.float32)
    if rng.random() < 0.7:
        from scipy.ndimage import zoom

        factor = float(rng.uniform(0.35, 0.7))
        small = zoom(out, factor, order=1)
        out = zoom(small, [out.shape[i] / small.shape[i] for i in range(3)], order=1).astype(
            np.float32
        )
        out2 = np.zeros_like(td, dtype=np.float32)
        slices = tuple(slice(0, min(out.shape[i], td.shape[i])) for i in range(3))
        out2[slices] = out[slices]
        out = out2
        out[~brain] = 0
    out = np.clip(out, 0, None)
    out[~brain] = 0
    return out


def crop_pair(brain: Path, labels: Path, ras: np.ndarray, dest_brain: Path, dest_lab: Path) -> None:
    tmp_mask = dest_brain.with_name(dest_brain.name + ".prior.nii.gz")
    img = nib.load(str(brain))
    mask = sphere_mask(img.shape[:3], img.affine, ras)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(tmp_mask))
    for src, dst in ((brain, dest_brain), (labels, dest_lab)):
        r = subprocess.run(
            [
                sys.executable,
                str(CROP_HELPER),
                str(src),
                str(tmp_mask),
                str(dst),
                "--threshold",
                "0.5",
                "--fov-mm",
                str(FOV_MM),
                str(FOV_MM),
                str(FOV_MM),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise SystemExit(f"crop_to_fov failed {src} -> {dst}: {r.stderr}")
    tmp_mask.unlink(missing_ok=True)


def remap_dense(src: Path, dst: Path, need_pc: int) -> dict[int, int]:
    img = nib.load(str(src))
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int16)
    out = np.zeros(data.shape, dtype=np.int16)
    counts: dict[int, int] = {}
    for old, new in REMAP.items():
        m = data == old
        out[m] = new
        counts[new] = int(m.sum())
    # other nonzero tissue → class 7
    known = set(REMAP.keys()) | {0}
    other = (data != 0) & ~np.isin(data, list(known))
    out[other] = 7
    counts[7] = int(other.sum())
    if counts.get(need_pc, 0) == 0:
        raise SystemExit(f"{src.name}: crop dropped PC class {need_pc}")
    hdr = img.header.copy()
    hdr.set_data_dtype(np.int16)
    nib.Nifti1Image(out, img.affine, hdr).to_filename(str(dst))
    return counts


def write_splits() -> None:
    sa = json.loads(LOCKED1.read_text())
    splits = []
    for fold in sa:
        val_bases = sorted(fold["val"])
        train_bases = sorted(set(fold["train"]) - set(val_bases))
        val_cases = [f"{b}_{h}" for b in val_bases for h in ("lh", "rh")]
        train_cases = []
        for b in train_bases:
            for h in ("lh", "rh"):
                base = f"{b}_{h}"
                train_cases.append(base)
                for k in range(1, N_AUG + 1):
                    train_cases.append(f"{base}_aug{k}")
        splits.append({"train": sorted(train_cases), "val": sorted(val_cases)})
    text = json.dumps(splits, indent=4) + "\n"
    LOCKED7.parent.mkdir(parents=True, exist_ok=True)
    LOCKED7.write_text(text)
    pre = ROOT / "nnunet/nnUNet_preprocessed/Dataset007_PiriformDenseCropDR"
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "splits_final.json").write_text(text)
    for i, sp in enumerate(splits):
        print(f"fold {i}: train={len(sp['train'])} val={len(sp['val'])}")


def main() -> int:
    if not CROP_HELPER.is_file():
        raise SystemExit(f"missing {CROP_HELPER}")
    try:
        import scipy.ndimage  # noqa: F401
    except ImportError as e:
        raise SystemExit("scipy required") from e

    if OUT.exists():
        shutil.rmtree(OUT)
    images = OUT / "imagesTr"
    labels_dir = OUT / "labelsTr"
    images.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    rng = np.random.default_rng(SEED)
    manifest = []
    for i in range(1, 31):
        aid_atlas = f"a{i:02d}"
        aid_seg = f"a{i}"
        case0 = f"Hammers_{i:03d}"
        brain = ATLAS / f"{aid_atlas}_BrainExtractionBrain.nii.gz"
        seg = SEG95 / f"{aid_seg}.nii.gz"
        if not brain.is_file() or not seg.is_file():
            raise SystemExit(f"missing {brain} or {seg}")
        gold = nib.load(str(seg))
        gd = np.asanyarray(gold.dataobj)
        for hemi, labid in PC.items():
            idx = np.argwhere(gd == labid)
            if idx.size == 0:
                raise SystemExit(f"{aid_seg} missing {labid}")
            ras = nib.affines.apply_affine(gold.affine, idx.mean(0))
            case = f"{case0}_{hemi}"
            tmp_lab = labels_dir / f"{case}_rawcrop.nii.gz"
            dest_b = images / f"{case}_0000.nii.gz"
            dest_l = labels_dir / f"{case}.nii.gz"
            crop_pair(brain, seg, ras, dest_b, tmp_lab)
            need = 1 if hemi == "lh" else 2
            counts = remap_dense(tmp_lab, dest_l, need)
            tmp_lab.unlink(missing_ok=True)
            print(f"{case} counts={counts} RAS={np.round(ras, 2).tolist()}")
            manifest.append({"case": case, "base": case, "aug": 0})
            td = np.asanyarray(nib.load(str(dest_b)).dataobj).astype(np.float32)
            img = nib.load(str(dest_b))
            for k in range(1, N_AUG + 1):
                name = f"{case}_aug{k}"
                out = corrupt(td, rng)
                hdr = img.header.copy()
                hdr.set_data_dtype(np.float32)
                nib.Nifti1Image(out, img.affine, hdr).to_filename(
                    str(images / f"{name}_0000.nii.gz")
                )
                shutil.copy2(dest_l, labels_dir / f"{name}.nii.gz")
                manifest.append({"case": name, "base": case, "aug": k})

    dataset = {
        "channel_names": {"0": "T1"},
        "labels": {
            "background": 0,
            "PC_left": 1,
            "PC_right": 2,
            "Amyg_left": 3,
            "Amyg_right": 4,
            "ATLmed_left": 5,
            "ATLmed_right": 6,
            "other_tissue": 7,
        },
        "numTraining": len(manifest),
        "file_ending": ".nii.gz",
        "licence": "Hammers n30r95 + Steinbart PC; research use",
        "dataset_name": "Dataset007_PiriformDenseCropDR",
        "reference": "Partial Mauri combo: dense FoV + offline DR (not GMM SynthSeg-PC)",
        "description": (
            f"Per-hemi 60 mm crop; dense neighbors + other_tissue; "
            f"{N_AUG} offline augs each. Val=originals. Score PC 1/2 only. "
            "Not inference prior-crop; not full SynthSeg-PC."
        ),
    }
    (OUT / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    (OUT / "aug_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_splits()
    print(f"wrote {len(manifest)} cases -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
