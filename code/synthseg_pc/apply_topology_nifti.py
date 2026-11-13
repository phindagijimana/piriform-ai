#!/usr/bin/env python3
"""Mauri topology fix on a stitched PC pred (labels 1/2)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "beyondfs" / "modules"))
from mauri_metrics import apply_topology_fix  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    img = nib.load(str(args.pred))
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int16)
    fixed = apply_topology_fix(data, (1, 2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(fixed, img.affine, img.header), str(args.out))
    print(f"topology_fix wrote {args.out}")


if __name__ == "__main__":
    main()
