#!/usr/bin/env python3
"""Run SynthSeg-PC on one FoV crop folder (CPU-safe)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--seg-labels", type=Path, required=True)
    ap.add_argument("--synthseg-root", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.synthseg_root))
    from SynthSeg import predict as ss_predict  # noqa: E402

    args.out.mkdir(parents=True, exist_ok=True)
    ss_predict.predict(
        path_images=str(args.images),
        path_segmentations=str(args.out),
        path_model=str(args.model),
        labels_segmentation=np.load(str(args.seg_labels)),
        n_neutral_labels=None,
        target_res=None,
        flip=False,
        cropping=None,
        keep_biggest_component=False,
        gt_folder=None,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
