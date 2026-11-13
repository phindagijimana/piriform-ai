#!/usr/bin/env python3
"""Pick SynthSeg-PC epoch on real Hammers T1 Dice (PC classes only).

Mauri et al. 2025: train on synthetic; select epoch on real T1 Dice.
Steinbart: score remapped PC 1/2. No test-time L/R flip.

Uses native resolution (target_res=None) so FoV crops (~0.937 mm Hammers)
are not forced to 1 mm (which broke smoke val: 64³ vs 61³).
After SynthSeg predict, scores with resample-safe score_val.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SYNTHSEG = Path(
    "/path/to/clau_seg/tools/SynthSeg"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=ROOT / "data/val_images")
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data/val_labels")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models/fold0")
    ap.add_argument("--validation-dir", type=Path, default=ROOT / "models/fold0_val")
    ap.add_argument("--config", type=Path, default=ROOT / "config")
    ap.add_argument("--step-eval", type=int, default=1)
    ap.add_argument("--cropping", type=int, default=None, help="optional net crop size")
    ap.add_argument(
        "--synthseg-root",
        type=Path,
        default=Path(os.environ.get("SYNTHSEG_ROOT", DEFAULT_SYNTHSEG)),
    )
    ap.add_argument(
        "--tables-prefix",
        type=Path,
        default=None,
        help="optional tables/synthseg_pc_* prefix for best-epoch summary",
    )
    ap.add_argument(
        "--topology-fix",
        action="store_true",
        help="Mauri topology fix before PC Dice/ASD (lever 6)",
    )
    args = ap.parse_args()

    ss = args.synthseg_root.resolve()
    sys.path.insert(0, str(ss))

    seg = args.config / "segmentation_labels.npy"
    ev = args.config / "evaluation_labels.npy"
    for p in (seg, ev, args.image_dir, args.gt_dir, args.models_dir):
        if not Path(p).exists():
            raise SystemExit(f"missing {p}")

    args.validation_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "image_dir": str(args.image_dir),
        "gt_dir": str(args.gt_dir),
        "models_dir": str(args.models_dir),
        "evaluation_labels": str(ev),
        "flip": False,
        "target_res": None,
        "cite": "Mauri et al. Hum Brain Mapp 2025 — epoch on real T1; score PC only",
    }
    (args.validation_dir / "validate_run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))

    from SynthSeg import predict as ss_predict  # noqa: E402
    from ext.lab2im import utils  # noqa: E402

    list_models = utils.list_files(str(args.models_dir), expr=["dice", ".h5"], cond_type="and")[
        :: args.step_eval
    ]
    if not list_models:
        raise SystemExit(f"no dice_*.h5 in {args.models_dir}")

    import numpy as np

    labels_seg = np.load(str(seg))

    epoch_rows = []
    for path_model in list_models:
        tag = Path(path_model).stem  # dice_002
        out_dir = args.validation_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"predict {tag} -> {out_dir}", flush=True)
        ss_predict.predict(
            path_images=str(args.image_dir),
            path_segmentations=str(out_dir),
            path_model=path_model,
            labels_segmentation=labels_seg,
            n_neutral_labels=None,
            target_res=None,
            flip=False,
            cropping=args.cropping,
            keep_biggest_component=False,
            gt_folder=None,  # score ourselves (resample-safe)
        )
        prefix = args.validation_dir / f"{tag}_pc"
        score_cmd = [
            sys.executable,
            str(ROOT / "score_val.py"),
            "--pred-dir",
            str(out_dir),
            "--gt-dir",
            str(args.gt_dir),
            "--out-prefix",
            str(prefix),
        ]
        if args.topology_fix:
            score_cmd.append("--topology-fix")
        subprocess.run(score_cmd, check=True)
        summary = json.loads(prefix.with_suffix(".json").read_text())["summary"]
        summary["model"] = tag
        epoch_rows.append(summary)
        print(json.dumps(summary), flush=True)

    # pick best by hemi Dice (FoV crops score one side each)
    valid = [r for r in epoch_rows if r.get("dice_hemi") is not None]
    best = max(valid, key=lambda r: r["dice_hemi"]) if valid else None
    out = {"epochs": epoch_rows, "best": best, "meta": meta}
    (args.validation_dir / "epoch_pick.json").write_text(json.dumps(out, indent=2) + "\n")
    if args.tables_prefix is not None:
        args.tables_prefix.parent.mkdir(parents=True, exist_ok=True)
        args.tables_prefix.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"best": best}))


if __name__ == "__main__":
    main()
